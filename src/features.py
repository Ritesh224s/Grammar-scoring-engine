"""Grammar scoring feature extraction module.

Combines linguistic, syntactic, and audio delivery features to predict continuous
English grammar proficiency scores:
- Linguistic breadth, lexical diversity (TTR, Root TTR, Herdan's C), and length statistics.
- Syntactic complexity proxies (subordination index, conjunctions, clause boundary density).
- Grammatical function words (personal/possessive/relative pronouns, modal & auxiliary verbs, prepositions).
- Hesitation and disfluency rates (overt pauses/fillers vs discourse markers, immediate word repetitions).
- Acoustic delivery & speaking rate proxies (RMS dynamics, ZCR, silence ratio, WPM, phonation rate).
- Split-aware, leakage-free processing with strict separation of identifiers, labels, and features.
"""

from __future__ import annotations

import json
import math
import re
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from .audio import AudioSignal, compute_silence_stats, load_audio
from .config import AppConfig, get_config
from .dataset import load_and_prepare_dataset
from .transcribe import load_transcripts_cache
from .utils import get_logger

logger = get_logger("features")

# -----------------------------------------------------------------------------
# Curated Lexicons for Rule-Based Linguistic & Syntactic Extraction
# (Lightweight, CPU-friendly, zero external model downloads required)
# -----------------------------------------------------------------------------
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
}

# Overt hesitation vocalizations
FILLER_DISFLUENCIES = {"um", "uh", "er", "ah", "hmm", "umm", "uhh"}

# Colloquial discourse words and multi-word phrases
FILLER_DISCOURSE_WORDS = {"like", "basically", "actually", "literally"}
FILLER_DISCOURSE_PHRASES = ["you know", "i mean", "sort of", "kind of"]

# Pronoun categories
PERSONAL_PRONOUNS = {
    "i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them",
}
POSSESSIVE_PRONOUNS = {
    "my", "mine", "your", "yours", "his", "her", "hers", "its", "our", "ours", "their", "theirs",
}
RELATIVE_PRONOUNS = {"who", "whom", "whose", "which", "that"}

# Verb auxiliary & modal indicators
MODAL_VERBS = {
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought",
}
AUXILIARY_VERBS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
}

# Conjunctions (crucial for clause structure and syntactic complexity)
SUBORDINATING_CONJUNCTIONS = {
    "because", "although", "though", "since", "unless", "while", "whereas",
    "if", "whether", "after", "before", "as", "until", "whenever", "wherever",
}
COORDINATING_CONJUNCTIONS = {"and", "but", "or", "nor", "for", "so", "yet"}

# Common prepositions
PREPOSITIONS = {
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "of", "off", "over", "under", "without", "within",
}

# Morphological affix indicators
NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship")
ADJ_SUFFIXES = ("able", "ible", "al", "ful", "ic", "ive", "less", "ous")
NON_ADVERB_LY_WORDS = {"friendly", "lovely", "lonely", "silly", "ugly", "early", "daily", "weekly", "monthly"}
NON_PROGRESSIVE_ING_WORDS = {"thing", "something", "anything", "nothing", "everything", "spring", "ring", "king", "wing", "sing"}

# -----------------------------------------------------------------------------
# Metadata Columns and Model Feature Definitions
# -----------------------------------------------------------------------------
METADATA_COLUMNS = [
    "sample_id",
    "split",
    "filename",
    "extraction_status",
    "extraction_error",
]

# Explicit, ordered list of all 72 numerical features
FEATURE_COLUMNS: list[str] = [
    # 1. Length & Basic Counts (4)
    "word_count",
    "sentence_count",
    "char_count",
    "char_count_nospace",
    # 2. Length Statistics (5)
    "avg_word_length",
    "avg_sentence_length_words",
    "min_sentence_length_words",
    "max_sentence_length_words",
    "std_sentence_length_words",
    # 3. Vocabulary Richness & Diversity (6)
    "long_word_count",
    "long_word_ratio",
    "unique_word_count",
    "ttr",
    "root_ttr",
    "log_ttr",
    # 4. Stop Words & Fluency (10)
    "stop_word_count",
    "stop_word_ratio",
    "repeated_word_count",
    "repeated_word_ratio",
    "filler_disfluency_count",
    "filler_disfluency_rate",
    "filler_discourse_count",
    "filler_discourse_rate",
    "total_filler_count",
    "total_filler_rate",
    # 5. Punctuation & Clause Boundaries (7)
    "punctuation_count",
    "punctuation_ratio",
    "comma_count",
    "period_count",
    "question_count",
    "exclamation_count",
    "clause_boundary_ratio",
    # 6. Pronouns & Function Words (6)
    "personal_pronoun_count",
    "personal_pronoun_ratio",
    "possessive_pronoun_count",
    "possessive_pronoun_ratio",
    "relative_pronoun_count",
    "relative_pronoun_ratio",
    # 7. Verb Auxiliary & Modal Structure (4)
    "modal_verb_count",
    "modal_verb_ratio",
    "auxiliary_verb_count",
    "auxiliary_verb_ratio",
    # 8. Syntactic Subordination & Coordination (5)
    "subordinating_conjunction_count",
    "subordinating_conjunction_ratio",
    "coordinating_conjunction_count",
    "coordinating_conjunction_ratio",
    "subordination_index",
    # 9. Prepositions (2)
    "preposition_count",
    "preposition_ratio",
    # 10. Morphological Complexity Signals (10)
    "noun_suffix_count",
    "noun_suffix_ratio",
    "adjective_suffix_count",
    "adjective_suffix_ratio",
    "adverb_suffix_count",
    "adverb_suffix_ratio",
    "past_tense_marker_count",
    "past_tense_marker_ratio",
    "progressive_marker_count",
    "progressive_marker_ratio",
    # 11. Acoustic Properties (9)
    "audio_duration_s",
    "audio_rms_mean",
    "audio_rms_std",
    "audio_rms_peak",
    "audio_zcr_mean",
    "audio_zcr_std",
    "audio_silence_ratio",
    "audio_speech_ratio",
    "audio_speech_duration_s",
    # 12. Multimodal Fluency & Rate Proxies (4)
    "speaking_rate_wps",
    "speaking_rate_wpm",
    "phonation_rate_wps",
    "articulation_pace",
]


def get_feature_columns() -> list[str]:
    """Return a copy of the list of model feature column names."""
    return list(FEATURE_COLUMNS)


def get_feature_descriptions() -> dict[str, dict[str, str]]:
    """Return comprehensive descriptions and category mappings for all features."""
    descriptions: dict[str, dict[str, str]] = {}
    for col in FEATURE_COLUMNS:
        category = "Linguistic"
        desc = col
        if "audio" in col:
            category = "Audio"
            desc = f"Acoustic delivery feature: {col}"
        elif any(k in col for k in ["speaking_rate", "phonation", "articulation"]):
            category = "Multimodal Fluency"
            desc = f"Rate proxy aligning text words with audio duration: {col}"
        elif any(k in col for k in ["pronoun", "modal", "auxiliary", "conjunction", "subordination", "preposition", "suffix", "tense", "progressive"]):
            category = "Grammar & Syntax"
            desc = f"Grammar/syntactic structure proxy: {col}"
        elif any(k in col for k in ["filler", "repeated"]):
            category = "Disfluency"
            desc = f"Speech hesitation / repetition indicator: {col}"
        elif any(k in col for k in ["ttr", "unique", "long_word"]):
            category = "Lexical Diversity"
            desc = f"Vocabulary breadth metric: {col}"
        descriptions[col] = {"category": category, "name": col}
    return descriptions


# -----------------------------------------------------------------------------
# Part A & B: Text & Linguistic Feature Extraction
# -----------------------------------------------------------------------------
def tokenize_words(text: str) -> list[str]:
    """Extract lowercase alphanumeric word tokens from text."""
    if not text:
        return []
    # Match standard English words including contractions (e.g. don't, family's)
    tokens = re.findall(r"\b[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\b", text.lower())
    return tokens


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentence strings using sentence boundary punctuation."""
    if not text or not text.strip():
        return []
    # Split on periods, exclamation marks, question marks, or multiple newlines
    raw_sentences = re.split(r"[.!?]+|\n+", text)
    sentences = [s.strip() for s in raw_sentences if any(c.isalnum() for c in s)]
    return sentences


def extract_text_features(text: str | None) -> dict[str, float | int]:
    """Extract linguistic, syntactic, and morphological features from raw text.

    Safely handles empty strings, whitespace, punctuation-only, and malformed inputs.
    All mathematical operations protect against zero-division errors.
    """
    raw_text = (text or "").strip()
    words = tokenize_words(raw_text)
    sentences = tokenize_sentences(raw_text)

    word_count = len(words)
    sentence_count = len(sentences)
    char_count = len(raw_text)
    char_count_nospace = len(re.sub(r"\s+", "", raw_text))

    # Sentence length statistics in words
    if sentence_count > 0:
        sent_word_counts = [len(tokenize_words(s)) for s in sentences]
        # Filter out 0-word sentences if any
        sent_word_counts = [c for c in sent_word_counts if c > 0] or [0]
        avg_sentence_len = float(np.mean(sent_word_counts))
        min_sentence_len = int(np.min(sent_word_counts))
        max_sentence_len = int(np.max(sent_word_counts))
        std_sentence_len = float(np.std(sent_word_counts))
    else:
        avg_sentence_len = 0.0
        min_sentence_len = 0
        max_sentence_len = 0
        std_sentence_len = 0.0

    # Word lengths
    if word_count > 0:
        avg_word_length = float(np.mean([len(w) for w in words]))
        long_words = [w for w in words if len(w) >= 7]
        long_word_count = len(long_words)
        long_word_ratio = round(long_word_count / word_count, 5)

        # Lexical diversity
        unique_words = set(words)
        unique_word_count = len(unique_words)
        ttr = round(unique_word_count / word_count, 5)
        root_ttr = round(unique_word_count / math.sqrt(word_count), 5)
        log_ttr = round(math.log(max(unique_word_count, 1)) / math.log(max(word_count, 2)), 5) if word_count > 1 else 0.0

        # Stop words
        stop_words_found = [w for w in words if w in STOP_WORDS]
        stop_word_count = len(stop_words_found)
        stop_word_ratio = round(stop_word_count / word_count, 5)

        # Repeated words (adjacent repetitions: "the the", "I I")
        repeated_word_count = sum(1 for i in range(word_count - 1) if words[i] == words[i + 1])
        repeated_word_ratio = round(repeated_word_count / word_count, 5)

        # Filler words & disfluencies
        filler_disfluency_count = sum(1 for w in words if w in FILLER_DISFLUENCIES)
        filler_disfluency_rate = round(filler_disfluency_count / word_count, 5)

        # Discourse markers ("like", "you know", etc.)
        discourse_words = sum(1 for w in words if w in FILLER_DISCOURSE_WORDS)
        lower_raw = raw_text.lower()
        discourse_phrases = sum(lower_raw.count(phrase) for phrase in FILLER_DISCOURSE_PHRASES)
        filler_discourse_count = discourse_words + discourse_phrases
        filler_discourse_rate = round(filler_discourse_count / word_count, 5)

        total_filler_count = filler_disfluency_count + filler_discourse_count
        total_filler_rate = round(total_filler_count / word_count, 5)

        # Punctuation counts
        punct_count = sum(1 for c in raw_text if c in string.punctuation)
        punct_ratio = round(punct_count / word_count, 5)
        comma_count = raw_text.count(",")
        period_count = raw_text.count(".")
        question_count = raw_text.count("?")
        exclamation_count = raw_text.count("!")
        clause_boundary_ratio = round((comma_count + period_count) / word_count, 5)

        # Pronouns
        pers_pron = sum(1 for w in words if w in PERSONAL_PRONOUNS)
        poss_pron = sum(1 for w in words if w in POSSESSIVE_PRONOUNS)
        rel_pron = sum(1 for w in words if w in RELATIVE_PRONOUNS)

        # Verbs (modals, auxiliaries)
        modal_verbs = sum(1 for w in words if w in MODAL_VERBS)
        aux_verbs = sum(1 for w in words if w in AUXILIARY_VERBS)

        # Conjunctions & Clause Subordination
        sub_conj = sum(1 for w in words if w in SUBORDINATING_CONJUNCTIONS)
        coord_conj = sum(1 for w in words if w in COORDINATING_CONJUNCTIONS)
        total_conj = sub_conj + coord_conj
        sub_index = round(sub_conj / total_conj, 5) if total_conj > 0 else 0.0

        # Prepositions
        preps = sum(1 for w in words if w in PREPOSITIONS)

        # Morphological Suffixes
        noun_suffixes = sum(1 for w in words if any(w.endswith(s) for s in NOUN_SUFFIXES) and len(w) > 4)
        adj_suffixes = sum(1 for w in words if any(w.endswith(s) for s in ADJ_SUFFIXES) and len(w) > 4)
        adv_suffixes = sum(1 for w in words if w.endswith("ly") and len(w) > 3 and w not in NON_ADVERB_LY_WORDS)
        past_tense = sum(1 for w in words if w.endswith("ed") and len(w) > 3)
        progressive = sum(1 for w in words if w.endswith("ing") and len(w) > 4 and w not in NON_PROGRESSIVE_ING_WORDS)

    else:
        avg_word_length = 0.0
        long_word_count = 0
        long_word_ratio = 0.0
        unique_word_count = 0
        ttr = 0.0
        root_ttr = 0.0
        log_ttr = 0.0
        stop_word_count = 0
        stop_word_ratio = 0.0
        repeated_word_count = 0
        repeated_word_ratio = 0.0
        filler_disfluency_count = 0
        filler_disfluency_rate = 0.0
        filler_discourse_count = 0
        filler_discourse_rate = 0.0
        total_filler_count = 0
        total_filler_rate = 0.0
        punct_count = sum(1 for c in raw_text if c in string.punctuation)
        punct_ratio = 0.0
        comma_count = raw_text.count(",")
        period_count = raw_text.count(".")
        question_count = raw_text.count("?")
        exclamation_count = raw_text.count("!")
        clause_boundary_ratio = 0.0
        pers_pron = 0
        poss_pron = 0
        rel_pron = 0
        modal_verbs = 0
        aux_verbs = 0
        sub_conj = 0
        coord_conj = 0
        sub_index = 0.0
        preps = 0
        noun_suffixes = 0
        adj_suffixes = 0
        adv_suffixes = 0
        past_tense = 0
        progressive = 0

    denom = max(word_count, 1)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": char_count,
        "char_count_nospace": char_count_nospace,
        "avg_word_length": round(avg_word_length, 4),
        "avg_sentence_length_words": round(avg_sentence_len, 4),
        "min_sentence_length_words": min_sentence_len,
        "max_sentence_length_words": max_sentence_len,
        "std_sentence_length_words": round(std_sentence_len, 4),
        "long_word_count": long_word_count,
        "long_word_ratio": long_word_ratio,
        "unique_word_count": unique_word_count,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "log_ttr": log_ttr,
        "stop_word_count": stop_word_count,
        "stop_word_ratio": stop_word_ratio,
        "repeated_word_count": repeated_word_count,
        "repeated_word_ratio": repeated_word_ratio,
        "filler_disfluency_count": filler_disfluency_count,
        "filler_disfluency_rate": filler_disfluency_rate,
        "filler_discourse_count": filler_discourse_count,
        "filler_discourse_rate": filler_discourse_rate,
        "total_filler_count": total_filler_count,
        "total_filler_rate": total_filler_rate,
        "punctuation_count": punct_count,
        "punctuation_ratio": punct_ratio,
        "comma_count": comma_count,
        "period_count": period_count,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "clause_boundary_ratio": clause_boundary_ratio,
        "personal_pronoun_count": pers_pron,
        "personal_pronoun_ratio": round(pers_pron / denom, 5),
        "possessive_pronoun_count": poss_pron,
        "possessive_pronoun_ratio": round(poss_pron / denom, 5),
        "relative_pronoun_count": rel_pron,
        "relative_pronoun_ratio": round(rel_pron / denom, 5),
        "modal_verb_count": modal_verbs,
        "modal_verb_ratio": round(modal_verbs / denom, 5),
        "auxiliary_verb_count": aux_verbs,
        "auxiliary_verb_ratio": round(aux_verbs / denom, 5),
        "subordinating_conjunction_count": sub_conj,
        "subordinating_conjunction_ratio": round(sub_conj / denom, 5),
        "coordinating_conjunction_count": coord_conj,
        "coordinating_conjunction_ratio": round(coord_conj / denom, 5),
        "subordination_index": sub_index,
        "preposition_count": preps,
        "preposition_ratio": round(preps / denom, 5),
        "noun_suffix_count": noun_suffixes,
        "noun_suffix_ratio": round(noun_suffixes / denom, 5),
        "adjective_suffix_count": adj_suffixes,
        "adjective_suffix_ratio": round(adj_suffixes / denom, 5),
        "adverb_suffix_count": adv_suffixes,
        "adverb_suffix_ratio": round(adv_suffixes / denom, 5),
        "past_tense_marker_count": past_tense,
        "past_tense_marker_ratio": round(past_tense / denom, 5),
        "progressive_marker_count": progressive,
        "progressive_marker_ratio": round(progressive / denom, 5),
    }


# -----------------------------------------------------------------------------
# Part C: Audio Feature Extraction
# -----------------------------------------------------------------------------
def extract_audio_features(
    signal_or_path: AudioSignal | Path | str,
    target_sr: int = 16000,
    silence_top_db: float = 30.0,
) -> dict[str, float]:
    """Extract acoustic delivery features from an AudioSignal or WAV file path."""
    if isinstance(signal_or_path, AudioSignal):
        sig = signal_or_path
    else:
        sig = load_audio(signal_or_path, target_sr=target_sr, mono=True)

    samples = sig.samples
    sr = sig.sample_rate
    duration = sig.duration_s

    if len(samples) == 0 or duration <= 0:
        return {
            "audio_duration_s": 0.0,
            "audio_rms_mean": 0.0,
            "audio_rms_std": 0.0,
            "audio_rms_peak": 0.0,
            "audio_zcr_mean": 0.0,
            "audio_zcr_std": 0.0,
            "audio_silence_ratio": 1.0,
            "audio_speech_ratio": 0.0,
            "audio_speech_duration_s": 0.0,
        }

    # Frame-level RMS energy
    frame_length = min(2048, len(samples))
    hop_length = 512
    rms_frames = librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_std = float(np.std(rms_frames)) if len(rms_frames) > 0 else 0.0

    # Frame-level Zero-Crossing Rate (articulation transitions)
    zcr_frames = librosa.feature.zero_crossing_rate(y=samples, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr_frames)) if len(zcr_frames) > 0 else 0.0
    zcr_std = float(np.std(zcr_frames)) if len(zcr_frames) > 0 else 0.0

    # Silence analysis
    silence_stats = compute_silence_stats(samples, sr, silence_top_db=silence_top_db)
    silence_ratio = silence_stats["silence_ratio"]
    speech_ratio = silence_stats["speech_ratio"]
    speech_duration = silence_stats["speech_duration_s"]

    return {
        "audio_duration_s": round(duration, 4),
        "audio_rms_mean": round(rms_mean, 5),
        "audio_rms_std": round(rms_std, 5),
        "audio_rms_peak": round(sig.peak_amplitude, 5),
        "audio_zcr_mean": round(zcr_mean, 5),
        "audio_zcr_std": round(zcr_std, 5),
        "audio_silence_ratio": round(silence_ratio, 4),
        "audio_speech_ratio": round(speech_ratio, 4),
        "audio_speech_duration_s": round(speech_duration, 4),
    }


def compute_speaking_rate_features(
    text_feat: dict[str, Any],
    audio_feat: dict[str, Any],
) -> dict[str, float]:
    """Align word counts with acoustic durations to compute delivery speed proxies."""
    word_count = text_feat.get("word_count", 0)
    duration = audio_feat.get("audio_duration_s", 0.0)
    speech_duration = audio_feat.get("audio_speech_duration_s", 0.0)

    # Words per second (overall)
    wps = round(word_count / duration, 4) if duration > 0.05 else 0.0
    wpm = round(wps * 60.0, 2)

    # Words per active phonation second
    phonation_wps = round(word_count / speech_duration, 4) if speech_duration > 0.05 else 0.0

    # Pace: seconds per word
    pace = round(duration / word_count, 4) if word_count > 0 else 0.0

    return {
        "speaking_rate_wps": wps,
        "speaking_rate_wpm": wpm,
        "phonation_rate_wps": phonation_wps,
        "articulation_pace": pace,
    }


# -----------------------------------------------------------------------------
# Part D: End-to-End Sample and Batch Extraction
# -----------------------------------------------------------------------------
def extract_single_sample_features(
    sample_id: str,
    split: str,
    filename: str,
    audio_path: Path | str,
    transcript: str | None,
    transcript_status: str = "ok",
    transcript_error: str = "",
    label: float | None = None,
    target_sr: int = 16000,
) -> dict[str, Any]:
    """Extract full feature record for a single sample.

    Guarantees:
    - If transcript failed or missing, status is set accordingly and features contain NaN.
    - If audio fails to load, status is set to 'failed_audio' and features contain NaN.
    - If successful, returns all 72 features populated as finite floats.
    """
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "filename": filename,
        "extraction_status": "ok",
        "extraction_error": "",
    }
    if label is not None:
        record["label"] = float(label)

    # 1. Check transcript health
    if transcript_status != "ok" or transcript is None:
        record["extraction_status"] = f"missing_transcript_{transcript_status}"
        record["extraction_error"] = transcript_error or f"Transcript status was {transcript_status}"
        for col in FEATURE_COLUMNS:
            record[col] = np.nan
        return record

    # 2. Extract text features
    try:
        text_feat = extract_text_features(transcript)
    except Exception as exc:
        record["extraction_status"] = "failed_text_extraction"
        record["extraction_error"] = str(exc)
        for col in FEATURE_COLUMNS:
            record[col] = np.nan
        return record

    # 3. Extract audio features
    try:
        audio_feat = extract_audio_features(audio_path, target_sr=target_sr)
    except Exception as exc:
        record["extraction_status"] = "failed_audio_extraction"
        record["extraction_error"] = str(exc)
        for col in FEATURE_COLUMNS:
            record[col] = np.nan
        return record

    # 4. Compute multimodal speaking rates
    rate_feat = compute_speaking_rate_features(text_feat, audio_feat)

    # 5. Assemble record
    record.update(text_feat)
    record.update(audio_feat)
    record.update(rate_feat)

    return record


def extract_features_for_dataframe(
    dataset_df: pd.DataFrame,
    transcripts_cache_df: pd.DataFrame,
    split: str,
    config: AppConfig | None = None,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Extract features for all samples (or filtered sample_ids) in a dataset split.

    Args:
        dataset_df: Prepared dataset DataFrame (contains sample_id, filename, audio_path).
        transcripts_cache_df: Loaded transcripts cache DataFrame.
        split: 'train' or 'test'.
        config: AppConfig instance.
        sample_ids: Optional list of specific sample_id strings to process (e.g. for smoke test).

    Returns:
        DataFrame containing metadata, labels (for train only), and consistent feature columns.
    """
    cfg = config or get_config()
    target_df = dataset_df.copy()

    if sample_ids is not None:
        target_df = target_df[target_df["sample_id"].isin(sample_ids)].reset_index(drop=True)

    # Index cache by sample_id for fast, collision-safe lookups
    cache_lookup: dict[str, dict[str, Any]] = {}
    if not transcripts_cache_df.empty and "sample_id" in transcripts_cache_df.columns:
        for _, c_row in transcripts_cache_df.iterrows():
            cache_lookup[c_row["sample_id"]] = c_row.to_dict()

    rows: list[dict[str, Any]] = []
    is_train = split == "train"

    for _, row in target_df.iterrows():
        sid = row["sample_id"]
        fname = row["filename"]
        apath = Path(row["audio_path"])
        lbl = row.get("label", None) if is_train else None

        cached = cache_lookup.get(sid, {})
        t_text = cached.get("transcript", None)
        t_status = cached.get("status", "missing_cache")
        t_err = cached.get("error", "")

        rec = extract_single_sample_features(
            sample_id=sid,
            split=split,
            filename=fname,
            audio_path=apath,
            transcript=t_text,
            transcript_status=t_status,
            transcript_error=t_err,
            label=lbl,
            target_sr=cfg.target_sample_rate,
        )
        rows.append(rec)

    # Build DataFrame with explicit column order
    df = pd.DataFrame(rows)

    # Column ordering: Metadata -> (label if train) -> FEATURE_COLUMNS
    cols = list(METADATA_COLUMNS)
    if is_train and "label" in df.columns:
        cols.append("label")
    for f in FEATURE_COLUMNS:
        if f not in cols and f in df.columns:
            cols.append(f)

    # Ensure all required feature columns exist
    for f in FEATURE_COLUMNS:
        if f not in df.columns:
            df[f] = np.nan

    df = df[cols]
    return df


def separate_features_and_metadata(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Separate a feature DataFrame into (X_features, y_target_or_none, metadata_df).

    Guarantees:
    - X_features contains ONLY the 72 mathematical features.
    - sample_id, split, filename, status, and labels are completely excluded from X.
    """
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feat_cols].copy()

    y = df["label"].copy() if "label" in df.columns else None

    meta_cols = [c for c in df.columns if c not in feat_cols and c != "label"]
    meta = df[meta_cols].copy()

    return X, y, meta


# -----------------------------------------------------------------------------
# Reporting and Diagnostics
# -----------------------------------------------------------------------------
def generate_feature_extraction_report(
    train_features_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    config: AppConfig | None = None,
    save_files: bool = True,
) -> tuple[dict[str, Any], str]:
    """Generate diagnostic report, Markdown summary, and plots for extracted features."""
    cfg = config or get_config()
    cfg.ensure_directories()

    # Schema consistency validation
    train_feats = [c for c in FEATURE_COLUMNS if c in train_features_df.columns]
    test_feats = [c for c in FEATURE_COLUMNS if c in test_features_df.columns]
    schemas_match = (train_feats == test_feats) and (len(train_feats) == len(FEATURE_COLUMNS))

    # Check for constant features across train features
    constant_features: list[str] = []
    if len(train_features_df) > 1:
        for f in train_feats:
            vals = train_features_df[f].dropna()
            if len(vals) > 1 and float(vals.std()) < 1e-6:
                constant_features.append(f)

    # Missing value statistics
    train_missing = int(train_features_df[train_feats].isna().sum().sum())
    test_missing = int(test_features_df[test_feats].isna().sum().sum())

    train_valid = int((train_features_df["extraction_status"] == "ok").sum())
    train_failed = int(len(train_features_df) - train_valid)
    test_valid = int((test_features_df["extraction_status"] == "ok").sum())
    test_failed = int(len(test_features_df) - test_valid)

    # Exclusions verification
    id_leakage_in_features = any(
        col in FEATURE_COLUMNS for col in ["sample_id", "filename", "relative_path", "split", "label"]
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_records_processed": {
            "train": len(train_features_df),
            "test": len(test_features_df),
            "total": len(train_features_df) + len(test_features_df),
        },
        "extraction_status": {
            "train_valid": train_valid,
            "train_failed": train_failed,
            "test_valid": test_valid,
            "test_failed": test_failed,
        },
        "feature_count": len(FEATURE_COLUMNS),
        "schemas_match": schemas_match,
        "id_or_label_in_features": id_leakage_in_features,
        "missing_values": {
            "train_missing_cells": train_missing,
            "test_missing_cells": test_missing,
        },
        "constant_features": constant_features,
        "external_nlp_pos_model_available": False,
        "feature_names": FEATURE_COLUMNS,
    }

    # Generate Markdown Summary
    summary_md = f"""# Feature Extraction Pipeline Report

**Generated**: `{report['timestamp']}`  
**Total Features Extracted**: `{report['feature_count']}`  
**Train / Test Feature Schema Match**: `{'PASSED' if schemas_match else 'FAILED'}`  
**Target & Identifier Exclusion**: `{'PASSED (Zero Leakage)' if not id_leakage_in_features else 'FAILED'}`  

---

## 1. Processed Samples Overview

| Split | Processed | Valid (`ok`) | Failed | Missing Feature Values |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | {len(train_features_df)} | {train_valid} | {train_failed} | {train_missing} |
| **Test** | {len(test_features_df)} | {test_valid} | {test_failed} | {test_missing} |
| **Total** | **{len(train_features_df) + len(test_features_df)}** | **{train_valid + test_valid}** | **{train_failed + test_failed}** | **{train_missing + test_missing}** |

---

## 2. Feature Categories & Definitions

The 72 features span 5 distinct complementary domains:

1. **Linguistic Length & Statistics (9)**: Word count, sentence count, char count, avg word length, avg/min/max/std sentence lengths.
2. **Lexical Diversity & Breadth (6)**: Long word count & ratio ($\\ge 7$ chars), unique words, Type-Token Ratio (TTR), Root TTR (Guiraud's Index), Herdan's log TTR.
3. **Disfluency & Hesitation Signals (10)**: Stop words ratio, immediately repeated words ratio, overt fillers (`um`, `uh`), discourse markers (`like`, `you know`), total filler rate.
4. **Grammar & Syntactic Complexity (30)**:
   - Punctuation & clause boundary ratios (commas, periods, question marks).
   - Personal, possessive, and relative pronouns.
   - Auxiliary verbs & modal verbs.
   - Subordinating vs. coordinating conjunctions & subordination index.
   - Preposition counts and proportions.
   - Morphological derivational suffixes (nominalizations `-tion`, `-ment`; adjectives `-able`, `-ive`; adverbs `-ly`; past tense `-ed`; progressive `-ing`).
5. **Acoustic & Multimodal Fluency (17)**: Audio duration, RMS energy dynamics, zero-crossing rates, silence & speech ratios, words-per-minute (WPM), phonation rate, and articulation pace.

---

## 3. Constant or Low-Variance Features
{f"- **Constant Features Identified**: `{constant_features}`" if constant_features else "- No constant features detected across valid samples."}

---

## 4. NLP Dependency Note
- **External POS Models (e.g. Spacy/NLTK)**: `Not Installed / Bypassed`.
- Feature extraction utilizes deterministic, CPU-efficient rule-based parsing and curated lexicons. No heavyweight downloads or external model dependencies are required.
"""

    if save_files:
        # Save JSON Report
        with open(cfg.feature_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        # Save Markdown Summary
        with open(cfg.feature_summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)

        # Generate Diagnostic Plots
        _generate_feature_plots(train_features_df, test_features_df, cfg)

    return report, summary_md


def _generate_feature_plots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: AppConfig,
) -> None:
    """Generate diagnostic distributions and correlation heatmap plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive, headless backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        key_cols = [
            "word_count",
            "speaking_rate_wpm",
            "ttr",
            "audio_silence_ratio",
            "subordination_index",
            "avg_sentence_length_words",
        ]
        available_cols = [c for c in key_cols if c in combined_df.columns]

        # 1. Distributions Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for i, col in enumerate(available_cols):
            ax = axes[i]
            train_vals = train_df[col].dropna()
            test_vals = test_df[col].dropna()

            if len(train_vals) > 0:
                ax.hist(train_vals, bins=10, alpha=0.6, label="Train", color="#1f77b4", density=False)
            if len(test_vals) > 0:
                ax.hist(test_vals, bins=10, alpha=0.6, label="Test", color="#ff7f0e", density=False)

            ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle("Feature Distributions (Smoke Test Samples)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(cfg.feature_distributions_plot, dpi=150)
        plt.close(fig)

        # 2. Correlation Plot
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr_cols = available_cols + (["label"] if "label" in train_df.columns else [])
        corr_matrix = train_df[corr_cols].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax_corr,
            vmin=-1.0,
            vmax=1.0,
        )
        ax_corr.set_title("Feature Correlations (Train Samples)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(cfg.feature_correlation_plot, dpi=150)
        plt.close(fig_corr)

    except Exception as exc:
        logger.warning("Could not generate feature diagnostic plots: %s", exc)


# -----------------------------------------------------------------------------
# Part E: Smoke Test Runner
# -----------------------------------------------------------------------------
SMOKE_TEST_TARGETS = [
    ("train", "audio_192.wav"),
    ("train", "audio_436.wav"),
    ("test", "audio_128.wav"),
    ("test", "audio_107.wav"),
]


def run_feature_smoke_test(config: AppConfig | None = None) -> dict[str, Any]:
    """Execute feature extraction strictly on the 4 smoke-test samples.

    Targets:
    - train/audio_192.wav
    - train/audio_436.wav
    - test/audio_128.wav
    - test/audio_107.wav
    """
    cfg = config or get_config()
    cfg.ensure_directories()

    logger.info("Starting Phase 5 Feature Extraction Smoke Test...")
    train_df, test_df, _, _ = load_and_prepare_dataset(cfg)
    cache_path = cfg.get_transcripts_cache_path()
    transcripts_cache = load_transcripts_cache(cache_path)

    train_sample_ids = [f"train/{fname}" for split, fname in SMOKE_TEST_TARGETS if split == "train"]
    test_sample_ids = [f"test/{fname}" for split, fname in SMOKE_TEST_TARGETS if split == "test"]

    logger.info("Extracting features for train smoke targets: %s", train_sample_ids)
    train_feat_df = extract_features_for_dataframe(
        dataset_df=train_df,
        transcripts_cache_df=transcripts_cache,
        split="train",
        config=cfg,
        sample_ids=train_sample_ids,
    )

    logger.info("Extracting features for test smoke targets: %s", test_sample_ids)
    test_feat_df = extract_features_for_dataframe(
        dataset_df=test_df,
        transcripts_cache_df=transcripts_cache,
        split="test",
        config=cfg,
        sample_ids=test_sample_ids,
    )

    # Save output feature tables
    train_feat_df.to_csv(cfg.train_features_path, index=False)
    test_feat_df.to_csv(cfg.test_features_path, index=False)

    # Generate full report, markdown summary, and diagnostic plots
    report, summary_md = generate_feature_extraction_report(
        train_features_df=train_feat_df,
        test_features_df=test_feat_df,
        config=cfg,
        save_files=True,
    )

    print("\n" + "=" * 76)
    print("FEATURE EXTRACTION SMOKE TEST SUMMARY")
    print("=" * 76)
    print(f"Train Smoke Records: {len(train_feat_df)} | Saved: {cfg.train_features_path.name}")
    print(f"Test Smoke Records : {len(test_feat_df)} | Saved: {cfg.test_features_path.name}")
    print(f"Feature Columns    : {report['feature_count']} features")
    print(f"Schema Match       : {'PASSED' if report['schemas_match'] else 'FAILED'}")
    print(f"Zero Target Leakage: {'CONFIRMED' if not report['id_or_label_in_features'] else 'FAILED'}")
    print("-" * 76)

    for idx, row in pd.concat([train_feat_df, test_feat_df], ignore_index=True).iterrows():
        lbl_str = f" | Label: {row['label']:.2f}" if "label" in row and not pd.isna(row["label"]) else ""
        print(f"[{row['split'].upper()}] {row['sample_id']}{lbl_str}")
        print(f"      Status: {row['extraction_status']} | Duration: {row['audio_duration_s']}s | Words: {row['word_count']} | WPM: {row['speaking_rate_wpm']}")
        print(f"      TTR: {row['ttr']} | Subordination Index: {row['subordination_index']} | Silence Ratio: {row['audio_silence_ratio']}")

    print("=" * 76 + "\n")
    return report


def extract_full_dataset_features(config: AppConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract features for the entire dataset (all 769 train and 216 test records).

    Guarantees:
    - Train table has 769 rows with label column.
    - Test table has 216 rows without any dummy labels.
    - Both tables have identical 72 numerical feature columns.
    - Explicit status and error logging for every record.
    - Saves train_features.csv, test_features.csv, report, markdown summary, and plots.
    """
    cfg = config or get_config()
    cfg.ensure_directories()

    logger.info("Starting Full Dataset Feature Extraction (769 train + 216 test)...")
    train_df, test_df, _, _ = load_and_prepare_dataset(cfg)
    cache_path = cfg.get_transcripts_cache_path()
    transcripts_cache = load_transcripts_cache(cache_path)

    logger.info("Extracting features for %d training recordings...", len(train_df))
    train_feat_df = extract_features_for_dataframe(
        dataset_df=train_df,
        transcripts_cache_df=transcripts_cache,
        split="train",
        config=cfg,
    )

    logger.info("Extracting features for %d test recordings...", len(test_df))
    test_feat_df = extract_features_for_dataframe(
        dataset_df=test_df,
        transcripts_cache_df=transcripts_cache,
        split="test",
        config=cfg,
    )

    # Save output feature tables
    train_feat_df.to_csv(cfg.train_features_path, index=False)
    test_feat_df.to_csv(cfg.test_features_path, index=False)
    logger.info("Saved train features to %s (%d rows)", cfg.train_features_path, len(train_feat_df))
    logger.info("Saved test features to %s (%d rows)", cfg.test_features_path, len(test_feat_df))

    # Generate full report, markdown summary, and diagnostic plots
    report, summary_md = generate_feature_extraction_report(
        train_features_df=train_feat_df,
        test_features_df=test_feat_df,
        config=cfg,
        save_files=True,
    )

    print("\n" + "=" * 76)
    print("FULL DATASET FEATURE EXTRACTION COMPLETE")
    print("=" * 76)
    print(f"Train Records: {len(train_feat_df)} | Saved: {cfg.train_features_path.name}")
    print(f"Test Records : {len(test_feat_df)} | Saved: {cfg.test_features_path.name}")
    print(f"Feature Count: {report['feature_count']} features")
    print(f"Schema Match : {'PASSED' if report['schemas_match'] else 'FAILED'}")
    print(f"Zero Leakage : {'CONFIRMED' if not report['id_or_label_in_features'] else 'FAILED'}")
    print("=" * 76 + "\n")

    return train_feat_df, test_feat_df, report


def main() -> int:
    """CLI entrypoint for feature extraction."""
    import argparse
    parser = argparse.ArgumentParser(description="Feature extraction for speech grammar scoring.")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test on 4 clips")
    parser.add_argument("--all", action="store_true", help="Extract features for full dataset (all 769 train + 216 test)")
    args = parser.parse_args()

    cfg = get_config()
    if args.all:
        _, _, report = extract_full_dataset_features(cfg)
        return 0 if report.get("schemas_match", False) else 1

    report = run_feature_smoke_test(cfg)
    return 0 if report.get("schemas_match", False) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
