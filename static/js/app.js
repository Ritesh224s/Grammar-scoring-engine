// ==========================================================================
// SHL Grammar Scoring Engine — Frontend Controller
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const audioInput = document.getElementById("audioInput");
  const uploadForm = document.getElementById("uploadForm");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const filePreviewBar = document.getElementById("filePreviewBar");
  const selectedFileName = document.getElementById("selectedFileName");
  const selectedFileSize = document.getElementById("selectedFileSize");
  const audioPreview = document.getElementById("audioPreview");
  const clearFileBtn = document.getElementById("clearFileBtn");
  
  const loadingSection = document.getElementById("loadingSection");
  const loadingStepTitle = document.getElementById("loadingStepTitle");
  const loadingStepDesc = document.getElementById("loadingStepDesc");
  const stepItems = [
    document.getElementById("step1"),
    document.getElementById("step2"),
    document.getElementById("step3"),
    document.getElementById("step4")
  ];
  
  const errorBanner = document.getElementById("errorBanner");
  const errorTitle = document.getElementById("errorTitle");
  const errorMessage = document.getElementById("errorMessage");
  const closeErrorBtn = document.getElementById("closeErrorBtn");
  
  const resultsSection = document.getElementById("resultsSection");
  const scoreDisplay = document.getElementById("scoreDisplay");
  const gaugeFill = document.getElementById("gaugeFill");
  const boundsMarker = document.getElementById("boundsMarker");
  const proficiencyTag = document.getElementById("proficiencyTag");
  const percentileText = document.getElementById("percentileText");
  const proficiencyDesc = document.getElementById("proficiencyDesc");
  const evalSpeedBadge = document.getElementById("evalSpeedBadge");
  const transcriptText = document.getElementById("transcriptText");
  const copyTranscriptBtn = document.getElementById("copyTranscriptBtn");
  
  // Metrics
  const metricWpm = document.getElementById("metricWpm");
  const metricTtr = document.getElementById("metricTtr");
  const metricWordCount = document.getElementById("metricWordCount");
  const metricSentenceCount = document.getElementById("metricSentenceCount");
  const metricSubordination = document.getElementById("metricSubordination");
  const metricFillers = document.getElementById("metricFillers");
  const metricSilenceRatio = document.getElementById("metricSilenceRatio");
  const metricDuration = document.getElementById("metricDuration");
  
  const samplesList = document.getElementById("samplesList");

  let currentSelectedFile = null;
  let activeSampleName = null;

  // -------------------------------------------------------------------------
  // 1. Load Curated Benchmark Samples
  // -------------------------------------------------------------------------
  async function loadSamples() {
    try {
      const resp = await fetch("/api/samples");
      const data = await resp.json();
      if (data.samples && data.samples.length > 0) {
        samplesList.innerHTML = "";
        data.samples.forEach(sample => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "sample-btn";
          btn.innerHTML = `
            <div class="sample-meta">
              <span class="sample-title">${sample.title}</span>
              <span class="sample-tier">${sample.description}</span>
            </div>
            <span class="sample-arrow">Play & Score ➔</span>
          `;
          btn.addEventListener("click", () => handleSampleSelect(sample));
          samplesList.appendChild(btn);
        });
      } else {
        samplesList.innerHTML = "<p class='drop-hint'>No sample clips available.</p>";
      }
    } catch (err) {
      console.warn("Could not load samples:", err);
      samplesList.innerHTML = "<p class='drop-hint'>Curated samples unavailable.</p>";
    }
  }

  // -------------------------------------------------------------------------
  // 2. Drag & Drop and File Selection Handlers
  // -------------------------------------------------------------------------
  ["dragenter", "dragover"].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  audioInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith(".wav")) {
      showError("Invalid File Format", "Please select a standard audio recording with a .wav file extension.");
      return;
    }

    currentSelectedFile = file;
    activeSampleName = null;

    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);
    audioPreview.src = URL.createObjectURL(file);
    filePreviewBar.style.display = "flex";
    analyzeBtn.disabled = false;
    hideError();
  }

  clearFileBtn.addEventListener("click", () => {
    currentSelectedFile = null;
    activeSampleName = null;
    audioInput.value = "";
    audioPreview.src = "";
    filePreviewBar.style.display = "none";
    analyzeBtn.disabled = true;
  });

  // -------------------------------------------------------------------------
  // 3. Sample Click Handler
  // -------------------------------------------------------------------------
  function handleSampleSelect(sample) {
    activeSampleName = sample.filename;
    currentSelectedFile = null;
    audioInput.value = "";

    selectedFileName.textContent = `${sample.title} (${sample.filename})`;
    selectedFileSize.textContent = "Dataset benchmark sample";
    audioPreview.src = `/api/samples/${sample.filename}`;
    filePreviewBar.style.display = "flex";
    analyzeBtn.disabled = false;
    hideError();

    // Trigger analysis immediately for quick demo
    submitEvaluation();
  }

  // -------------------------------------------------------------------------
  // 4. Form Submit & Inference Call
  // -------------------------------------------------------------------------
  uploadForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitEvaluation();
  });

  async function submitEvaluation() {
    if (!currentSelectedFile && !activeSampleName) {
      showError("No Audio Selected", "Please select a WAV file or choose one of the benchmark samples.");
      return;
    }

    hideError();
    resultsSection.style.display = "none";
    loadingSection.style.display = "block";
    analyzeBtn.disabled = true;

    // Simulate animated step progression
    startProgressSteps();

    const formData = new FormData();
    if (currentSelectedFile) {
      formData.append("audio", currentSelectedFile);
    } else if (activeSampleName) {
      formData.append("sample_name", activeSampleName);
    }

    try {
      const resp = await fetch("/api/score", {
        method: "POST",
        body: formData
      });

      const data = await resp.json();

      if (!resp.ok || !data.success) {
        throw new Error(data.error || "Inference failed with an unknown error.");
      }

      displayResults(data);
    } catch (err) {
      showError("Evaluation Failed", err.message);
    } finally {
      loadingSection.style.display = "none";
      analyzeBtn.disabled = false;
    }
  }

  // -------------------------------------------------------------------------
  // 5. Render Results
  // -------------------------------------------------------------------------
  function displayResults(data) {
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    // Score & Gauge animation
    const score = data.grammar_score;
    scoreDisplay.textContent = score.toFixed(2);
    
    // Gauge stroke dashoffset calculation: circumference is 2 * PI * 68 ~= 427.26
    const circumference = 427;
    const progress = Math.min(Math.max(score / 5.0, 0), 1.0);
    const offset = circumference - (progress * circumference);
    gaugeFill.style.strokeDashoffset = offset;

    // Bounds Marker
    boundsMarker.style.width = `${(progress * 100).toFixed(1)}%`;

    // Proficiency Band styling
    proficiencyTag.textContent = data.score_band;
    percentileText.textContent = data.percentile_estimate;
    proficiencyDesc.textContent = data.score_description;

    // Badge styling based on score
    if (score >= 4.0) {
      proficiencyTag.style.background = "rgba(16, 185, 129, 0.18)";
      proficiencyTag.style.color = "#6ee7b7";
      proficiencyTag.style.borderColor = "rgba(16, 185, 129, 0.35)";
    } else if (score >= 3.0) {
      proficiencyTag.style.background = "rgba(139, 92, 246, 0.18)";
      proficiencyTag.style.color = "#c4b5fd";
      proficiencyTag.style.borderColor = "rgba(139, 92, 246, 0.35)";
    } else if (score >= 2.0) {
      proficiencyTag.style.background = "rgba(245, 158, 11, 0.18)";
      proficiencyTag.style.color = "#fcd34d";
      proficiencyTag.style.borderColor = "rgba(245, 158, 11, 0.35)";
    } else {
      proficiencyTag.style.background = "rgba(244, 63, 94, 0.18)";
      proficiencyTag.style.color = "#fca5a5";
      proficiencyTag.style.borderColor = "rgba(244, 63, 94, 0.35)";
    }

    evalSpeedBadge.textContent = `Processed in ${data.elapsed_time_s}s (${data.device || 'GPU'})`;

    // Transcript
    transcriptText.textContent = data.transcript || "(No speech detected)";

    // Linguistic metrics
    const ling = data.linguistic_metrics || {};
    const audio = data.audio_metrics || {};

    metricWpm.textContent = `${ling.speaking_rate_wpm || 0} WPM`;
    metricTtr.textContent = (ling.ttr_lexical_diversity || 0).toFixed(3);
    metricWordCount.textContent = ling.word_count || 0;
    metricSentenceCount.textContent = `${ling.sentence_count || 0} sentences`;
    metricSubordination.textContent = (ling.subordination_index || 0).toFixed(3);
    metricFillers.textContent = ling.filler_count || 0;
    metricSilenceRatio.textContent = `${((audio.silence_ratio || 0) * 100).toFixed(1)}%`;
    metricDuration.textContent = `Duration: ${audio.duration_s || 0}s`;
  }

  // -------------------------------------------------------------------------
  // 6. UI Helpers & Clipboard
  // -------------------------------------------------------------------------
  copyTranscriptBtn.addEventListener("click", () => {
    const text = transcriptText.textContent;
    if (!text) return;

    navigator.clipboard.writeText(text).then(() => {
      const copyText = document.getElementById("copyText");
      const copyIcon = document.getElementById("copyIcon");
      copyText.textContent = "Copied!";
      copyIcon.textContent = "✓";
      setTimeout(() => {
        copyText.textContent = "Copy";
        copyIcon.textContent = "📋";
      }, 2000);
    });
  });

  closeErrorBtn.addEventListener("click", hideError);

  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorBanner.style.display = "flex";
  }

  function hideError() {
    errorBanner.style.display = "none";
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  let progressInterval = null;
  function startProgressSteps() {
    stepItems.forEach(item => item.classList.remove("active"));
    stepItems[0].classList.add("active");
    loadingStepTitle.textContent = "1. Validating Audio Format...";
    loadingStepDesc.textContent = "Inspecting WAV header, sample rate, and signal integrity";

    let step = 0;
    clearInterval(progressInterval);
    progressInterval = setInterval(() => {
      step++;
      if (step === 1) {
        stepItems[1].classList.add("active");
        loadingStepTitle.textContent = "2. Transcribing with Whisper ASR...";
        loadingStepDesc.textContent = "Running FP16 neural speech-to-text inference";
      } else if (step === 2) {
        stepItems[2].classList.add("active");
        loadingStepTitle.textContent = "3. Extracting 72 Multimodal Features...";
        loadingStepDesc.textContent = "Computing lexical diversity, syntax subordination, and acoustics";
      } else if (step === 3) {
        stepItems[3].classList.add("active");
        loadingStepTitle.textContent = "4. Scoring with Extra Trees Ensemble...";
        loadingStepDesc.textContent = "Evaluating final model trained on 769 human-scored recordings";
      } else {
        clearInterval(progressInterval);
      }
    }, 900);
  }

  // Load samples on init
  loadSamples();
});
