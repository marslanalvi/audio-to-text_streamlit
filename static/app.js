const form = document.getElementById("upload-form");
const fileInput = document.getElementById("audio-file");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");
const dropZone = document.getElementById("drop-zone");
const fileMeta = document.getElementById("file-meta");
const audioPreview = document.getElementById("audio-preview");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");

const loaderOverlay = document.getElementById("loader-overlay");
const loaderText = document.getElementById("loader-text");
const loaderElapsed = document.getElementById("loader-elapsed");
const loaderBar = document.getElementById("loader-bar");

let loaderTicker = null;
let loaderProgressTicker = null;

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.1 }
);

document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));

function startLoader() {
  let seconds = 0;
  let progress = 6;

  loaderOverlay.hidden = false;
  loaderText.textContent = "Uploading audio...";
  loaderElapsed.textContent = "0s elapsed · 6%";
  loaderBar.style.width = "6%";

  loaderProgressTicker = setInterval(() => {
    // Natural-feeling progress: fast at start, slower near completion.
    if (progress < 85) {
      progress += 5;
    } else if (progress < 94) {
      progress += 1;
    }
    loaderBar.style.width = `${progress}%`;

    if (progress < 22) {
      loaderText.textContent = "Uploading audio...";
    } else if (progress < 52) {
      loaderText.textContent = "Reading speech patterns...";
    } else if (progress < 84) {
      loaderText.textContent = "Generating transcript...";
    } else {
      loaderText.textContent = "Finalizing notes and minutes...";
    }
  }, 900);

  loaderTicker = setInterval(() => {
    seconds += 1;
    loaderElapsed.textContent = `${seconds}s elapsed · ${progress}%`;
  }, 1000);
}

function stopLoader() {
  loaderBar.style.width = "100%";
  loaderText.textContent = "Completed.";
  loaderOverlay.hidden = true;
  if (loaderTicker) {
    clearInterval(loaderTicker);
    loaderTicker = null;
  }
  if (loaderProgressTicker) {
    clearInterval(loaderProgressTicker);
    loaderProgressTicker = null;
  }
}

function setFile(file) {
  if (!file) {
    fileMeta.textContent = "No file selected";
    audioPreview.hidden = true;
    audioPreview.removeAttribute("src");
    return;
  }

  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  fileMeta.textContent = `${file.name} (${sizeMb} MB)`;
  audioPreview.src = URL.createObjectURL(file);
  audioPreview.hidden = false;
}

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  fileInput.files = event.dataTransfer.files;
  setFile(file);
});

fileInput.addEventListener("change", () => {
  setFile(fileInput.files?.[0]);
});

copyBtn.addEventListener("click", async () => {
  if (!resultEl.textContent) return;
  await navigator.clipboard.writeText(resultEl.textContent);
  statusEl.textContent = "Copied to clipboard.";
});

downloadBtn.addEventListener("click", () => {
  const text = resultEl.textContent || "";
  if (!text) return;
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "transcript.txt";
  link.click();
  URL.revokeObjectURL(url);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files?.[0];
  if (!file) {
    statusEl.textContent = "Please select a file first.";
    return;
  }

  const data = new FormData();
  data.append("file", file);

  submitBtn.disabled = true;
  startLoader();
  statusEl.classList.add("busy");
  copyBtn.disabled = true;
  downloadBtn.disabled = true;
  statusEl.textContent = "Transcribing audio...";
  resultEl.textContent = "";

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: data,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Transcription failed.");
    }

    statusEl.textContent = "Completed.";
    resultEl.textContent = payload.transcription || "No transcription generated.";
    copyBtn.disabled = !resultEl.textContent;
    downloadBtn.disabled = !resultEl.textContent;
  } catch (error) {
    statusEl.textContent = "Request failed.";
    resultEl.textContent = error.message;
  } finally {
    submitBtn.disabled = false;
    statusEl.classList.remove("busy");
    stopLoader();
  }
});
