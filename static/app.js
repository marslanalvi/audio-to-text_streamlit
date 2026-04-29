const form = document.getElementById("upload-form");
const fileInput = document.getElementById("audio-file");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");

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
  } catch (error) {
    statusEl.textContent = "Request failed.";
    resultEl.textContent = error.message;
  } finally {
    submitBtn.disabled = false;
  }
});
