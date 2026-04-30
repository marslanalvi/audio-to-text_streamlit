import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from services.audio import transcribe_audio

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".opus"}

app = FastAPI(title="Audio To Text API", version="1.0.0")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/transcribe", response_class=JSONResponse)
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size is 25MB.")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

    payload = await transcribe_audio(audio_bytes, file.filename or "audio.mp3")
    return JSONResponse(payload)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
