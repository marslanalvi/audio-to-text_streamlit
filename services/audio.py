import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydub import AudioSegment

# ── Your exact ffmpeg path ───────────────────────────────────────
AudioSegment.converter = r"C:\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffprobe.exe"

MODEL_NAME = "gpt-4o-transcribe-api-ev3"
POSTPROCESS_MODEL = "gpt-4.1-mini"


def _convert_to_valid_mp3(audio_bytes: bytes, original_suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name

    output_path = input_path + "_converted.mp3"

    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(
            output_path,
            format="mp3",
            bitrate="128k",
            parameters=["-ar", "16000", "-ac", "1"],
        )
        return output_path
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


def _extract_transcript_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return (result.get("text") or "").strip()
    return (getattr(result, "text", "") or "").strip()


def _estimate_minutes(text: str) -> int:
    words = len(text.split())
    if words == 0:
        return 1
    # Approximate spoken conversation speed.
    return max(1, round(words / 130))


def _diarize_and_summarize(client: OpenAI, transcript_text: str) -> str:
    estimated_minutes = _estimate_minutes(transcript_text)
    completion = client.chat.completions.create(
        model=POSTPROCESS_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a transcript post-processor. Convert transcript text into speaker-wise format. "
                    "If names are present, map them consistently to Speaker 1, Speaker 2, Speaker 3, etc. "
                    "Do output real names. Keep original meaning. "
                    "Final output must be in Roman Urdu (Urdu written in English letters)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Raw transcript:\n{transcript_text}\n\n"
                    "Output rules:\n"
                    "1) Return only plain text.\n"
                    "2) Format each line as 'Speaker N: ...'.\n"
                    "3) Merge consecutive lines of same speaker when natural.\n"
                    f"4) At the end, add heading 'Summary ({estimated_minutes} minutes):' then 3 concise bullet points.\n"
                    "5) Bullets must reflect key decisions, issues, and action items.\n"
                    "6) Write everything in Roman Urdu (English typed Urdu), not Urdu script."
                ),
            },
        ],
    )
    ai_text = (completion.choices[0].message.content or "").strip()
    return ai_text if ai_text else f"Speaker 1: {transcript_text}"


def _transcribe_sync(audio_bytes: bytes, filename: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    original_suffix = Path(filename).suffix.lower() or ".mp3"
    converted_path = _convert_to_valid_mp3(audio_bytes, original_suffix)

    try:
        with open(converted_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=MODEL_NAME,
                file=audio_file,
                language="ur",              # Force Urdu language
                response_format="json",
                prompt=(
                    "یہ اردو میں ایک میٹنگ کی ریکارڈنگ ہے۔ "
                    "براہ کرم اردو رسم الخط میں ٹرانسکرائب کریں۔ "
                    "انگریزی میں ترجمہ مت کریں۔ "
                    "جو الفاظ انگریزی میں بولے گئے ہیں انہیں انگریزی میں ہی رکھیں۔"
                ),
            )
        transcript_text = _extract_transcript_text(result)
        if not transcript_text:
            return "No transcription generated."
        return _diarize_and_summarize(client, transcript_text)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        try:
            os.remove(converted_path)
        except OSError:
            pass


async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    return await asyncio.to_thread(_transcribe_sync, audio_bytes, filename)