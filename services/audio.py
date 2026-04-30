import asyncio
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI

MODEL_NAME = "gpt-4o-transcribe"
POSTPROCESS_MODEL = "gpt-4.1-mini"
TEAM_MEMBERS = [
    "Saad Hassan",
    "Maaz",
    "Husnain Zafar",
    "Hasaan",
    "Ammar",
    "Mahad",
    "Musab",
    "Awais",
    "Zulqarnain",
    "Yasir",
    "Asad",
    "Eisha Nadeem",
    "Zoonisha Abbas",
    "Zainab Bashir",
    "Tayyeba Arif",
]
PROJECT_NAMES = [
    "Signize.us",
    "Signs.inc",
    "Signmakerz.com",
    "Signage.com",
    "Signage.inc",
    "Higgsfield",
]
TOKEN_NORMALIZATION = {
    "n10": "n8n",
    "nl0": "n8n",
    "n-10": "n8n",
    "hikram": "Zulqarnain",
    "xfail": "Higgsfield",
    "signals.com": "Signage.com",
}
NAME_ALIASES = {
    "hikram": "Zulqarnain",
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _best_match(value: str, choices: list[str], cutoff: float = 0.9) -> str | None:
    norm_to_original = {_normalize_key(choice): choice for choice in choices}
    candidate = _normalize_key(value)
    if not candidate:
        return None
    if candidate in norm_to_original:
        return norm_to_original[candidate]
    matches = difflib.get_close_matches(candidate, list(norm_to_original.keys()), n=1, cutoff=cutoff)
    if matches:
        return norm_to_original[matches[0]]
    return None


def _canonicalize_text(text: str) -> str:
    if not text:
        return text
    updated = text
    for wrong, correct in TOKEN_NORMALIZATION.items():
        updated = re.sub(rf"\b{re.escape(wrong)}\b", correct, updated, flags=re.IGNORECASE)

    for project in PROJECT_NAMES:
        # Replace close variants like "signage inc" -> "Signage.inc"
        project_parts = re.split(r"[.\s]+", project.lower())
        if len(project_parts) >= 2:
            pattern = r"\b" + r"[\s.\-]*".join([re.escape(part) for part in project_parts if part]) + r"\b"
            updated = re.sub(pattern, project, updated, flags=re.IGNORECASE)
    return updated


def _convert_to_valid_wav(audio_bytes: bytes, original_suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name

    output_path = input_path + "_converted.wav"

    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg is not available on this server for audio conversion fallback.")

        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {completed.stderr.strip()}")
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


def _parse_json_payload(payload: str) -> dict[str, Any] | None:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(payload[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _build_english_minutes_summary(client: OpenAI, transcript_text: str, structured_summary: list[Any]) -> str:
    summary_hint = "\n".join([f"- {str(item).strip()}" for item in structured_summary if str(item).strip()])
    completion = client.chat.completions.create(
        model=POSTPROCESS_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write concise meeting minutes in English. "
                    "Return exactly 5 lines, each one sentence, no markdown bullets."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Transcript context:\n{transcript_text}\n\n"
                    f"Structured summary hints:\n{summary_hint or 'Not clear'}\n\n"
                    "Write exactly 5 lines of meeting minutes in plain English."
                ),
            },
        ],
    )
    text = (completion.choices[0].message.content or "").strip()
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if len(lines) >= 5:
        return "\n".join(lines[:5])
    fallback = lines + ["Not clear."] * (5 - len(lines))
    return "\n".join(fallback[:5])


def _diarize_and_summarize(client: OpenAI, transcript_text: str) -> tuple[str, str]:
    estimated_minutes = _estimate_minutes(transcript_text)
    team_names_csv = ", ".join(TEAM_MEMBERS)
    project_names_csv = ", ".join(PROJECT_NAMES)
    base_system = (
        "You are a strict meeting information extractor. "
        "Do not invent text, names, tasks, ETAs, or blockers. "
        "Use only information explicitly present in transcript. "
        "If uncertain, return 'Not clear'. "
        "For consistency across runs, keep speaker ordering by first appearance in transcript. "
        "For names, prioritize exact matches from approved team list only."
    )
    shape = (
        "Return JSON ONLY with this exact shape:\n"
        "{\n"
        '  "people": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "tasks": ["string"],\n'
        '      "eta": "string",\n'
        '      "blockage": "string",\n'
        '      "not_clear": ["string"]\n'
        "    }\n"
        "  ],\n"
        '  "transcript_lines": ["Name: exact spoken line in Roman Urdu"],\n'
        '  "meeting_notes": ["string"],\n'
        '  "summary": ["string"]\n'
        "}\n"
    )
    pass1 = client.chat.completions.create(
        model=POSTPROCESS_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": base_system},
            {
                "role": "user",
                "content": (
                    f"Raw transcript:\n{transcript_text}\n\n"
                    f"Approved team members: {team_names_csv}\n"
                    f"Approved project names: {project_names_csv}\n"
                    "Normalize likely mistakes to approved forms (example: Hikram -> Zulqarnain, n10 -> n8n, xfail -> Higgsfield, Signals.com -> Signage.com).\n"
                    "Never replace one approved member with another unless transcript clearly supports it.\n\n"
                    f"{shape}\n"
                    "Rules:\n"
                    "1) Roman Urdu output only.\n"
                    "2) Use approved names if clear, else 'Unknown Speaker'.\n"
                    "3) If tasks/eta/blockage unknown, use 'Not clear'.\n"
                    "4) Never hallucinate details not in transcript.\n"
                    "5) transcript_lines must cover conversation and not skip major spoken text.\n"
                    "6) JSON only, no markdown."
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    pass1_text = (pass1.choices[0].message.content or "").strip()
    pass2 = client.chat.completions.create(
        model=POSTPROCESS_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": base_system},
            {
                "role": "user",
                "content": (
                    f"Raw transcript:\n{transcript_text}\n\n"
                    f"Approved team members: {team_names_csv}\n"
                    f"Approved project names: {project_names_csv}\n\n"
                    f"Draft JSON to verify and correct:\n{pass1_text}\n\n"
                    "Do a strict second-pass validation:\n"
                    "- Correct wrong names to closest approved name only if transcript evidence exists.\n"
                    "- If unsure, set name as 'Unknown Speaker' (do not guess).\n"
                    "- Correct term normalization: Signals.com -> Signage.com, xfail -> Higgsfield, n10 -> n8n.\n"
                    "- Keep transcript_lines comprehensive.\n"
                    "- Return corrected JSON only.\n\n"
                    f"{shape}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    pass2_text = (pass2.choices[0].message.content or "").strip()
    data = _parse_json_payload(pass2_text) or _parse_json_payload(pass1_text) or {}

    people = data.get("people", []) if isinstance(data, dict) else []
    transcript_lines = data.get("transcript_lines", []) if isinstance(data, dict) else []
    notes = data.get("meeting_notes", []) if isinstance(data, dict) else []
    summary = data.get("summary", []) if isinstance(data, dict) else []

    lines = []
    if not people:
        people = [
            {
                "name": "Unknown Speaker",
                "tasks": ["Not clear"],
                "eta": "Not clear",
                "blockage": "Not clear",
                "not_clear": ["Could not identify participants"],
            }
        ]

    for person in people:
        raw_name = (person.get("name") or "Unknown Speaker").strip()
        alias_name = NAME_ALIASES.get(raw_name.lower())
        matched_name = alias_name or _best_match(raw_name, TEAM_MEMBERS)
        name = matched_name or raw_name or "Unknown Speaker"
        tasks = person.get("tasks") or ["Not clear"]
        eta = _canonicalize_text((person.get("eta") or "Not clear").strip())
        blockage = _canonicalize_text((person.get("blockage") or "Not clear").strip())
        not_clear = person.get("not_clear") or ["None"]

        normalized_tasks = [_canonicalize_text(str(t).strip()) for t in tasks if str(t).strip()]
        normalized_not_clear = [_canonicalize_text(str(n).strip()) for n in not_clear if str(n).strip()]
        tasks_text = ", ".join(normalized_tasks) or "Not clear"
        not_clear_text = ", ".join(normalized_not_clear) or "None"

        lines.extend(
            [
                f"Name: {name}",
                f"Tasks: {tasks_text}",
                f"ETA: {eta}",
                f"Blockage: {blockage}",
                f"Not Clear: {not_clear_text}",
                "",
            ]
        )

    lines.append(f"Meeting Duration: {estimated_minutes} minutes (estimated)")
    lines.append("Transcript:")
    if transcript_lines:
        lines.extend([_canonicalize_text(str(item).strip()) for item in transcript_lines if str(item).strip()])
    else:
        # Never drop the base transcript if formatter output is partial/invalid.
        lines.append(_canonicalize_text(transcript_text))
    lines.append("Meeting Notes:")
    if notes:
        lines.extend([f"- {_canonicalize_text(str(item).strip())}" for item in notes if str(item).strip()])
    else:
        lines.append("- Not clear")
    lines.append("Summary:")
    if summary:
        lines.extend([f"- {_canonicalize_text(str(item).strip())}" for item in summary if str(item).strip()])
    else:
        lines.append("- Not clear")

    missing_members = [name for name in TEAM_MEMBERS if name.lower() not in "\n".join(lines).lower()]
    if missing_members:
        lines.append("")
        lines.append("Team Members Not Mentioned:")
        for member in missing_members:
            lines.extend(
                [
                    f"Name: {member}",
                    "Tasks: Not clear",
                    "ETA: Not clear",
                    "Blockage: Not clear",
                    "Not Clear: Member not clearly mentioned in transcript",
                    "",
                ]
            )

    formatted_text = "\n".join(lines).strip()
    english_minutes = _build_english_minutes_summary(client, transcript_text, summary if isinstance(summary, list) else [])
    return formatted_text, english_minutes


def _transcribe_sync(audio_bytes: bytes, filename: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    original_suffix = Path(filename).suffix.lower() or ".mp3"
    temp_input_path = None
    fallback_wav_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp_audio:
            tmp_audio.write(audio_bytes)
            temp_input_path = tmp_audio.name

        with open(temp_input_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=MODEL_NAME,
                file=audio_file,
                language="ur",              # Force Urdu language
                response_format="json",
                prompt=(
                    "یہ اردو میں ایک میٹنگ کی ریکارڈنگ ہے۔ "
                    "براہ کرم اردو رسم الخط میں ٹرانسکرائب کریں۔ "
                    "انگریزی میں ترجمہ مت کریں۔ "
                    "جو الفاظ انگریزی میں بولے گئے ہیں انہیں انگریزی میں ہی رکھیں۔ "
                    "الفاظ کو نہ بدلیں، نہ خلاصہ کریں، نہ درست کریں۔"
                ),
            )
    except BadRequestError as e:
        error_message = str(e).lower()
        if "corrupted" not in error_message and "unsupported" not in error_message:
            raise RuntimeError(f"Transcription failed: {e}") from e
        fallback_wav_path = _convert_to_valid_wav(audio_bytes, original_suffix)
        try:
            with open(fallback_wav_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model=MODEL_NAME,
                    file=audio_file,
                    language="ur",
                    response_format="json",
                    prompt=(
                        "یہ اردو میں ایک میٹنگ کی ریکارڈنگ ہے۔ "
                        "براہ کرم اردو رسم الخط میں ٹرانسکرائب کریں۔ "
                        "انگریزی میں ترجمہ مت کریں۔ "
                        "جو الفاظ انگریزی میں بولے گئے ہیں انہیں انگریزی میں ہی رکھیں۔ "
                        "الفاظ کو نہ بدلیں، نہ خلاصہ کریں، نہ درست کریں۔"
                    ),
                )
        except Exception as inner_error:
            raise RuntimeError(f"Transcription failed: {inner_error}") from inner_error
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    try:
        transcript_text = _extract_transcript_text(result)
        if not transcript_text:
            return "No transcription generated."
        return _diarize_and_summarize(client, transcript_text)
    finally:
        try:
            if temp_input_path:
                os.remove(temp_input_path)
        except OSError:
            pass
        try:
            if fallback_wav_path:
                os.remove(fallback_wav_path)
        except OSError:
            pass


def _transcribe_and_format_sync(audio_bytes: bytes, filename: str) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    original_suffix = Path(filename).suffix.lower() or ".mp3"
    temp_input_path = None
    fallback_wav_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp_audio:
            tmp_audio.write(audio_bytes)
            temp_input_path = tmp_audio.name

        with open(temp_input_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=MODEL_NAME,
                file=audio_file,
                language="ur",              # Force Urdu language
                response_format="json",
                prompt=(
                    "یہ اردو میں ایک میٹنگ کی ریکارڈنگ ہے۔ "
                    "براہ کرم اردو رسم الخط میں ٹرانسکرائب کریں۔ "
                    "انگریزی میں ترجمہ مت کریں۔ "
                    "جو الفاظ انگریزی میں بولے گئے ہیں انہیں انگریزی میں ہی رکھیں۔ "
                    "الفاظ کو نہ بدلیں، نہ خلاصہ کریں، نہ درست کریں۔"
                ),
            )
    except BadRequestError as e:
        error_message = str(e).lower()
        if "corrupted" not in error_message and "unsupported" not in error_message:
            raise RuntimeError(f"Transcription failed: {e}") from e
        fallback_wav_path = _convert_to_valid_wav(audio_bytes, original_suffix)
        try:
            with open(fallback_wav_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model=MODEL_NAME,
                    file=audio_file,
                    language="ur",
                    response_format="json",
                    prompt=(
                        "یہ اردو میں ایک میٹنگ کی ریکارڈنگ ہے۔ "
                        "براہ کرم اردو رسم الخط میں ٹرانسکرائب کریں۔ "
                        "انگریزی میں ترجمہ مت کریں۔ "
                        "جو الفاظ انگریزی میں بولے گئے ہیں انہیں انگریزی میں ہی رکھیں۔ "
                        "الفاظ کو نہ بدلیں، نہ خلاصہ کریں، نہ درست کریں۔"
                    ),
                )
        except Exception as inner_error:
            raise RuntimeError(f"Transcription failed: {inner_error}") from inner_error
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    try:
        transcript_text = _extract_transcript_text(result)
        if not transcript_text:
            return {
                "transcription": "No transcription generated.",
                "meeting_minutes_en": "Not clear.\nNot clear.\nNot clear.\nNot clear.\nNot clear.",
            }
        formatted_text, english_minutes = _diarize_and_summarize(client, transcript_text)
        return {
            "transcription": formatted_text,
            "meeting_minutes_en": english_minutes,
        }
    finally:
        try:
            if temp_input_path:
                os.remove(temp_input_path)
        except OSError:
            pass
        try:
            if fallback_wav_path:
                os.remove(fallback_wav_path)
        except OSError:
            pass


async def transcribe_audio(audio_bytes: bytes, filename: str) -> dict[str, str]:
    return await asyncio.to_thread(_transcribe_and_format_sync, audio_bytes, filename)