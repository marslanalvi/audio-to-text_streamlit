import asyncio
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI

MODEL_NAME = "gpt-4o-transcribe"
POSTPROCESS_MODEL = "gpt-4o"  # Upgraded for better speaker attribution

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
    "Eishah Nadeem",
    "Zoonisha Abbas",
    "Zainab Bashir",
    "Tayyaba Arif",
    "Hamza",
    "Arsalan",
    "Meerza",
    "Taheera",
    "Rida",
]

PROJECT_NAMES = [
    "Signize.us",
    "Signs.inc",
    "Signmakerz.com",
    "Signage.com",
    "Signage.inc",
    "QuickSignage.com",
    "Higgsfield",
]

TOKEN_NORMALIZATION = {
    # n8n variants
    "n10": "n8n",
    "nl0": "n8n",
    "n-10": "n8n",
    # Name typos
    "hikram": "Zulqarnain",
    "hekram": "Zulqarnain",
    # Project name fixes
    "xfail": "Higgsfield",
    "signals.com": "Signage.com",
    "signage inc": "Signage.inc",
    "signs inc": "Signs.inc",
    "quick signage": "QuickSignage.com",
    "quicksignage": "QuickSignage.com",
    # Phonetic STT fixes
    "eleven laps": "ElevenLabs",
    "11 laps": "ElevenLabs",
    "eleven labs": "ElevenLabs",
    "elevenlap": "ElevenLabs",
    "jop": "JOP",
    "j.o.p": "JOP",
}

NAME_ALIASES: dict[str, str] = {
    # Zulqarnain variants
    "hikram": "Zulqarnain",
    "hekram": "Zulqarnain",
    "zulqernain": "Zulqarnain",
    "zulkarnain": "Zulqarnain",
    "zulqarnian": "Zulqarnain",
    # Tayyaba Arif variants
    "tayba": "Tayyaba Arif",
    "tayyeba": "Tayyaba Arif",
    "tayyeba arif": "Tayyaba Arif",
    "tayba arif": "Tayyaba Arif",
    "tayyaba": "Tayyaba Arif",
    # Eishah Nadeem variants
    "eisha": "Eishah Nadeem",
    "aisha": "Eishah Nadeem",
    "aisha nadeem": "Eishah Nadeem",
    "eisha nadeem": "Eishah Nadeem",
    "ayesha nadeem": "Eishah Nadeem",
    # Husnain Zafar variants
    "husnain": "Husnain Zafar",
    "husnain zafar": "Husnain Zafar",
    # Hasaan variants (including honorific)
    "hasan": "Hasaan",
    "hassan": "Hasaan",
    "sir hasaan": "Hasaan",
    "sir hassan": "Hasaan",
    "sir hasan": "Hasaan",
    # Arsalan variants
    "arslan": "Arsalan",
    "arslaan": "Arsalan",
    # Saad Hassan variants
    "saad": "Saad Hassan",
    "saad hassan": "Saad Hassan",
    # Other existing
    "zoonisha": "Zoonisha Abbas",
    "maaz": "Maaz",
    # New members — Meerza variants
    "mirza": "Meerza",
    "meerza": "Meerza",
    "mirza sahab": "Meerza",
    # New members — Taheera variants
    "tahira": "Taheera",
    "tahera": "Taheera",
    "taheera": "Taheera",
    "tahira arif": "Taheera",
    # New members — Rida variants (Ma'am Rida)
    "rida": "Rida",
    "maam rida": "Rida",
    "ma'am rida": "Rida",
    "madam rida": "Rida",
    "miss rida": "Rida",
}

# Domain vocabulary fed to the STT model as a prompt bias — the model strongly
# prefers these spellings over phonetically similar alternatives.
_DOMAIN_VOCAB: list[str] = [
    # AI / automation tools
    "ElevenLabs", "n8n", "Higgsfield", "Clarity", "JOP",
    # Dev / infrastructure
    "Shopify", "Postman", "GitHub", "CI/CD", "QA", "ECS", "API",
    "Python", "automation", "workflow", "dashboard", "white label",
    "private app", "inbound calling", "AI agent", "retargeting",
    # Logistics / finance
    "DHL", "FedEx", "surcharges", "refunds", "CTR",
    # Internal domain terms
    "job card", "sales calculator", "EPIC Crafting", "order confirmation",
    "image resize", "marketing report", "multi-login",
]

# Included in transcription API call so the model recognises names better.
_TRANSCRIPTION_PROMPT = (
    "This is a daily standup meeting recording spoken in Urdu/Roman Urdu. "
    "Team members who may be speaking: "
    + ", ".join(TEAM_MEMBERS)
    + ". Projects discussed: "
    + ", ".join(PROJECT_NAMES)
    + ". Technical terms and tools: "
    + ", ".join(_DOMAIN_VOCAB)
    + ". "
    "Transcribe in Urdu script exactly as spoken. "
    "Keep all English words (names, tools, project names) in English. "
    "Do not translate, summarise, or correct any words. "
    "Preserve every speaker name exactly as heard."
)

_JSON_SHAPE = (
    "Return JSON ONLY — no markdown — with this exact shape:\n"
    "{\n"
    '  "people": [\n'
    "    {\n"
    '      "name": "Approved team member name or Unknown Speaker",\n'
    '      "tasks": ["concise English description of what they worked on or completed"],\n'
    '      "eta": "time estimate in English, or Not clear",\n'
    '      "blocker": "blocker description in English, or None"\n'
    "    }\n"
    "  ]\n"
    "}"
)

_BASE_SYSTEM = (
    "You are a strict meeting-minutes extractor for a software team's daily standup. "
    "Rules you must never break:\n"
    "1. Do NOT invent tasks, names, ETAs, or blockers not present in the transcript.\n"
    "2. Do NOT skip any speaker — every person who spoke must appear.\n"
    "3. Match speaker names exclusively to the approved team-member list.\n"
    "4. Translate task descriptions to clear, concise English even if spoken in Urdu.\n"
    "5. Use 'Not clear' when information is genuinely missing, 'None' when there is no blocker.\n"
    "6. JSON only — absolutely no markdown or extra text.\n"
    "7. IMPORTANT — Proxy reporting: In this meeting one person may report tasks on behalf "
    "of another (e.g. 'Awais is working on X'). In that case the task belongs to the NAMED "
    "person (Awais), not the speaker. Always assign tasks to the person they are about, "
    "not the person speaking.\n"
    "8. Team members include both Shopify developers and an AI team (working on n8n, "
    "ElevenLabs, Python, automation). Context helps identify who is who."
)

_NAME_RULES = (
    "Name normalisation rules (apply strictly):\n"
    "- Hikram / Hekram / Zulqernain / Zulkarnain → Zulqarnain\n"
    "- Tayyeba / Tayba → Tayyaba Arif\n"
    "- Eisha / Aisha / Ayesha Nadeem → Eishah Nadeem\n"
    "- Husnain → Husnain Zafar\n"
    "- Hassan / Hasan / Sir Hasaan / Sir Hassan → Hasaan\n"
    "- Arslan → Arsalan\n"
    "- Saad → Saad Hassan\n"
    "- Mirza → Meerza\n"
    "- Tahira / Tahera → Taheera\n"
    "- Rida / Ma'am Rida / Maam Rida / Madam Rida → Rida\n"
    "- n10 / nl0 / n-10 → n8n\n"
    "- xfail / Higgsfield → Higgsfield\n"
    "- Signals.com → Signage.com\n"
    "- Eleven laps / 11 laps / Eleven Labs → ElevenLabs\n"
    "- jop / j.o.p → JOP\n"
    "- Quick Signage / QuickSignage → QuickSignage.com\n"
    "- If a name cannot be matched at all, use 'Unknown Speaker'.\n"
    "- NEVER replace one approved member with a different approved member unless "
    "the transcript clearly supports it.\n"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _best_match(value: str, choices: list[str], cutoff: float = 0.82) -> str | None:
    norm_to_original = {_normalize_key(c): c for c in choices}
    candidate = _normalize_key(value)
    if not candidate:
        return None
    if candidate in norm_to_original:
        return norm_to_original[candidate]
    matches = difflib.get_close_matches(
        candidate, list(norm_to_original.keys()), n=1, cutoff=cutoff
    )
    return norm_to_original[matches[0]] if matches else None


def _canonicalize_text(text: str) -> str:
    if not text:
        return text
    updated = text
    for wrong, correct in TOKEN_NORMALIZATION.items():
        updated = re.sub(
            rf"\b{re.escape(wrong)}\b", correct, updated, flags=re.IGNORECASE
        )
    for project in PROJECT_NAMES:
        parts = re.split(r"[.\s]+", project.lower())
        if len(parts) >= 2:
            pattern = (
                r"\b"
                + r"[\s.\-]*".join(re.escape(p) for p in parts if p)
                + r"\b"
            )
            updated = re.sub(pattern, project, updated, flags=re.IGNORECASE)
    return updated


def _resolve_name(raw_name: str) -> str:
    key = raw_name.strip().lower()
    if key in NAME_ALIASES:
        return NAME_ALIASES[key]
    matched = _best_match(raw_name, TEAM_MEMBERS)
    return matched or raw_name or "Unknown Speaker"


def _convert_to_valid_wav(audio_bytes: bytes, original_suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name

    output_path = input_path + "_converted.wav"
    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "ffmpeg is not available on this server for audio conversion fallback."
            )
        command = [
            ffmpeg_bin, "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", output_path,
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
                return json.loads(payload[start: end + 1])
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# Core diarisation + formatting
# ---------------------------------------------------------------------------

def _diarize_and_summarize(client: OpenAI, transcript_text: str) -> str:
    """Run two-pass GPT extraction and return formatted meeting minutes string."""
    today_str = date.today().strftime("%B %d, %Y")
    team_csv = ", ".join(TEAM_MEMBERS)
    project_csv = ", ".join(PROJECT_NAMES)

    # ---- Pass 1: initial extraction ----------------------------------------
    pass1 = client.chat.completions.create(
        model=POSTPROCESS_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _BASE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Raw transcript:\n{transcript_text}\n\n"
                    f"Approved team members: {team_csv}\n"
                    f"Approved project names: {project_csv}\n\n"
                    f"{_NAME_RULES}\n"
                    "Instructions:\n"
                    "- List EVERY person who spoke, in order of first appearance.\n"
                    "- For tasks: translate what they said they are doing/completed into "
                    "  clear English bullet points.\n"
                    "- For ETA: extract any time mentioned (e.g. 'end of day', '1 hour', "
                    "  'today', 'completed').\n"
                    "- For blocker: any issue stopping progress. Use 'None' if no blocker.\n\n"
                    f"{_JSON_SHAPE}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    pass1_text = (pass1.choices[0].message.content or "").strip()

    # ---- Pass 2: validation & correction ------------------------------------
    pass2 = client.chat.completions.create(
        model=POSTPROCESS_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _BASE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Raw transcript:\n{transcript_text}\n\n"
                    f"Approved team members: {team_csv}\n"
                    f"Approved project names: {project_csv}\n\n"
                    f"Draft JSON to validate:\n{pass1_text}\n\n"
                    "Second-pass checklist — fix every issue found:\n"
                    "1. Is every speaker from the transcript represented? Add any missing ones.\n"
                    "2. Are all names from the approved list? Correct or set 'Unknown Speaker'.\n"
                    f"{_NAME_RULES}"
                    "3. Are all task descriptions in clear English? Fix any that are not.\n"
                    "4. Are project names normalised? (Signals.com → Signage.com, n10 → n8n, "
                    "   xfail → Higgsfield)\n"
                    "5. Blocker field: 'None' if no blocker, 'Not clear' if unclear.\n\n"
                    f"{_JSON_SHAPE}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    pass2_text = (pass2.choices[0].message.content or "").strip()

    data = _parse_json_payload(pass2_text) or _parse_json_payload(pass1_text) or {}
    people: list[dict] = data.get("people", []) if isinstance(data, dict) else []

    if not people:
        people = [
            {
                "name": "Unknown Speaker",
                "tasks": ["Not clear"],
                "eta": "Not clear",
                "blocker": "None",
            }
        ]

    # ---- Format output ------------------------------------------------------
    lines: list[str] = [f"Minutes of Meeting Daily \u2013 {today_str}", ""]

    for person in people:
        raw_name = (person.get("name") or "Unknown Speaker").strip()
        name = _resolve_name(raw_name)

        raw_tasks: list = person.get("tasks") or ["Not clear"]
        tasks_list = [
            _canonicalize_text(str(t).strip()) for t in raw_tasks if str(t).strip()
        ]
        tasks_text = ", ".join(tasks_list) or "Not clear"

        eta = _canonicalize_text((person.get("eta") or "Not clear").strip())
        blocker = _canonicalize_text((person.get("blocker") or "None").strip())

        lines.extend([
            name,
            f"Tasks: {tasks_text}",
            f"ETA: {eta}",
            f"Blocker: {blocker}",
            "",
        ])

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Transcription + full pipeline
# ---------------------------------------------------------------------------

def _run_transcription(client: OpenAI, audio_file: Any) -> Any:
    return client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=audio_file,
        language="ur",
        response_format="json",
        prompt=_TRANSCRIPTION_PROMPT,
    )


def _transcribe_and_format_sync(audio_bytes: bytes, filename: str) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    original_suffix = Path(filename).suffix.lower() or ".mp3"
    temp_input_path: str | None = None
    fallback_wav_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp:
            tmp.write(audio_bytes)
            temp_input_path = tmp.name

        try:
            with open(temp_input_path, "rb") as af:
                result = _run_transcription(client, af)
        except BadRequestError as e:
            err = str(e).lower()
            if "corrupted" not in err and "unsupported" not in err:
                raise RuntimeError(f"Transcription failed: {e}") from e
            # Fallback: re-encode to WAV via ffmpeg
            fallback_wav_path = _convert_to_valid_wav(audio_bytes, original_suffix)
            try:
                with open(fallback_wav_path, "rb") as af:
                    result = _run_transcription(client, af)
            except Exception as inner:
                raise RuntimeError(f"Transcription failed: {inner}") from inner

        transcript_text = _extract_transcript_text(result)
        if not transcript_text:
            return {"transcription": "No transcription generated."}

        formatted = _diarize_and_summarize(client, transcript_text)
        return {"transcription": formatted}

    except Exception as e:
        raise RuntimeError(f"Pipeline error: {e}") from e
    finally:
        for path in (temp_input_path, fallback_wav_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


async def transcribe_audio(audio_bytes: bytes, filename: str) -> dict[str, str]:
    return await asyncio.to_thread(_transcribe_and_format_sync, audio_bytes, filename)