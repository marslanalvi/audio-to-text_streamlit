"""Compatibility entrypoint.

Use `main.py` as the FastAPI backend entrypoint.
"""

from main import app

MODEL_NAME = "gpt-4o-transcribe-diarize"

__all__ = ["app", "MODEL_NAME"]
