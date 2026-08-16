"""Audio -> text transcription with open-source OpenAI Whisper."""

from __future__ import annotations

from pathlib import Path

from . import config as cfg

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aac", ".mp4", ".webm"}


def transcribe_file(
    audio_path: Path | str,
    model_name: str = cfg.WHISPER_MODEL,
    out_dir: Path | str = cfg.TRANSCRIPTS_DIR,
    language: str | None = None,
) -> Path:
    """Transcribe one audio file and write `<out_dir>/<stem>.txt`. Returns the path."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language, fp16=False)
    text = str(result["text"]).strip()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{audio_path.stem}.txt"
    out_path.write_text(text + "\n")
    return out_path


def collect_audio_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files.extend(
                f for f in sorted(p.rglob("*"))
                if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES
            )
        elif p.is_file():
            files.append(p)
    return files
