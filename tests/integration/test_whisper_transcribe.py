"""Integration: Whisper transcribes a generated speech clip (macOS `say` + ffmpeg)."""

import shutil
import subprocess

import pytest

from audio_rag.transcribe import collect_audio_files, transcribe_file

pytestmark = pytest.mark.integration

needs_say = pytest.mark.skipif(
    shutil.which("say") is None or shutil.which("ffmpeg") is None,
    reason="requires macOS `say` and ffmpeg to synthesize test audio",
)


@needs_say
def test_whisper_transcribes_generated_audio(tmp_path):
    audio = tmp_path / "greeting.aiff"
    subprocess.run(
        ["say", "-o", str(audio),
         "Hello world. This is a test of the audio retrieval pipeline."],
        check=True,
    )

    out = transcribe_file(audio, model_name="tiny", out_dir=tmp_path / "transcripts")

    assert out.name == "greeting.txt"
    text = out.read_text().lower()
    assert "hello" in text
    assert "test" in text


def test_collect_audio_files_filters_by_suffix(tmp_path):
    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.wav").touch()
    (tmp_path / "notes.txt").touch()
    files = collect_audio_files([str(tmp_path)])
    assert [f.name for f in files] == ["a.mp3", "b.wav"]
