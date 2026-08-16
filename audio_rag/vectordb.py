"""Local Chroma vector database: server lifecycle + client/collection access.

The server is started once with `audiorag db-start` (data persisted under
.chroma/); all other commands talk to it over HTTP on localhost.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config as cfg


def _pid_file(data_dir: Path) -> Path:
    return data_dir / "chroma.pid"


def _log_file(data_dir: Path) -> Path:
    return data_dir / "chroma.log"


def _chroma_executable() -> str:
    exe = Path(sys.executable).parent / "chroma"
    if exe.is_file():
        return str(exe)
    found = shutil.which("chroma")
    if found:
        return found
    raise FileNotFoundError(
        "The 'chroma' CLI was not found. Install project dependencies with: uv sync"
    )


def is_running(host: str = cfg.CHROMA_HOST, port: int = cfg.CHROMA_PORT) -> bool:
    try:
        client = get_client(host, port)
        client.heartbeat()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return False


def start_server(
    data_dir: Path | str = cfg.CHROMA_DIR,
    host: str = cfg.CHROMA_HOST,
    port: int = cfg.CHROMA_PORT,
    timeout: float = 30.0,
) -> str:
    """Start the Chroma server in the background. Returns a status message."""
    data_dir = Path(data_dir)
    if is_running(host, port):
        return f"Chroma is already running at http://{host}:{port} (data: {data_dir})"

    data_dir.mkdir(parents=True, exist_ok=True)
    log = _log_file(data_dir)
    with log.open("ab") as log_fh:
        proc = subprocess.Popen(
            [_chroma_executable(), "run", "--path", str(data_dir),
             "--host", host, "--port", str(port)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survive the CLI process exiting
        )
    _pid_file(data_dir).write_text(str(proc.pid))

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Chroma server exited immediately (code {proc.returncode}). See {log}"
            )
        if is_running(host, port):
            return f"Chroma started at http://{host}:{port} (pid {proc.pid}, data: {data_dir})"
        time.sleep(0.5)
    raise TimeoutError(f"Chroma did not become healthy within {timeout}s. See {log}")


def stop_server(data_dir: Path | str = cfg.CHROMA_DIR) -> str:
    data_dir = Path(data_dir)
    pid_file = _pid_file(data_dir)
    if not pid_file.is_file():
        return "No pid file found; Chroma does not appear to have been started here."
    pid = int(pid_file.read_text().strip())
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink()
        return f"Process {pid} was not running; removed stale pid file."
    pid_file.unlink()
    return f"Stopped Chroma (pid {pid})."


def get_client(host: str = cfg.CHROMA_HOST, port: int = cfg.CHROMA_PORT):
    import chromadb

    return chromadb.HttpClient(host=host, port=port)


def require_client(host: str = cfg.CHROMA_HOST, port: int = cfg.CHROMA_PORT):
    """Client that fails with a friendly hint when the server is down."""
    try:
        client = get_client(host, port)
        client.heartbeat()
        return client
    except Exception as exc:
        raise ConnectionError(
            f"Cannot reach Chroma at http://{host}:{port}. "
            f"Start it with: uv run audiorag db-start  ({exc})"
        ) from exc


def get_collection(client, name: str = cfg.COLLECTION_NAME):
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
