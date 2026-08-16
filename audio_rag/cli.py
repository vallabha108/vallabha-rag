"""audiorag CLI: audio -> Whisper transcript -> Chroma vector DB -> Ollama answer.

Typical flow:
    uv run audiorag db-start
    uv run audiorag transcribe recording.mp3
    uv run audiorag upload transcripts/recording.txt
    uv run audiorag search "what was discussed about pricing?"
    uv run audiorag ask "summarize the pricing discussion"
"""

from __future__ import annotations

import argparse
import sys

from . import config as cfg
from .config import Config


def _config_from_args(args: argparse.Namespace) -> Config:
    config = Config.load()
    if getattr(args, "llm_model", None):
        config.llm_model = args.llm_model
        config.source["llm_model"] = "cli:--llm-model"
    if getattr(args, "embedding_model", None):
        config.embedding_model = args.embedding_model
        config.source["embedding_model"] = "cli:--embedding-model"
    return config


def _embedder(config: Config):
    from .embeddings import Embedder

    return Embedder(config.embedding_model)


def _collection():
    from . import vectordb

    client = vectordb.require_client()
    return vectordb.get_collection(client)


def cmd_transcribe(args: argparse.Namespace) -> None:
    from .transcribe import collect_audio_files, transcribe_file

    files = collect_audio_files(args.paths)
    if not files:
        sys.exit("No audio files found in the given paths.")
    for f in files:
        print(f"Transcribing {f.name} (whisper model: {args.whisper_model}) ...", flush=True)
        out = transcribe_file(f, model_name=args.whisper_model,
                              out_dir=args.out_dir, language=args.language)
        print(f"  -> {out}")


def cmd_db_start(args: argparse.Namespace) -> None:
    from . import vectordb

    print(vectordb.start_server())


def cmd_db_stop(args: argparse.Namespace) -> None:
    from . import vectordb

    print(vectordb.stop_server())


def cmd_db_status(args: argparse.Namespace) -> None:
    from . import vectordb

    if vectordb.is_running():
        client = vectordb.get_client()
        collection = vectordb.get_collection(client)
        print(f"Chroma is UP at http://{cfg.CHROMA_HOST}:{cfg.CHROMA_PORT}")
        print(f"Collection '{cfg.COLLECTION_NAME}': {collection.count()} chunks")
    else:
        print(f"Chroma is DOWN (expected at http://{cfg.CHROMA_HOST}:{cfg.CHROMA_PORT}).")
        print("Start it with: uv run audiorag db-start")
        sys.exit(1)


def cmd_upload(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .ingest import upload_transcript

    config = _config_from_args(args)
    collection = _collection()
    embedder = _embedder(config)

    paths = []
    for raw in args.paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("*.txt")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"warning: {p} not found, skipping", file=sys.stderr)
    if not paths:
        sys.exit("No transcript .txt files found.")

    for p in paths:
        result = upload_transcript(collection, embedder, p, force=args.force)
        if result.skipped:
            print(f"  skip     {result.source} (already uploaded, use --force to replace)")
        else:
            print(f"  upload   {result.source}: {result.chunks} chunks "
                  f"(embeddings: {config.embedding_model})")


def cmd_list(args: argparse.Namespace) -> None:
    from .ingest import list_sources

    sources = list_sources(_collection())
    if not sources:
        print("Collection is empty.")
        return
    for source, count in sorted(sources.items()):
        print(f"{source}: {count} chunks")


def cmd_search(args: argparse.Namespace) -> None:
    from .search import search

    config = _config_from_args(args)
    chunks = search(_collection(), _embedder(config), args.query, top_k=args.top_k)
    if not chunks:
        print("No matching chunks found (is anything uploaded?).")
        return
    for i, c in enumerate(chunks, 1):
        print(f"--- [{i}] {c.source} chunk {c.chunk_index} (score {c.score:.3f}) ---")
        print(c.text)
        print()


def cmd_ask(args: argparse.Namespace) -> None:
    from .answer import generate_answer
    from .search import search

    config = _config_from_args(args)
    chunks = search(_collection(), _embedder(config), args.query, top_k=args.top_k)
    if not chunks:
        sys.exit("No context found in the vector DB — upload a transcript first.")
    print(f"Retrieved {len(chunks)} chunks; asking {config.llm_model} ...\n", flush=True)
    answer = generate_answer(args.query, chunks, config.llm_model)
    print(answer.text)
    if args.show_chunks:
        print("\n--- context used ---")
        for i, c in enumerate(answer.chunks, 1):
            print(f"[{i}] {c.source} chunk {c.chunk_index} (score {c.score:.3f})")


def cmd_config_show(args: argparse.Namespace) -> None:
    config = Config.load()
    print(f"config file: {cfg.config_path()} "
          f"({'exists' if cfg.config_path().is_file() else 'not created yet'})")
    for key in cfg.CONFIGURABLE_KEYS:
        print(f"  {key} = {getattr(config, key)}  (from {config.source[key]})")


def cmd_config_set(args: argparse.Namespace) -> None:
    config = Config.load()
    config.set_and_save(args.key, args.value)
    print(f"Set {args.key} = {args.value} in {cfg.config_path()}")


def _add_model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embedding-model", default=None,
                   help="Override the embedding model for this command")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audiorag",
        description="Audio RAG: Whisper transcripts + local Chroma vector DB + Ollama answers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("transcribe", help="Transcribe audio files to text with Whisper")
    p_tr.add_argument("paths", nargs="+", help="Audio files or directories")
    p_tr.add_argument("--whisper-model", default=cfg.WHISPER_MODEL,
                      help=f"Whisper model size (default: {cfg.WHISPER_MODEL})")
    p_tr.add_argument("--out-dir", default=cfg.TRANSCRIPTS_DIR,
                      help=f"Where to write .txt transcripts (default: {cfg.TRANSCRIPTS_DIR}/)")
    p_tr.add_argument("--language", default=None, help="Force a language (default: auto-detect)")
    p_tr.set_defaults(func=cmd_transcribe)

    sub.add_parser("db-start", help="Start the local Chroma vector DB server").set_defaults(func=cmd_db_start)
    sub.add_parser("db-stop", help="Stop the local Chroma server").set_defaults(func=cmd_db_stop)
    sub.add_parser("db-status", help="Show Chroma server and collection status").set_defaults(func=cmd_db_status)

    p_up = sub.add_parser("upload", help="Chunk, embed and upload transcript .txt files to Chroma")
    p_up.add_argument("paths", nargs="+", help="Transcript files or directories")
    p_up.add_argument("--force", action="store_true", help="Replace an already-uploaded transcript")
    _add_model_flags(p_up)
    p_up.set_defaults(func=cmd_upload)

    sub.add_parser("list", help="List uploaded transcripts and chunk counts").set_defaults(func=cmd_list)

    p_se = sub.add_parser("search", help="Semantic search: return matching transcript chunks")
    p_se.add_argument("query")
    p_se.add_argument("--top-k", type=int, default=5, help="Chunks to return (default: 5)")
    _add_model_flags(p_se)
    p_se.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="RAG answer: retrieve chunks, then ask the local LLM")
    p_ask.add_argument("query")
    p_ask.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve (default: 5)")
    p_ask.add_argument("--llm-model", default=None, help="Override the Ollama model for this command")
    p_ask.add_argument("--show-chunks", action="store_true", help="Also print the retrieved context")
    _add_model_flags(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_cfg = sub.add_parser("config", help="Show or change the configuration")
    cfg_sub = p_cfg.add_subparsers(dest="config_command", required=True)
    cfg_sub.add_parser("show", help="Show effective configuration").set_defaults(func=cmd_config_show)
    p_cfg_set = cfg_sub.add_parser("set", help="Persist a config value to audiorag.json")
    p_cfg_set.add_argument("key", choices=list(cfg.CONFIGURABLE_KEYS))
    p_cfg_set.add_argument("value")
    p_cfg_set.set_defaults(func=cmd_config_set)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ConnectionError, FileNotFoundError, TimeoutError, ValueError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
