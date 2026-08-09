"""CLI for managing a Vertex AI RAG Engine corpus for the Vallabha VLE project.

Uses Application Default Credentials. Defaults (project, location, corpus name)
can be overridden with flags or the VALLABHA_RAG_* environment variables.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# vertexai.rag still works; hide the SDK's migration notice until we move to agentplatform.
warnings.filterwarnings("ignore", message=".*vertexai.rag.*deprecated.*")

import vertexai
from vertexai import rag

DEFAULT_PROJECT = os.environ.get("VALLABHA_RAG_PROJECT", "vallabha-systems-vle")
DEFAULT_LOCATION = os.environ.get("VALLABHA_RAG_LOCATION", "europe-west4")
DEFAULT_CORPUS = os.environ.get("VALLABHA_RAG_CORPUS", "vallabha-vle-docs")

SUPPORTED_SUFFIXES = {
    ".pdf", ".txt", ".md", ".html", ".doc", ".docx",
    ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".json",
}


def init(args: argparse.Namespace) -> None:
    vertexai.init(project=args.project, location=args.location)


def find_corpus_name(display_name: str) -> str | None:
    for corpus in rag.list_corpora():
        if corpus.display_name == display_name:
            return str(corpus.name)
    return None


def ensure_corpus_name(display_name: str) -> str:
    name = find_corpus_name(display_name)
    if name is not None:
        return name
    print(f"Creating corpus '{display_name}' (default embedding + indexing) ...")
    return str(rag.create_corpus(display_name=display_name).name)


def collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files.extend(
                f for f in sorted(p.rglob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {p} not found, skipping", file=sys.stderr)
    return files


def cmd_create_corpus(args: argparse.Namespace) -> None:
    init(args)
    name = ensure_corpus_name(args.corpus)
    print(f"Corpus ready: {args.corpus}\n  {name}")


def cmd_upload(args: argparse.Namespace) -> None:
    init(args)
    files = collect_files(args.paths)
    if not files:
        sys.exit("No uploadable files found.")

    corpus_name = ensure_corpus_name(args.corpus)
    existing = {f.display_name for f in rag.list_files(corpus_name=corpus_name)}

    uploaded = skipped = failed = 0
    for f in files:
        if f.name in existing and not args.force:
            print(f"  skip     {f.name} (already in corpus, use --force to re-upload)")
            skipped += 1
            continue
        print(f"  upload   {f.name} ({f.stat().st_size // 1024} KB) ...", flush=True)
        try:
            rag.upload_file(
                corpus_name=corpus_name,
                path=str(f),
                display_name=f.name,
                description=f"Uploaded from {f}",
            )
            uploaded += 1
        except Exception as exc:  # noqa: BLE001 - report per-file failure and continue
            print(f"  FAILED   {f.name}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped, {failed} failed.")
    if failed:
        sys.exit(1)


def cmd_list_files(args: argparse.Namespace) -> None:
    init(args)
    corpus_name = find_corpus_name(args.corpus)
    if corpus_name is None:
        sys.exit(f"Corpus '{args.corpus}' does not exist yet. Run: uv run rag create-corpus")
    files = list(rag.list_files(corpus_name=corpus_name))
    if not files:
        print("Corpus is empty.")
        return
    for f in files:
        print(f"{f.display_name}\n  {f.name}")
    print(f"\n{len(files)} file(s) in corpus '{args.corpus}'.")


def cmd_list_corpora(args: argparse.Namespace) -> None:
    init(args)
    corpora = list(rag.list_corpora())
    if not corpora:
        print(f"No corpora in {args.project}/{args.location}.")
        return
    for c in corpora:
        print(f"{c.display_name}\n  {c.name}")


def cmd_query(args: argparse.Namespace) -> None:
    init(args)
    corpus_name = find_corpus_name(args.corpus)
    if corpus_name is None:
        sys.exit(f"Corpus '{args.corpus}' does not exist yet.")
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=args.text,
        rag_retrieval_config=rag.RagRetrievalConfig(top_k=args.top_k),
    )
    contexts = list(response.contexts.contexts)
    if not contexts:
        print("No matching contexts found.")
        return
    for i, ctx in enumerate(contexts, 1):
        source = ctx.source_display_name or ctx.source_uri
        print(f"--- [{i}] {source} (score {ctx.score:.3f}) ---")
        print(ctx.text.strip()[:800])
        print()


def cmd_delete_file(args: argparse.Namespace) -> None:
    init(args)
    corpus_name = find_corpus_name(args.corpus)
    if corpus_name is None:
        sys.exit(f"Corpus '{args.corpus}' does not exist.")
    for f in rag.list_files(corpus_name=corpus_name):
        if f.display_name == args.display_name or f.name == args.display_name:
            rag.delete_file(str(f.name))
            print(f"Deleted {f.display_name}")
            return
    sys.exit(f"No file named '{args.display_name}' in corpus '{args.corpus}'.")


def cmd_delete_corpus(args: argparse.Namespace) -> None:
    init(args)
    corpus_name = find_corpus_name(args.corpus)
    if corpus_name is None:
        sys.exit(f"Corpus '{args.corpus}' does not exist.")
    if not args.yes:
        sys.exit("Refusing to delete without --yes.")
    rag.delete_corpus(corpus_name)
    print(f"Deleted corpus '{args.corpus}'.")


def cmd_test_run(args: argparse.Namespace) -> None:
    import rag_eval

    rag_eval.run_prompts(args.project, args.location, args.corpus, top_k=args.top_k)


def cmd_test_eval(args: argparse.Namespace) -> None:
    import rag_eval

    rag_eval.evaluate(args.project)


def cmd_test_recommend(args: argparse.Namespace) -> None:
    import rag_eval

    rag_eval.recommend(args.project)


def cmd_test_report(args: argparse.Namespace) -> None:
    import rag_report

    path = rag_report.generate()
    print(f"Report written to {path}")
    if args.open:
        import webbrowser

        webbrowser.open(path.as_uri())


def cmd_test_all(args: argparse.Namespace) -> None:
    cmd_test_run(args)
    cmd_test_eval(args)
    cmd_test_recommend(args)
    cmd_test_report(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Vertex AI RAG Engine corpus management for Vallabha VLE.",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"GCP project (default: {DEFAULT_PROJECT})")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"Vertex AI region (default: {DEFAULT_LOCATION})")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help=f"Corpus display name (default: {DEFAULT_CORPUS})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("create-corpus", help="Create the corpus if it does not exist").set_defaults(func=cmd_create_corpus)

    p_upload = sub.add_parser("upload", help="Upload files or folders to the corpus")
    p_upload.add_argument("paths", nargs="+", help="Files or directories to upload")
    p_upload.add_argument("--force", action="store_true", help="Re-upload even if a file with the same name exists")
    p_upload.set_defaults(func=cmd_upload)

    sub.add_parser("list-files", help="List files in the corpus").set_defaults(func=cmd_list_files)
    sub.add_parser("list-corpora", help="List all corpora in the project/region").set_defaults(func=cmd_list_corpora)

    p_query = sub.add_parser("query", help="Run a retrieval query against the corpus")
    p_query.add_argument("text", help="Query text")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of contexts to retrieve (default: 5)")
    p_query.set_defaults(func=cmd_query)

    p_del_file = sub.add_parser("delete-file", help="Delete one file from the corpus by display name")
    p_del_file.add_argument("display_name")
    p_del_file.set_defaults(func=cmd_delete_file)

    p_del_corpus = sub.add_parser("delete-corpus", help="Delete the whole corpus")
    p_del_corpus.add_argument("--yes", action="store_true", help="Confirm deletion")
    p_del_corpus.set_defaults(func=cmd_delete_corpus)

    p_test_run = sub.add_parser("test-run", help="Answer every prompt in test/prompt.json via RAG")
    p_test_run.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per prompt (default: 5)")
    p_test_run.set_defaults(func=cmd_test_run)

    sub.add_parser("test-eval", help="Score results with Opik + DeepEval LLM-as-judge metrics").set_defaults(func=cmd_test_eval)

    sub.add_parser(
        "test-recommend", help="Diagnose low-scoring cases and add concrete tuning mitigations to metrics.json"
    ).set_defaults(func=cmd_test_recommend)

    p_test_report = sub.add_parser("test-report", help="Generate the HTML metrics report (test/report.html)")
    p_test_report.add_argument("--open", action="store_true", help="Open the report in a browser")
    p_test_report.set_defaults(func=cmd_test_report)

    p_test_all = sub.add_parser("test-all", help="test-run + test-eval + test-report in one go")
    p_test_all.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per prompt (default: 5)")
    p_test_all.add_argument("--open", action="store_true", help="Open the report in a browser")
    p_test_all.set_defaults(func=cmd_test_all)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        if "Reauthentication" in str(exc):
            sys.exit(
                "Your Google credentials have expired.\n"
                "Run:  gcloud auth application-default login\n"
                "then re-run this command."
            )
        raise


if __name__ == "__main__":
    main()
