# Audio RAG — Usage Guide

Turn audio recordings into a searchable, question-answerable knowledge base — fully local:

| Stage | Tool | Runs |
|---|---|---|
| Audio → text | OpenAI Whisper (open source) | locally |
| Text → embeddings | sentence-transformers (configurable, default `all-MiniLM-L6-v2`) | locally |
| Embedding storage + search | Chroma vector database | local server |
| Final summarized answer | Ollama LLM (configurable, default `gemma4`) | locally |

Everything is driven through one CLI: `uv run audiorag <command>`.

## One-time setup

```bash
uv sync                          # installs whisper, chromadb, sentence-transformers, ...
brew install ffmpeg              # whisper needs ffmpeg (skip if already installed)
ollama pull gemma4               # the default answer model (any Ollama model works)
```

The first `transcribe` downloads the Whisper model and the first `upload`/`search`
downloads the embedding model; both are cached after that.

## 1. Start the vector database

```bash
uv run audiorag db-start         # starts Chroma at http://127.0.0.1:8765, data in .chroma/
uv run audiorag db-status        # server health + chunk count
uv run audiorag db-stop          # stop the server (data is persisted on disk)
```

The server keeps running in the background until stopped, so you only start it once
per working session (or leave it running).

## 2. Transcribe audio with Whisper

```bash
uv run audiorag transcribe audio_samples/                 # a directory of audio files
uv run audiorag transcribe recording.m4a interview.mp3    # or individual files
```

Writes one `.txt` per audio file into `transcripts/`. Options:

- `--whisper-model tiny|base|small|medium|large` — accuracy vs. speed (default `base`)
- `--out-dir DIR` — where transcripts are written
- `--language en` — force a language instead of auto-detect

Supported audio: mp3, wav, m4a, flac, ogg, aiff, aac, mp4, webm.

## 3. Upload transcripts to the vector DB

```bash
uv run audiorag upload transcripts/                # every .txt in the directory
uv run audiorag upload transcripts/meeting.txt     # or a single file
uv run audiorag list                               # what is stored, chunk counts
```

Each transcript is split into overlapping chunks (200 words, 40-word overlap),
embedded locally, and stored in Chroma. Re-uploading the same file is skipped
unless you pass `--force` (which replaces the old chunks).

## 4. Search: retrieve relevant chunks

```bash
uv run audiorag search "what did the customer say about invoice approval?"
uv run audiorag search "payment terms" --top-k 3
```

Prints the best-matching transcript chunks with their source file and a
relevance score — usable directly, or as input to any downstream tool.

## 5. Ask: retrieved chunks + question → local LLM answer

```bash
uv run audiorag ask "Summarize the customer's invoice processing requirements"
uv run audiorag ask "What integrations were requested?" --top-k 8 --show-chunks
uv run audiorag ask "..." --llm-model llama3        # one-off model override
```

Retrieves the top chunks, sends them with your question to the configured
Ollama model, and prints a grounded, summarized answer. `--show-chunks` also
lists which chunks were used.

## Configuration

Exactly two things are configurable — the answer LLM and the embedding model:

```bash
uv run audiorag config show                                  # effective config + where each value came from
uv run audiorag config set llm_model llama3                  # any model in `ollama list`
uv run audiorag config set embedding_model all-mpnet-base-v2 # any sentence-transformers model
```

Precedence (highest wins):

1. CLI flags: `--llm-model`, `--embedding-model`
2. Environment: `AUDIORAG_LLM_MODEL`, `AUDIORAG_EMBEDDING_MODEL`
3. Config file `audiorag.json` (written by `config set`)
4. Built-in defaults: `gemma4` / `all-MiniLM-L6-v2`

> If you change the embedding model after uploading, re-upload with `--force`
> so stored vectors and query vectors come from the same model.

## Complete example session

```bash
uv run audiorag db-start
uv run audiorag transcribe audio_samples/
uv run audiorag upload transcripts/
uv run audiorag list
uv run audiorag search "invoice approval workflow"
uv run audiorag ask "What are the customer's key requirements for the invoice processing product?"
uv run audiorag db-stop
```

## Running the tests

```bash
uv run pytest                    # unit tests (fast, no models/servers needed)
uv run pytest -m integration     # integration tests: real Chroma server, real
                                 # embeddings, Whisper on synthesized audio,
                                 # and a real Ollama answer (auto-skip when a
                                 # dependency such as Ollama is unavailable)
```
