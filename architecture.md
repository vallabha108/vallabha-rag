# Audio RAG — Architecture

A fully local Retrieval-Augmented Generation pipeline over audio recordings:
Whisper transcribes audio, a sentence-transformers model embeds transcript
chunks into a local Chroma vector database, and an Ollama-hosted LLM produces
grounded, summarized answers.

## Level 1 — System Context

```mermaid
C4Context
    title System Context — Audio RAG

    Person(user, "User", "Has audio recordings (meetings, customer calls) and wants to search them and ask questions")

    System(audiorag, "Audio RAG CLI", "uv run audiorag — transcribe, store, search and answer, all on the local machine")

    System_Ext(ollama, "Ollama", "Local LLM runtime serving the configured answer model (default: gemma4)")
    System_Ext(hf, "Hugging Face Hub", "One-time downloads of the Whisper and embedding model weights (cached locally afterwards)")

    Rel(user, audiorag, "Runs commands: transcribe / db-start / upload / search / ask")
    Rel(audiorag, ollama, "Sends query + retrieved chunks", "HTTP, localhost")
    Rel(audiorag, hf, "Downloads model weights once", "HTTPS")
```

## Level 2 — Containers

```mermaid
C4Container
    title Container Diagram — Audio RAG

    Person(user, "User")

    System_Boundary(local, "Local machine") {
        Container(cli, "audiorag CLI", "Python (argparse)", "Single entry point for the whole pipeline; wires config into every stage")
        Container(whisper, "Whisper", "openai-whisper (PyTorch)", "Speech-to-text; writes .txt transcripts to transcripts/")
        Container(embed, "Embedder", "sentence-transformers", "Text chunks -> normalized vectors (configurable model)")
        ContainerDb(chroma, "Chroma server", "chroma run, HTTP :8765", "Stores chunk text + embeddings + metadata; cosine HNSW search; persists to .chroma/")
        Container(ollamac, "Ollama daemon", "ollama", "Runs the configured LLM (default gemma4) for answer generation")
        ContainerDb(files, "File system", "transcripts/, audiorag.json", "Transcript files and the two-key config file")
    }

    Rel(user, cli, "audio files, queries")
    Rel(cli, whisper, "transcribe(audio)")
    Rel(whisper, files, "writes transcript .txt")
    Rel(cli, embed, "embed(chunks | query)")
    Rel(cli, chroma, "add / query / get / delete", "HTTP")
    Rel(cli, ollamac, "chat(query + chunks)", "HTTP")
    Rel(cli, files, "reads transcripts + audiorag.json")
```

## Level 3 — Components (`audio_rag` package)

```mermaid
C4Component
    title Component Diagram — audio_rag package

    Container_Boundary(pkg, "audio_rag") {
        Component(cli, "cli.py", "argparse", "Subcommands: transcribe, db-start/stop/status, upload, list, search, ask, config")
        Component(config, "config.py", "dataclass + JSON", "Two configurable keys (llm_model, embedding_model); precedence CLI > env > file > default; fixed pipeline constants")
        Component(transcribe, "transcribe.py", "whisper", "Audio discovery + transcription to .txt")
        Component(chunking, "chunking.py", "pure python", "Word-window chunking: 200 words, 40-word overlap")
        Component(embeddings, "embeddings.py", "sentence-transformers", "Lazy-loaded Embedder(model_name).embed(texts)")
        Component(vectordb, "vectordb.py", "chromadb + subprocess", "Server lifecycle (pid file, health wait) + HttpClient + collection access")
        Component(ingest, "ingest.py", "python", "upload_transcript: dedup by source, chunk, embed, add; list_sources")
        Component(search, "search.py", "python", "search: embed query, top-k cosine query, distance -> score")
        Component(answer, "answer.py", "ollama", "build_prompt(context + query) -> ollama.chat -> Answer")
    }

    Rel(cli, config, "loads + applies flag overrides")
    Rel(cli, transcribe, "transcribe command")
    Rel(cli, ingest, "upload command")
    Rel(cli, search, "search / ask commands")
    Rel(cli, answer, "ask command")
    Rel(cli, vectordb, "db-* commands, collection handle")
    Rel(ingest, chunking, "chunk_text")
    Rel(ingest, embeddings, "embed(chunks)")
    Rel(search, embeddings, "embed(query)")
    Rel(answer, search, "consumes Chunk results")
```

## Ingestion Pipeline — flow and configuration

How audio becomes searchable vectors, and where each configuration value
enters the pipeline (dashed boxes = config points):

```mermaid
flowchart TB
    subgraph ingestion["INGESTION PIPELINE  (audiorag transcribe → audiorag upload)"]
        direction TB
        A["Audio file<br/>(.m4a / .mp3 / .wav ...)"] --> B["Whisper speech-to-text<br/><i>transcribe.py</i>"]
        B --> C["Transcript file<br/>transcripts/&lt;name&gt;.txt"]
        C --> D["Dedup check<br/>source already in DB?<br/><i>ingest.py</i>"]
        D -- "exists & no --force" --> SKIP["Skip (unchanged)"]
        D -- "new or --force" --> E["Chunking<br/>200-word windows,<br/>40-word overlap<br/><i>chunking.py</i>"]
        E --> F["Embedding<br/>sentence-transformers<br/><i>embeddings.py</i>"]
        F --> G[("Chroma collection<br/>'audio_transcripts'<br/>ids + text + vectors + metadata<br/>{source, chunk_index, embedding_model}")]
    end

    W1["⚙ --whisper-model<br/>(tiny/base/small/medium/large,<br/>default: base)"] -.-> B
    W2["⚙ --language<br/>(default: auto-detect)"] -.-> B
    W3["⚙ --out-dir<br/>(default: transcripts/)"] -.-> C
    E1["⚙ embedding_model<br/>--embedding-model flag ><br/>AUDIORAG_EMBEDDING_MODEL env ><br/>audiorag.json > all-MiniLM-L6-v2"] -.-> F
    S1["⚙ fixed: chunk size 200 / overlap 40<br/>(config.py constants)"] -.-> E
    S2["⚙ fixed: Chroma at 127.0.0.1:8765,<br/>data in .chroma/ (db-start)"] -.-> G
```

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as audiorag CLI
    participant W as Whisper
    participant CH as chunking.py
    participant EM as Embedder (sentence-transformers)
    participant DB as Chroma server (:8765)

    U->>CLI: audiorag transcribe audio_samples/
    CLI->>W: load model (whisper base), transcribe each file
    W-->>CLI: text
    CLI->>CLI: write transcripts/<name>.txt

    U->>CLI: audiorag upload transcripts/
    CLI->>CLI: Config.load() → embedding_model
    CLI->>DB: get(where source = file)  — dedup check
    alt already uploaded and not --force
        DB-->>CLI: chunks exist → skip
    else new or --force
        CLI->>CH: chunk_text(text, 200, 40)
        CH-->>CLI: chunks[]
        CLI->>EM: embed(chunks) with configured model
        EM-->>CLI: normalized vectors[]
        CLI->>DB: add(ids, documents, embeddings, metadata)
        DB-->>CLI: stored
    end
    CLI-->>U: "uploaded N chunks (embeddings: <model>)"
```

## Retrieval Pipeline — flow and configuration

How a question becomes a grounded answer (`search` stops after retrieval;
`ask` continues into the LLM):

```mermaid
flowchart TB
    subgraph retrieval["RETRIEVAL PIPELINE  (audiorag search / audiorag ask)"]
        direction TB
        Q["User query<br/>'summarize the invoice requirements'"] --> QE["Embed query<br/>same embedding model<br/>as ingestion<br/><i>embeddings.py</i>"]
        QE --> VS["Vector search<br/>cosine similarity, top-k<br/><i>search.py</i>"]
        DB[("Chroma collection<br/>'audio_transcripts'")] --> VS
        VS --> CK["Ranked chunks<br/>text + source + score"]
        CK -- "audiorag search" --> OUT1["Printed chunks<br/>(for downstream use)"]
        CK -- "audiorag ask" --> P["Prompt assembly<br/>system prompt + context<br/>excerpts + question<br/><i>answer.py</i>"]
        P --> LLM["Local LLM via Ollama<br/>grounded, summarized answer"]
        LLM --> OUT2["Answer<br/>(+ sources with --show-chunks)"]
    end

    C1["⚙ embedding_model<br/>--embedding-model flag ><br/>AUDIORAG_EMBEDDING_MODEL env ><br/>audiorag.json > all-MiniLM-L6-v2<br/>(must match ingestion)"] -.-> QE
    C2["⚙ --top-k<br/>(default: 5)"] -.-> VS
    C3["⚙ llm_model<br/>--llm-model flag ><br/>AUDIORAG_LLM_MODEL env ><br/>audiorag.json > gemma4"] -.-> LLM
```

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as audiorag CLI
    participant EM as Embedder (sentence-transformers)
    participant DB as Chroma server (:8765)
    participant OL as Ollama (LLM)

    U->>CLI: audiorag ask "question" [--top-k 5]
    CLI->>CLI: Config.load() → embedding_model, llm_model
    CLI->>EM: embed(query)
    EM-->>CLI: query vector
    CLI->>DB: query(query_embeddings, n_results = top-k)
    DB-->>CLI: documents + metadata + cosine distances
    CLI->>CLI: distance → score (1 − d), build Chunk list
    CLI->>OL: chat(llm_model, system prompt + context excerpts + question)
    OL-->>CLI: grounded summarized answer
    CLI-->>U: answer (+ chunk sources with --show-chunks)
```

## Configuration model

Only two values are user-configurable; both flow through `config.py` with this
precedence (highest wins):

| Key | CLI flag | Env var | `audiorag.json` | Default |
|---|---|---|---|---|
| `llm_model` | `--llm-model` | `AUDIORAG_LLM_MODEL` | `config set llm_model X` | `gemma4` |
| `embedding_model` | `--embedding-model` | `AUDIORAG_EMBEDDING_MODEL` | `config set embedding_model X` | `all-MiniLM-L6-v2` |

Everything else is a deliberate one-time convention in `config.py`:
Whisper model `base` (per-run override via `--whisper-model`), Chroma at
`127.0.0.1:8765` with data in `.chroma/`, collection `audio_transcripts`,
chunking 200/40 words.

**Consistency rule:** the embedding model used at query time must match the one
used at ingestion (each chunk records its `embedding_model` in metadata; after
changing it, re-upload with `--force`).

## Data at rest

| Location | Contents |
|---|---|
| `transcripts/*.txt` | Whisper output, one file per audio recording |
| `.chroma/` | Chroma persistence (SQLite + HNSW index), `chroma.pid`, `chroma.log` |
| `audiorag.json` | The two configurable keys, written by `audiorag config set` |
