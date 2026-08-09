# vallabha-rag

CLI for managing a **Vertex AI RAG Engine** corpus in the **Vallabha VLE** GCP project
(`vallabha-systems-vle`, region `europe-west4`).

Built as a [uv](https://docs.astral.sh/uv/) Python project. Uses Application Default
Credentials (ADC) and the RAG Engine **default** embedding model, chunking, and
indexing strategies (no custom `TransformationConfig` is passed, so service defaults apply).

Beyond corpus management, the CLI ships an **evaluation suite**: test prompts are
answered via retrieval + Gemini generation, scored by **Opik** and **DeepEval**
LLM-as-judge metrics, diagnosed into experimental `[SYSTEM]`/`[PROMPT]`-scoped
tuning mitigations, and rendered into a tabbed HTML report.

---

## Architecture (C4 model)

### Level 1 — System Context

```mermaid
C4Context
    title System Context — Vallabha VLE RAG ingestion & evaluation

    Person(user, "Developer / Operator", "Uploads documents, runs retrieval queries, and evaluates RAG quality from the terminal")

    System(ragcli, "vallabha-rag CLI", "uv-managed Python CLI (`uv run rag ...`): corpus lifecycle (create, upload, list, query, delete) + evaluation pipeline (test-run, test-eval, test-recommend, test-report)")

    System_Ext(vertex, "Vertex AI RAG Engine", "Google Cloud managed RAG service in europe-west4: parses, chunks, embeds, and indexes documents")
    System_Ext(gemini, "Gemini on Vertex AI", "gemini-3.5-flash (global endpoint): grounded answer generation, LLM-as-judge scoring, and tuning-mitigation generation")
    System_Ext(ollama, "Ollama (local)", "Optional --backend ollama: a local model (default gemma4) replaces cloud Gemini for generation, judging, and mitigations")
    System_Ext(gcloud, "Google Auth (ADC)", "Application Default Credentials issued via `gcloud auth application-default login`")

    Rel(user, ragcli, "Runs commands", "terminal")
    Rel(ragcli, vertex, "Creates corpus, uploads files, retrieves contexts", "gRPC / HTTPS")
    Rel(ragcli, gemini, "Generates answers, judges metrics, drafts mitigations", "HTTPS")
    Rel(ragcli, ollama, "Same roles when --backend ollama", "HTTP localhost:11434")
    Rel(ragcli, gcloud, "Obtains OAuth2 access tokens", "local credential file")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Level 2 — Containers

```mermaid
C4Container
    title Container view — CLI and Google Cloud

    Person(user, "Developer / Operator")

    System_Boundary(local, "Local machine (uv project)") {
        Container(cli, "rag CLI", "Python 3.12, argparse, rag_cli.py", "Entry point `rag = rag_cli:main`; corpus lifecycle + test-* subcommands")
        Container(evalmod, "Evaluation runner", "rag_eval.py", "run_prompts (retrieve+generate), evaluate (Opik+DeepEval scoring), recommend (scoped mitigations)")
        Container(report, "Report generator", "rag_report.py", "Renders metrics.json into a self-contained tabbed HTML report")
        Container(sdk, "google-cloud-aiplatform SDK", "vertexai.rag module", "Wraps VertexRagDataService and VertexRagService gRPC APIs")
        Container(judges, "Judge libraries", "opik + deepeval (via litellm / google-genai)", "LLM-as-judge metrics: relevance, hallucination, faithfulness, context precision/recall")
        ContainerDb(docs, "docs/ folder", "PDF, DOCX, TXT, MD, ...", "Source documents to ingest")
        ContainerDb(testdir, "test/ folder", "prompt.json, results/*.json, report.html", "Eval prompts + expected answers; generated results, metrics, mitigations, report")
        ContainerDb(adc, "ADC credentials", "~/.config/gcloud/...json", "OAuth2 refresh token (gcloud ADC login)")
        Container_Ext(ollamasrv, "Ollama server", "localhost:11434", "Optional local backend (--backend ollama): gemma4 (or any pulled model) for generation, judging, and mitigations")
    }

    System_Boundary(gcp, "GCP project vallabha-systems-vle") {
        Container(ragdata, "VertexRagDataService", "Vertex AI API, europe-west4", "CreateRagCorpus, UploadRagFile, ListRagFiles, Delete*")
        Container(ragquery, "VertexRagService", "Vertex AI API, europe-west4", "RetrieveContexts (semantic search)")
        Container(gemini, "Gemini gemini-3.5-flash", "Vertex AI global endpoint", "Grounded generation, judge verdicts, mitigation drafting")
        ContainerDb(corpus, "RAG corpus 'vallabha-vle-docs'", "Managed vector store", "Default chunking + default embedding model")
    }

    Rel(user, cli, "uv run rag <command>")
    Rel(cli, docs, "Reads files")
    Rel(cli, sdk, "Calls")
    Rel(cli, evalmod, "test-run / test-eval / test-recommend")
    Rel(cli, report, "test-report")
    Rel(evalmod, testdir, "Reads prompt.json, writes results + metrics")
    Rel(report, testdir, "Reads metrics.json, writes report.html")
    Rel(evalmod, judges, "Scores each case")
    Rel(evalmod, gemini, "Answer generation + mitigations (--backend vertex)", "google-genai")
    Rel(judges, gemini, "Judge calls (--backend vertex)", "litellm vertex_ai / google-genai")
    Rel(evalmod, ollamasrv, "Answer generation + mitigations (--backend ollama)", "litellm ollama_chat")
    Rel(judges, ollamasrv, "Judge calls (--backend ollama)", "deepeval OllamaModel / litellm")
    Rel(sdk, adc, "Signs requests with")
    Rel(sdk, ragdata, "Corpus + file management", "gRPC")
    Rel(sdk, ragquery, "Retrieval queries", "gRPC")
    Rel(ragdata, corpus, "Parses, chunks, embeds, indexes")
    Rel(ragquery, corpus, "Vector similarity search")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Level 3 — Components (rag_cli.py)

```mermaid
C4Component
    title Component view — rag_cli.py

    Container_Boundary(cli, "rag CLI") {
        Component(parser, "Argument parser", "argparse", "Global flags --project / --location / --corpus + subcommands; env-var overrides VALLABHA_RAG_*")
        Component(corpushelpers, "Corpus helpers", "find_corpus_name / ensure_corpus_name", "Resolves display name to resource name; creates corpus on first use")
        Component(collector, "File collector", "collect_files", "Expands files/dirs recursively, filters to supported suffixes")
        Component(upload, "upload", "cmd_upload", "Idempotent: skips files already in corpus unless --force; per-file error handling")
        Component(listing, "list-files / list-corpora", "cmd_list_*", "Inspection commands")
        Component(query, "query", "cmd_query", "retrieval_query with configurable --top-k")
        Component(deletion, "delete-file / delete-corpus", "cmd_delete_*", "Cleanup; corpus deletion requires --yes")
        Component(testcmds, "test-run / test-eval / test-recommend / test-report / test-all", "cmd_test_*", "Evaluation pipeline stages; test-all chains all four")
    }

    System_Ext(vertex, "Vertex AI RAG Engine")
    System_Ext(evalmod, "rag_eval.py + rag_report.py")

    Rel(parser, upload, "dispatches")
    Rel(parser, listing, "dispatches")
    Rel(parser, query, "dispatches")
    Rel(parser, deletion, "dispatches")
    Rel(parser, testcmds, "dispatches")
    Rel(upload, collector, "collects paths")
    Rel(upload, corpushelpers, "ensures corpus")
    Rel(upload, vertex, "rag.upload_file")
    Rel(listing, vertex, "rag.list_files / list_corpora")
    Rel(query, vertex, "rag.retrieval_query")
    Rel(deletion, vertex, "rag.delete_file / delete_corpus")
    Rel(testcmds, evalmod, "lazy import")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Level 3 — Components (evaluation subsystem)

```mermaid
C4Component
    title Component view — rag_eval.py and rag_report.py

    Container_Boundary(evalmod, "rag_eval.py") {
        Component(runner, "run_prompts", "retrieval + generation", "Per prompt: retrieval_query (per-prompt top_k override supported) then grounded Gemini answer -> rag_results.json")
        Component(scorer, "evaluate", "Opik + DeepEval scoring", "Opik: AnswerRelevance, Hallucination, ContextPrecision, ContextRecall. DeepEval: AnswerRelevancy, Faithfulness, ContextualPrecision, ContextualRecall, Hallucination -> metrics.json")
        Component(recommender, "recommend", "experimental mitigations", "Flags scores outside the healthy band, diagnoses root cause (retrieval / generation / test-design / metric artifact), emits [SYSTEM] or [PROMPT] scoped fixes + run-level recommendations")
    }

    Container_Boundary(reportmod, "rag_report.py") {
        Component(generator, "generate", "HTML renderer", "Tabs (Opik | DeepEval | Test Cases), score chips, 'How to fix' callouts, run-level recommendations, glossary + scope-legend + experimental-disclaimer footer")
    }

    System_Ext(ragquery, "VertexRagService (europe-west4)")
    System_Ext(gemini, "Gemini gemini-3.5-flash (global)")
    ComponentDb(artifacts, "test/ artifacts", "prompt.json, rag_results.json, metrics.json, report.html")

    Rel(runner, ragquery, "RetrieveContexts(top_k)")
    Rel(runner, gemini, "grounded answer generation")
    Rel(scorer, gemini, "LLM-as-judge calls", "opik/litellm + deepeval/google-genai")
    Rel(recommender, gemini, "root-cause diagnosis + mitigations")
    Rel(runner, artifacts, "writes rag_results.json")
    Rel(scorer, artifacts, "writes metrics.json")
    Rel(recommender, artifacts, "augments metrics.json with mitigations")
    Rel(generator, artifacts, "reads metrics.json, writes report.html")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### End-to-end flows

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer
    participant CLI as rag CLI (uv run rag)
    participant ADC as ADC credentials
    participant Data as VertexRagDataService
    participant Corpus as RAG corpus (europe-west4)

    Note over Dev,Corpus: Ingestion — uv run rag upload docs
    Dev->>CLI: uv run rag upload docs
    CLI->>ADC: get OAuth2 access token
    ADC-->>CLI: token
    CLI->>Data: ListRagCorpora (find "vallabha-vle-docs")
    alt corpus missing
        CLI->>Data: CreateRagCorpus (default embedding + indexing)
        Data-->>CLI: corpus resource name
    end
    CLI->>Data: ListRagFiles (dedupe against already-uploaded)
    loop each new file in docs/
        CLI->>Data: UploadRagFile (default chunking)
        Data->>Corpus: parse → chunk → embed → index
        Data-->>CLI: RagFile resource
    end
    CLI-->>Dev: summary (uploaded / skipped / failed)
```

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer
    participant CLI as rag CLI
    participant Eval as rag_eval.py
    participant Query as VertexRagService
    participant Gemini as Gemini (Vertex AI, global)
    participant Judges as Opik + DeepEval judges

    Note over Dev,Judges: Evaluation — uv run rag test-all
    Dev->>CLI: uv run rag test-all --open
    CLI->>Eval: run_prompts()
    loop each prompt in test/prompt.json
        Eval->>Query: RetrieveContexts(question, top_k — per-prompt "top_k" override wins)
        Query-->>Eval: top-k chunks
        Eval->>Gemini: generate grounded answer from chunks
        Gemini-->>Eval: answer
    end
    Eval-->>CLI: test/results/rag_results.json
    CLI->>Eval: evaluate()
    loop each answered case
        Eval->>Judges: score(question, answer, expected, contexts)
        Judges->>Gemini: LLM-as-judge calls (relevance, hallucination, precision, recall, faithfulness)
        Gemini-->>Judges: verdicts + reasoning
        Judges-->>Eval: metric scores
    end
    Eval-->>CLI: test/results/metrics.json
    CLI->>Eval: recommend()  (experimental)
    loop each case with a low metric
        Eval->>Gemini: diagnose root cause + draft mitigations
        Gemini-->>Eval: [SYSTEM]/[PROMPT]-scoped fixes (or "metric artifact - no action")
    end
    Eval->>Gemini: synthesize run-level recommendations
    Gemini-->>Eval: prioritized tuning actions
    Eval-->>CLI: metrics.json + mitigations
    CLI->>CLI: rag_report.generate() → test/report.html
    CLI-->>Dev: tabbed HTML report (Opik | DeepEval | Test Cases) with "How to fix" callouts
    Note over Dev,Judges: Mitigations are experimental — human judgement required before applying
    Note over Gemini: With --backend ollama, every Gemini call above goes to local Ollama (default gemma4) instead — retrieval stays on Vertex AI RAG Engine
```

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer
    participant CLI as rag CLI
    participant Query as VertexRagService
    participant Corpus as RAG corpus

    Note over Dev,Corpus: Retrieval — uv run rag query "..."
    Dev->>CLI: uv run rag query "What is the ICT master plan about?" --top-k 3
    CLI->>Query: RetrieveContexts(text, top_k)
    Query->>Corpus: embed query → vector similarity search
    Corpus-->>Query: top-k chunks + scores
    Query-->>CLI: contexts
    CLI-->>Dev: ranked chunks with source file + score
```

---

## Setup

```sh
uv sync                                 # install dependencies + the `rag` entry point
gcloud auth application-default login   # once, or whenever the ADC token expires
```

Requires the Vertex AI API (`aiplatform.googleapis.com`) to be enabled in the project
(already enabled for `vallabha-systems-vle`).

## Usage

### Upload documents

```sh
# Upload everything in docs/ — creates the corpus on first run,
# skips files already in the corpus (idempotent, safe to re-run)
uv run rag upload docs

# Upload specific files and/or folders
uv run rag upload report.pdf ./more-docs

# Re-upload files that already exist in the corpus
uv run rag upload docs --force
```

### Inspect

```sh
uv run rag list-corpora      # all corpora in the project/region
uv run rag list-files        # files in the default corpus
```

### Query (retrieval sanity check)

```sh
uv run rag query "What is the ICT master plan about?"
uv run rag query "Broadband targets for rural India" --top-k 10
```

### Cleanup

```sh
uv run rag delete-file ict-masterplan-final-.pdf   # by display name
uv run rag delete-corpus --yes                     # deletes the corpus and all its files
```

### Create the corpus without uploading

```sh
uv run rag create-corpus
```

### Evaluate RAG quality (Opik + DeepEval)

Test prompts live in `test/prompt.json` (factual, synthesis, cross-document, and
out-of-scope categories, each with a hand-written expected answer).

```sh
uv run rag test-run              # answer every prompt via retrieval + Gemini generation
uv run rag test-eval             # score answers with Opik + DeepEval LLM-as-judge metrics
uv run rag test-recommend        # diagnose low scores -> concrete tuning mitigations (experimental)
uv run rag test-report --open    # build test/report.html and open it in the browser

uv run rag test-all --open       # all four stages in one go
uv run rag test-run --top-k 10   # experiment with retrieval depth
```

A prompt entry in `prompt.json` may carry its own `"top_k": 10` field, which overrides
`--top-k` for that prompt alone — the smallest-scope tuning knob. Auto-generated
mitigations are tagged `[PROMPT]` (change one entry in `prompt.json`) or `[SYSTEM]`
(default top-k, chunking + re-upload, grounding prompt, model) so engineers know the
blast radius. **They are experimental — apply human judgement, prefer PROMPT-scope
fixes, and validate with a fresh `test-all` run.**

Artifacts land in `test/`:

- `test/results/rag_results.json` — question, generated answer, retrieved chunks + scores
- `test/results/metrics.json` — per-case and averaged Opik/DeepEval scores with judge reasoning
- `test/report.html` — self-contained tabbed report (Opik | DeepEval | Test Cases) with a
  metric glossary, prompt/chunk tuning guidance, and references in the footer

Metrics: answer relevance/relevancy, hallucination, faithfulness (DeepEval),
context(ual) precision, and context(ual) recall. The generation and judge model is
`gemini-3.5-flash` on Vertex AI (override with `VALLABHA_RAG_EVAL_MODEL`).

#### Local model backend (Ollama)

`test-run`, `test-eval`, `test-recommend`, and `test-all` accept `--backend` and
`--model` to swap the cloud frontier model for a local Ollama model — no cloud LLM
calls for generation, judging, or mitigations (document **retrieval still uses the
Vertex AI RAG Engine corpus**, so ADC is still needed):

```sh
uv run rag test-all --backend ollama --open        # local gemma4 end to end
uv run rag test-run --backend ollama --model gemma4
uv run rag test-eval --backend ollama              # judge with local gemma4
uv run rag test-run --backend vertex --model gemini-3-flash-preview   # other cloud model
```

Requires [Ollama](https://ollama.com) running locally with the model pulled
(`ollama pull gemma4`). Defaults: backend `vertex`; ollama model `gemma4`;
server `http://localhost:11434`. Env overrides: `VALLABHA_RAG_EVAL_BACKEND`,
`VALLABHA_RAG_OLLAMA_MODEL`, `VALLABHA_RAG_OLLAMA_URL`. Note that small local
models are noticeably weaker judges than frontier models — expect noisier scores
and treat cross-backend comparisons with care.

## Defaults and overrides

Every command accepts the global flags below (place them **before** the subcommand),
or set the corresponding environment variable.

| Setting | Default                | Flag         | Env var                 |
|---------|------------------------|--------------|-------------------------|
| Project | `vallabha-systems-vle` | `--project`  | `VALLABHA_RAG_PROJECT`  |
| Region  | `europe-west4`         | `--location` | `VALLABHA_RAG_LOCATION` |
| Corpus  | `vallabha-vle-docs`    | `--corpus`   | `VALLABHA_RAG_CORPUS`   |

Example:

```sh
uv run rag --corpus experiments upload docs
VALLABHA_RAG_LOCATION=europe-west3 uv run rag list-corpora
```

## Supported file types

`.pdf` `.txt` `.md` `.html` `.doc` `.docx` `.ppt` `.pptx` `.xls` `.xlsx` `.csv` `.json`

Other files in an uploaded folder are silently ignored; a nonexistent path prints a warning.

## Project layout

```
RAG/
├── docs/                       # source documents to ingest
├── test/
│   ├── prompt.json             # evaluation prompts + expected answers
│   ├── results/                # rag_results.json, metrics.json (generated)
│   └── report.html             # tabbed metrics report (generated)
├── rag_cli.py                  # CLI entry point (installed as the `rag` command)
├── rag_eval.py                 # retrieval+generation runner and Opik/DeepEval scoring
├── rag_report.py               # HTML report generator
├── pyproject.toml              # uv project; entry point rag = rag_cli:main
└── README.md
```
