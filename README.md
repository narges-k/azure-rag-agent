# azure-rag-agent

A small, readable RAG (retrieval-augmented generation) pipeline on Azure, with an agent on top and a
built-in evaluation script. Documents go into Blob Storage, get chunked and embedded, and are indexed in
Azure AI Search. Questions are answered with hybrid retrieval and cited sources.

The language models can come from **Azure OpenAI** or from a **local Ollama server**, so the project also
works without Azure OpenAI quota. Authentication uses **Microsoft Entra ID** (`az login`). There are no API keys
in the code or in the configuration.

> **Status: work in progress.** Chunking, blob upload, index creation and embedding with a local Ollama model
> have been run. Retrieval, the agent and the evaluation are still being validated end to end, and the agent
> loop has so far only been tested with a mock model. Expect small first-run issues (deployment names,
> role assignments).

## Architecture

```
docs/*.pdf|txt|md
   -> Azure Blob Storage            (upload_docs.py)
   -> text extraction + chunking    (ingest.py, chunking.py)
   -> embeddings                    (Azure OpenAI or Ollama)
   -> Azure AI Search index         (BM25 + vector HNSW + optional semantic ranker)
   -> hybrid retrieval              (ask.py)
   -> LLM answer with [1], [2] citations
   -> evaluation: hit@k, MRR, groundedness   (evaluate.py)
   -> agent: the model decides when and what to search   (agent.py)
```

| File | Purpose |
|---|---|
| `upload_docs.py` | Uploads local files from `docs/` to a Blob container |
| `ingest.py` | Reads blobs, extracts text, chunks (about 1200 characters, 200 overlap), embeds, indexes |
| `ask.py` | Hybrid search (keyword + vector, optional semantic reranking), then a cited answer |
| `agent.py` | Tool-calling loop with a `search_documents` tool, step limit and numbered citations |
| `evaluate.py` | Retrieval metrics (hit@k, MRR) and an optional LLM judge for groundedness |
| `common.py` | Configuration and Azure/OpenAI/Ollama clients |
| `chunking.py` | Paragraph-aware chunking with overlap (pure Python, no Azure needed) |

## Prerequisites

- Python 3.10+
- An Azure subscription with: a Storage account, an Azure AI Search service, and (optionally) Azure OpenAI
- [Azure CLI](https://learn.microsoft.com/cli/azure/) and `az login` with the account that owns the resources
- Optional: [Ollama](https://ollama.com) with `llama3.2` and `nomic-embed-text` pulled

### Roles (RBAC) for the signed-in user

| Resource | Role |
|---|---|
| Storage account | Storage Blob Data Contributor |
| Azure AI Search | Search Service Contributor, Search Index Data Contributor, Search Index Data Reader |
| Azure OpenAI (if used) | Cognitive Services OpenAI User |

Role assignments can take up to about 10 minutes to take effect. Note that "Owner" does not grant data-plane
access to Search or Blob data.

## Setup

```bash
git clone https://github.com/<your-user>/azure-rag-agent.git
cd azure-rag-agent
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env with your resource names
az login
```

### Option A: Azure OpenAI

Fill in `AZURE_OPENAI_ENDPOINT` (the base URL only, for example `https://<name>.openai.azure.com`) and the
names of your chat and embedding **deployments**. Set `EMBED_DIMENSIONS` to match the embedding model
(1536 for `text-embedding-3-small`).

### Option B: local Ollama

```
LLM_PROVIDER=ollama
AZURE_OPENAI_CHAT_DEPLOYMENT=llama3.2
AZURE_OPENAI_EMBED_DEPLOYMENT=nomic-embed-text
EMBED_DIMENSIONS=768
AZURE_SEARCH_INDEX=rag-demo-ollama
```

Use a separate index name whenever the embedding dimension changes.

## Usage

Put **public** documents (PDF, TXT or MD) into `docs/`, then:

```bash
python upload_docs.py                       # 1. upload to Blob Storage
python ingest.py                            # 2. chunk, embed, index (safe to re-run: ids are deterministic)
python ask.py "What does the report say about X?"          # 3. ask, add --semantic for reranking
python agent.py "Compare what A and B say about X" -v      # 4. agent, several searches per question
python evaluate.py --k 5 --judge            # 5. measure quality
```

### Evaluation

Edit `eval_questions.json`: each item has a `question` and the `expected_source` file that should be
retrieved (use `null` for a question the documents do not answer). `evaluate.py` reports:

- **hit@k**: the share of questions where the expected source is in the top k results
- **MRR**: mean reciprocal rank of the first correct source
- **groundedness** (with `--judge`): an LLM grades whether the answer is supported by the retrieved sources

Change one variable at a time (chunk size, `k`, `--semantic`) and compare. A chunk-size experiment needs a
new index name.

## Design notes

- **Hybrid retrieval**: BM25 keyword search and vector search are combined by the service (reciprocal rank fusion).
- **Grounding**: the prompt allows only the retrieved sources and requires citations. "I don't know" is an accepted answer.
- **Prompt injection**: retrieved text is treated as untrusted data in the system prompt.
- **Agent guardrails**: a maximum number of steps, and a forced final answer when the budget runs out.
- **Idempotent ingestion**: chunk ids are hashes of file, page and chunk number, so re-runs do not create duplicates.

## Security and data

- Never commit `.env`. It is in `.gitignore`.
- Use public or synthetic documents only. Do not index confidential or employer data in a personal or university subscription.
- Uploaded documents are in `docs/`, which is ignored by git except for a placeholder.

## Cost and cleanup

Azure AI Search has a fixed hourly cost while the service exists. Delete the resource group when you are
done experimenting:

```bash
az group delete --name <resource-group>
```
