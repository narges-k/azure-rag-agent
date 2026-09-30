# azure-rag-agent

A small RAG (retrieval-augmented generation) project on Azure, with a simple agent and an evaluation script.

You put documents in, ask questions, and get answers with sources. The models can run on **Azure OpenAI**
or locally with **Ollama**. Login uses `az login`, so there are no API keys in the code.

- Ollama is enough for small project. You should install first and download LLM model and embedded model and use them.
- For using Azure services freely, you can create a free trial version easily.

## How it works

```
documents -> Blob Storage -> split into chunks -> embeddings -> Azure AI Search index
question  -> search the index (keyword + vector) -> LLM answers with sources [1], [2]
```

| File | What it does |
|---|---|
| `upload_docs.py` | Uploads files from `docs/` to Blob Storage |
| `ingest.py` | Splits the text into chunks, creates embeddings, saves them in Azure AI Search |
| `ask.py` | Searches the index and answers with sources |
| `agent.py` | An agent that decides when to search, and can search more than once |
| `evaluate.py` | Measures search quality (hit@k, MRR) and answer quality (groundedness) |
| `chunking.py` | Splits text into chunks |
| `common.py` | Settings and connections to Azure and Ollama |

## What you need

- Python 3.10 or newer
- Azure: a Storage account and an Azure AI Search service (and Azure OpenAI if you use it)
- [Azure CLI](https://learn.microsoft.com/cli/azure/), then run `az login`
- Optional: [Ollama](https://ollama.com) with `llama3.2` and `nomic-embed-text`

Roles you need on your Azure account:

| Resource | Role |
|---|---|
| Storage account | Storage Blob Data Contributor |
| Azure AI Search | Search Service Contributor, Search Index Data Contributor, Search Index Data Reader |
| Azure OpenAI | Cognitive Services OpenAI User |

New roles can take about 10 minutes to start working.

## Setup

```bash
git clone https://github.com/narges-k/azure-rag-agent.git
cd azure-rag-agent
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
az login
```

Then open `.env` and fill in your resource names.

**Using Azure OpenAI:** set the endpoint (base URL only) and your chat and embedding deployment names.
Set `EMBED_DIMENSIONS=1536` for `text-embedding-3-small`.

**Using Ollama instead:**

```
LLM_PROVIDER=ollama
AZURE_OPENAI_CHAT_DEPLOYMENT=llama3.2
AZURE_OPENAI_EMBED_DEPLOYMENT=nomic-embed-text
EMBED_DIMENSIONS=768
AZURE_SEARCH_INDEX=rag-demo-ollama
```

Use a new index name whenever the embedding size changes.

## Run it

Put some **public** documents (PDF, TXT or MD) in the `docs/` folder, then:

```bash
python upload_docs.py                                   # 1. upload
python ingest.py                                        # 2. index (safe to run again)
python ask.py "What does the report say about X?"       # 3. ask a question
python agent.py "Compare A and B on X" -v               # 4. use the agent
python evaluate.py --k 5 --judge                        # 5. measure quality
```

### Why evaluating a RAG system is hard

Evaluating a RAG system is challenging, because there are several parts that can fail, and a good score on
one part does not mean the whole system is good. We need to look at different aspects:

- **Retrieval:** does the search find the right passages? (hit@k, MRR)
- **Answer quality:** is the answer correct, complete and based on the sources? (groundedness)
- **Missing answers:** does the system say "I don't know" when the documents have no answer?
- **Speed and cost:** how long does an answer take, and how many tokens does it use?
- **Test data:** a few questions are not enough. The results depend on which questions we choose.

This project only measures the first two parts. Good next steps are more test questions, a larger
document set, and human review of a sample of the answers.

## Good to know

- The answer uses only the retrieved sources and must cite them. "I don't know" is allowed.
- Retrieved text is treated as untrusted, to reduce prompt injection.
- The agent has a step limit, so it cannot loop forever.
- Running `ingest.py` again does not create duplicates.

te --name <resource-group>
```
