"""Shared configuration and Azure clients. All authentication uses Microsoft Entra ID
(your `az login` session) for Azure Search and Blob Storage. The language models can come from
Azure OpenAI (Entra ID) or from a local Ollama server (LLM_PROVIDER=ollama). No Azure keys are used."""
import os

from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Token audience for Azure OpenAI. If you get a 401, try "https://ai.azure.com/.default" instead.
OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"


def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable {name}. See .env.example")
    return value


_credential = None


def credential() -> AzureCliCredential:
    """Use ONLY the account from `az login` (avoids picking up another account, e.g. Chalmers)."""
    global _credential
    if _credential is None:
        _credential = AzureCliCredential()
    return _credential


def openai_client() -> OpenAI:
    """Client for chat + embeddings. LLM_PROVIDER=ollama uses a local Ollama server (OpenAI-compatible API,
    no cloud, no quota). Otherwise Azure OpenAI with an Entra ID token (v1 API)."""
    if os.getenv("LLM_PROVIDER", "azure").lower() == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAI(base_url=base_url, api_key="ollama", timeout=300)  # api_key is ignored by Ollama
    token_provider = get_bearer_token_provider(credential(), OPENAI_SCOPE)
    endpoint = env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    return OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=token_provider)


def search_index_client() -> SearchIndexClient:
    return SearchIndexClient(endpoint=env("AZURE_SEARCH_ENDPOINT"), credential=credential())


def search_client() -> SearchClient:
    return SearchClient(
        endpoint=env("AZURE_SEARCH_ENDPOINT"),
        index_name=env("AZURE_SEARCH_INDEX"),
        credential=credential(),
    )


def blob_service() -> BlobServiceClient:
    return BlobServiceClient(account_url=env("AZURE_STORAGE_ACCOUNT_URL"), credential=credential())


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Create embeddings using Ollama or Azure OpenAI."""

    if os.getenv("LLM_PROVIDER", "azure").lower() == "ollama":
        import requests

        url = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/embed"

        response = requests.post(
            url,
            json={
                "model": env("AZURE_OPENAI_EMBED_DEPLOYMENT"),
                "input": texts,
            },
            timeout=300,
        )

        response.raise_for_status()

        return response.json()["embeddings"]

    # Azure OpenAI
    response = client.embeddings.create(
        model=env("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        input=texts,
    )

    return [
        item.embedding
        for item in sorted(response.data, key=lambda d: d.index)
    ]