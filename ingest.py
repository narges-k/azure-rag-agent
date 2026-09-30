"""Step 2: read documents from Blob Storage -> extract text -> chunk -> embed -> index in Azure AI Search.
Run again any time documents change (chunk ids are deterministic, so uploads are idempotent)."""
import hashlib
import io

from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from pypdf import PdfReader

from chunking import chunk_text
from common import blob_service, embed, env, openai_client, search_client, search_index_client

VECTOR_PROFILE = "vector-profile"
HNSW_CONFIG = "hnsw-config"
SEMANTIC_CONFIG = "semantic-config"


def create_index() -> None:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_no", type=SearchFieldDataType.Int32),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=int(env("EMBED_DIMENSIONS")),
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=HNSW_CONFIG)],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_CONFIG)],
    )
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="content")]),
            )
        ]
    )
    index = SearchIndex(
        name=env("AZURE_SEARCH_INDEX"), fields=fields, vector_search=vector_search, semantic_search=semantic_search
    )
    search_index_client().create_or_update_index(index)
    print("Index ready:", index.name)


def extract_pages(name: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Text files count as page 1."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(1, data.decode("utf-8", errors="ignore"))]


def build_records() -> list[dict]:
    container = blob_service().get_container_client(env("AZURE_STORAGE_CONTAINER"))
    records = []
    for blob in container.list_blobs():
        data = container.download_blob(blob.name).readall()
        for page_no, text in extract_pages(blob.name, data):
            for i, chunk in enumerate(chunk_text(text)):
                key = hashlib.sha1(f"{blob.name}|{page_no}|{i}".encode()).hexdigest()
                records.append(
                    {"id": key, "content": chunk, "source": blob.name, "page": page_no, "chunk_no": i}
                )
    return records


def main() -> None:
    create_index()
    records = build_records()
    print(f"{len(records)} chunks to embed")
    client = openai_client()

    for start in range(0, len(records), 16):  # embed in small batches
        batch = records[start : start + 16]
        vectors = embed(client, [r["content"] for r in batch])
        for record, vector in zip(batch, vectors):
            record["contentVector"] = vector

    sc = search_client()
    for start in range(0, len(records), 100):
        results = sc.upload_documents(documents=records[start : start + 100])
        failed = [r for r in results if not r.succeeded]
        if failed:
            print("Failed:", [(f.key, f.error_message) for f in failed][:3])
    print("Indexed", len(records), "chunks")


if __name__ == "__main__":
    main()
