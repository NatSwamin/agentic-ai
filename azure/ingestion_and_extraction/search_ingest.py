# search_ingest.py
# Purpose: Create Azure AI Search index + upload embeddings + run vector search
# Works directly with extractor.py + embedder.py + test_pipeline.py

import json
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


# ---------------------------------------------------------
# 1. Load secrets from Key Vault
# ---------------------------------------------------------

KV_URL = "https://azure-openai-kv.vault.azure.net/"  # <-- your vault

credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=KV_URL, credential=credential)

search_endpoint = secret_client.get_secret("AZURE-SEARCH-ENDPOINT").value
search_key = secret_client.get_secret("AZURE-SEARCH-KEY").value

print("Loaded Azure AI Search secrets from Key Vault")


# ---------------------------------------------------------
# 2. Load your ingestion output (from test_pipeline.py)
# ---------------------------------------------------------

with open("ingestion_output.json", "r") as f:
    result = json.load(f)

chunks = result["chunks"]
embeddings = result["embeddings"]
source = result["source"]

print("Loaded", len(chunks), "chunks and", len(embeddings), "embeddings")
print("Source document:", source)


# ---------------------------------------------------------
# 3. Create Azure AI Search vector index
# ---------------------------------------------------------

index_name = "policy-index"

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_key)
)

index = SearchIndex(
    name=index_name,
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="vector-profile"
        )
    ],
    vector_search=VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ]
    )
)

print("Creating index:", index_name)
index_client.create_index(index)
print("Index created successfully:", index_name)


# ---------------------------------------------------------
# 4. Upload chunks + embeddings
# ---------------------------------------------------------

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_key)
)

docs = []
for i, chunk in enumerate(chunks):
    docs.append({
        "id": f"chunk-{i}",
        "content": chunk,
        "source": source,
        "embedding": embeddings[i]
    })

print("Uploading", len(docs), "documents...")
search_client.upload_documents(docs)
print("Upload complete.")


# ---------------------------------------------------------
# 5. Run vector search
# ---------------------------------------------------------

query_embedding = embeddings[0]  # or embed a new query

print("\nRunning vector search...")

results = search_client.search(
    search_text=None,
    vector_queries=[{
        "kind": "vector",
        "vector": query_embedding,
        "fields": "embedding",
        "k": 3
    }]
)

print("\nTop matches:")
for r in results:
    print("--------------------------------------------------")
    print("Score:", r["@search.score"])
    print("Content:", r["content"])
    print("Source:", r["source"])