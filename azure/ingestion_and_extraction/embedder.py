# embedder.py
from openai import AzureOpenAI

def get_embeddings_client(endpoint: str, api_key: str, api_version="2024-02-01"):
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key
    )

def embed_chunks(chunks, client, deployment_name="text-embedding-3-large"):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model=deployment_name,
            input=[chunk]
        )
        embeddings.append(response.data[0].embedding)
    return embeddings