# test_pipeline.py
# Purpose: Run extractor + embedder pipeline and SAVE output for search ingestion

import os
import json
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from embedder import get_embeddings_client
from extractor import process_document

# -----------------------------
# 1. Connect to Azure Key Vault
# -----------------------------
KV_URL = "https://azure-openai-kv.vault.azure.net/"

credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=KV_URL, credential=credential)

# -----------------------------------------
# 2. Retrieve Azure OpenAI endpoint + key
# -----------------------------------------
endpoint = secret_client.get_secret("AZ-OPENAI-ENDPOINT").value
api_key = secret_client.get_secret("AZ-OPENAI-KEY").value

# -----------------------------------------
# 3. Create embeddings client
# -----------------------------------------
client = get_embeddings_client(endpoint, api_key)

# -----------------------------------------
# 4. Run your extraction + embedding pipeline
# -----------------------------------------
result = process_document("../data/Policy_doc.pdf", client)

print("Source:", result["source"])
print("Total Chunks:", len(result["chunks"]))
print("Total Embeddings:", len(result["embeddings"]))
print("Entities Extracted:", len(result["entities"]))

print("\nFirst embedding vector length:", len(result["embeddings"][0]))
print("First 5 values:", result["embeddings"][0][:5])

# -----------------------------------------
# 5. SAVE RESULT TO JSON FILE
# -----------------------------------------
with open("ingestion_output.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nSaved ingestion_output.json successfully.")
