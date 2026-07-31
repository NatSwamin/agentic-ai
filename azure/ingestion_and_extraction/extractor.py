import os
import pdfplumber
from dotenv import load_dotenv
from gliner2 import GLiNER2
from embedder import get_embeddings_client, embed_chunks

# Initialize the GLiNER2 model
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts and concatenates text cleanly from all pages of a PDF."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num} ---\n" + page_text + "\n"
    return text

def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Splits text into safe chunks on sentence/paragraph boundaries to prevent token limits."""
    lines = text.split("\n")
    chunks = []
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 < max_chars:
            current_chunk += line + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def extract_entities(text: str):
    """Extracts entities using GLiNER2 and normalizes them into a flat list."""
    entity_types = [
        # Policy Summary[cite: 2]
        "person", "company", "address", "location", "date",
        "policy_number", "premium", "coverage_type",

        # Claim History[cite: 2]
        "claim_id", "claim_date", "loss_cause",
        "claim_amount", "settlement_amount",
        "claim_status", "loss_location"
    ]
    
    chunks = chunk_text(text)
    flat_entities = []
    seen = set()  # Prevents duplicate entries across chunks
    
    for chunk in chunks:
        result = model.extract_entities(chunk, entity_types)
        
        # Handle GLiNER2's dictionary or list output formats gracefully
        entities_dict = result.get("entities", result) if isinstance(result, dict) else {}
        
        if isinstance(entities_dict, dict):
            for label, values in entities_dict.items():
                if isinstance(values, list):
                    for val in values:
                        identifier = (val, label)
                        if identifier not in seen:
                            seen.add(identifier)
                            flat_entities.append({
                                "text": val,
                                "label": label
                            })
        elif isinstance(result, list):
            for ent in result:
                text_val = ent.get("text")
                label_val = ent.get("label")
                if text_val and label_val:
                    identifier = (text_val, label_val)
                    if identifier not in seen:
                        seen.add(identifier)
                        flat_entities.append({
                            "text": text_val,
                            "label": label_val
                        })
                        
    return flat_entities

def process_document(file_path: str, client):
    text = extract_text_from_pdf(file_path)
    entities = extract_entities(text)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, client)

    # Ensure the dictionary is returned properly
    return {
        "source": file_path,
        "text": text,
        "entities": entities,
        "chunks": chunks,
        "embeddings": embeddings
    }

if __name__ == "__main__":
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    client = get_embeddings_client(endpoint=endpoint, api_key=api_key)
    
    result = process_document("Policy_doc.pdf", client=client)
    
    print(f"Successfully processed: {result['source']}")
    print(f"Total Chunks: {len(result['chunks'])}, Total Embeddings: {len(result['embeddings'])}")
    print(f"Extracted Entities Count: {len(result['entities'])}")