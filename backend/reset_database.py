import os
import requests
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
INDEX_NAME = os.getenv("VECTORIZE_INDEX_NAME", "talal-portfolio")

API_BASE = "https://api.cloudflare.com/client/v4"
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}
index_url = f"{API_BASE}/accounts/{ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}"

print(f"1. Deleting old index: {INDEX_NAME}...")
requests.delete(index_url, headers=headers)

print("2. Creating new empty index...")
requests.post(
    f"{API_BASE}/accounts/{ACCOUNT_ID}/vectorize/v2/indexes", 
    headers=headers, 
    json={"name": INDEX_NAME, "config": {"dimensions": 768, "metric": "cosine"}}
)

print("3. Configuring category metadata...")
requests.post(
    f"{index_url}/metadata_index/create", 
    headers=headers, 
    json={"propertyName": "category", "indexType": "string"}
)

print("\nDatabase reset complete! It is now completely empty.")