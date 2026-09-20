import os
import requests

from dotenv import load_dotenv


load_dotenv()

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
INDEX_NAME = os.getenv("VECTORIZE_INDEX_NAME", "talal-portfolio")

API_BASE_URL = "https://api.cloudflare.com/client/v4"


def validate_environment():
    """Validate required environment variables."""
    if not ACCOUNT_ID:
        raise ValueError("CLOUDFLARE_ACCOUNT_ID is missing from .env")

    if not API_TOKEN:
        raise ValueError("CLOUDFLARE_API_TOKEN is missing from .env")

    if not INDEX_NAME:
        raise ValueError("VECTORIZE_INDEX_NAME is missing from .env")


def get_headers():
    """Return Cloudflare API headers."""
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }


def create_vectorize_index():
    """Create the Vectorize index."""
    url = (
        f"{API_BASE_URL}/accounts/"
        f"{ACCOUNT_ID}/vectorize/v2/indexes"
    )

    payload = {
        "name": INDEX_NAME,
        "description": "Talal personal portfolio RAG knowledge base",
        "config": {
            "dimensions": 768,
            "metric": "cosine",
        },
    }

    response = requests.post(
        url,
        headers=get_headers(),
        json=payload,
        timeout=60,
    )

    print("Create Index Status:", response.status_code)

    try:
        result = response.json()
        print(result)
    except ValueError:
        print(response.text)


def create_category_metadata_index():
    """Create a metadata index for the category field."""
    url = (
        f"{API_BASE_URL}/accounts/"
        f"{ACCOUNT_ID}/vectorize/v2/indexes/"
        f"{INDEX_NAME}/metadata_index/create"
    )

    payload = {
        "propertyName": "category",
        "indexType": "string",
    }

    response = requests.post(
        url,
        headers=get_headers(),
        json=payload,
        timeout=60,
    )

    print("\nCategory Metadata Index Status:", response.status_code)

    try:
        result = response.json()
        print(result)
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    validate_environment()

    print(f"Vectorize Index: {INDEX_NAME}")

    create_vectorize_index()
    create_category_metadata_index()