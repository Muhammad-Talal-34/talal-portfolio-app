import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
INDEX_NAME = os.getenv(
    "VECTORIZE_INDEX_NAME",
    "talal-portfolio",
)


# ============================================================
# CONFIGURATION
# ============================================================

CLOUDFLARE_API_BASE = (
    "https://api.cloudflare.com/client/v4"
)

EMBEDDING_MODEL = (
    "@cf/baai/bge-base-en-v1.5"
)

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

REQUEST_TIMEOUT = 120


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

def validate_environment():
    """Validate required environment variables."""

    if not ACCOUNT_ID:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID is missing from .env"
        )

    if not API_TOKEN:
        raise RuntimeError(
            "CLOUDFLARE_API_TOKEN is missing from .env"
        )

    if not INDEX_NAME:
        raise RuntimeError(
            "VECTORIZE_INDEX_NAME is missing from .env"
        )


# ============================================================
# CLOUDFLARE HEADERS
# ============================================================

def cloudflare_headers():
    """Return standard Cloudflare API headers."""

    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Split knowledge text into overlapping chunks.

    Overlap helps preserve context between neighboring chunks.
    """

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# DETERMINE KNOWLEDGE CATEGORY
# ============================================================

def get_knowledge_category(filename: str) -> str:
    """
    Determine the category of a knowledge file.

    Categories are stored as Vectorize metadata and can later
    be used by the RAG retrieval layer.
    """

    normalized_name = filename.lower()

    if normalized_name == "projects_overview.md":
        return "projects_overview"

    if normalized_name.startswith("project_"):
        return "project"

    if normalized_name == "about.md":
        return "about"

    if normalized_name == "education.md":
        return "education"

    if normalized_name == "skills.md":
        return "skills"

    if "experience" in normalized_name:
        return "experience"

    if normalized_name == "career.md":
        return "career"

    if normalized_name == "contact.md":
        return "contact"

    return "general"


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):
    """
    Create embeddings using Cloudflare Workers AI.
    """

    if not texts:
        return []

    url = (
        f"{CLOUDFLARE_API_BASE}"
        f"/accounts/{ACCOUNT_ID}"
        f"/ai/run/{EMBEDDING_MODEL}"
    )

    payload = {
        "text": texts
    }

    response = requests.post(
        url,
        headers=cloudflare_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Cloudflare embedding request failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Cloudflare embedding API returned invalid JSON."
        ) from error

    if not data.get("success"):
        raise RuntimeError(
            "Cloudflare embedding API returned an error:\n"
            f"{data}"
        )

    result = data.get("result")

    if not result:
        raise RuntimeError(
            "Embedding result is missing from Cloudflare response."
        )

    embeddings = result.get("data")

    if not embeddings:
        raise RuntimeError(
            "Embedding data is missing from Cloudflare response."
        )

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Embedding count does not match text chunk count.\n"
            f"Texts: {len(texts)}\n"
            f"Embeddings: {len(embeddings)}"
        )

    return embeddings


# ============================================================
# UPLOAD VECTORS TO VECTORIZE
# ============================================================

def upload_vectors(vectors):
    """
    Upload vectors and metadata to Cloudflare Vectorize.
    Requires NDJSON format for Vectorize v2.
    """
    if not vectors:
        return None

    url = (
        f"{CLOUDFLARE_API_BASE}"
        f"/accounts/{ACCOUNT_ID}"
        f"/vectorize/v2"
        f"/indexes/{INDEX_NAME}"
        f"/upsert"
    )

    # Convert the list of vectors into a Newline-Delimited JSON string (NDJSON)
    ndjson_payload = "\n".join(json.dumps(vector) for vector in vectors)

    # Use the strict x-ndjson content type required by Vectorize v2
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/x-ndjson",
    }

    response = requests.post(
        url,
        headers=headers,
        data=ndjson_payload,  # Send as raw data, not json=
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            "Cloudflare Vectorize upload failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Cloudflare Vectorize returned invalid JSON."
        ) from error

    if not data.get("success"):
        raise RuntimeError(
            "Cloudflare Vectorize returned an error:\n"
            f"{data}"
        )

    return data


# ============================================================
# READ KNOWLEDGE FILES
# ============================================================

def load_knowledge():
    """
    Read all Markdown files from the knowledge directory
    and convert them into chunks with metadata.
    """

    if not KNOWLEDGE_DIR.exists():
        raise RuntimeError(
            f"Knowledge directory does not exist: "
            f"{KNOWLEDGE_DIR}"
        )

    if not KNOWLEDGE_DIR.is_dir():
        raise RuntimeError(
            f"Knowledge path is not a directory: "
            f"{KNOWLEDGE_DIR}"
        )

    all_texts = []
    all_metadata = []
    all_ids = []

    knowledge_files = sorted(
        KNOWLEDGE_DIR.glob("*.md")
    )

    if not knowledge_files:
        raise RuntimeError(
            f"No Markdown knowledge files found in "
            f"{KNOWLEDGE_DIR}"
        )

    print("Reading knowledge files...")
    print()

    for filepath in knowledge_files:

        filename = filepath.name

        try:
            text = filepath.read_text(
                encoding="utf-8"
            )
        except OSError as error:
            raise RuntimeError(
                f"Unable to read knowledge file: "
                f"{filepath}"
            ) from error

        if not text.strip():
            print(
                f"Skipping empty file: {filename}"
            )
            continue

        category = get_knowledge_category(
            filename
        )

        chunks = create_chunks(text)

        print(
            f"{filename:<35} "
            f"{len(chunks):>3} chunks "
            f"[{category}]"
        )

        for chunk_index, chunk in enumerate(chunks):

            vector_id = (
                f"{filepath.stem}-{chunk_index}"
            )

            metadata = {
                "source": filename,
                "category": category,
                "chunk": chunk_index,
                "text": chunk,
            }

            all_texts.append(chunk)
            all_metadata.append(metadata)
            all_ids.append(vector_id)

    return (
        all_texts,
        all_metadata,
        all_ids,
    )


# ============================================================
# BUILD VECTOR PAYLOAD
# ============================================================

def build_vectors(
    embeddings,
    metadata,
    vector_ids,
):
    """
    Combine embeddings, IDs, and metadata into the
    Vectorize upsert format.
    """

    if not (
        len(embeddings)
        == len(metadata)
        == len(vector_ids)
    ):
        raise RuntimeError(
            "Vector data lengths do not match."
        )

    vectors = []

    for index, embedding in enumerate(embeddings):

        vectors.append(
            {
                "id": vector_ids[index],
                "values": embedding,
                "metadata": metadata[index],
            }
        )

    return vectors


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def main():
    """
    Run the complete RAG ingestion pipeline.

    Flow:

    Markdown files
        ↓
    Text chunks
        ↓
    Metadata
        ↓
    Cloudflare embeddings
        ↓
    Vectorize
    """

    print("=" * 60)
    print("TALAL AI RAG KNOWLEDGE INGESTION")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # 1. Validate configuration
    # --------------------------------------------------------

    validate_environment()

    print(
        f"Vectorize index : {INDEX_NAME}"
    )

    print(
        f"Embedding model : {EMBEDDING_MODEL}"
    )

    print(
        f"Knowledge dir   : {KNOWLEDGE_DIR}"
    )

    print()

    # --------------------------------------------------------
    # 2. Read knowledge files
    # --------------------------------------------------------

    (
        all_texts,
        all_metadata,
        all_ids,
    ) = load_knowledge()

    if not all_texts:
        raise RuntimeError(
            "No usable knowledge chunks were created."
        )

    print()
    print(
        f"Total knowledge chunks: {len(all_texts)}"
    )

    # --------------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------------

    print()
    print(
        "Creating embeddings with Cloudflare Workers AI..."
    )

    embeddings = create_embeddings(
        all_texts
    )

    print(
        f"Created {len(embeddings)} embeddings."
    )

    # --------------------------------------------------------
    # 4. Build Vectorize records
    # --------------------------------------------------------

    vectors = build_vectors(
        embeddings=embeddings,
        metadata=all_metadata,
        vector_ids=all_ids,
    )

    print(
        f"Prepared {len(vectors)} vectors for upload."
    )

    # --------------------------------------------------------
    # 5. Upload to Vectorize
    # --------------------------------------------------------

    print()
    print(
        "Uploading vectors to Cloudflare Vectorize..."
    )

    result = upload_vectors(
        vectors
    )

    # --------------------------------------------------------
    # 6. Complete
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RAG KNOWLEDGE BASE UPDATED SUCCESSFULLY")
    print("=" * 60)
    print()

    print(
        f"Files processed : "
        f"{len(list(KNOWLEDGE_DIR.glob('*.md')))}"
    )

    print(
        f"Chunks indexed  : "
        f"{len(vectors)}"
    )

    print(
        f"Vector index    : "
        f"{INDEX_NAME}"
    )

    print()

    if result:
        print("Cloudflare response:")
        print(result)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()