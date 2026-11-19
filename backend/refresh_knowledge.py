import time
import requests

import create_vectorize
import ingest


def delete_existing_index():
    """Delete the old Vectorize index to remove all stale embeddings."""
    url = (
        f"{create_vectorize.API_BASE_URL}/accounts/"
        f"{create_vectorize.ACCOUNT_ID}/vectorize/v2/indexes/"
        f"{create_vectorize.INDEX_NAME}"
    )

    print(f"Deleting old index '{create_vectorize.INDEX_NAME}'...")
    response = requests.delete(
        url,
        headers=create_vectorize.get_headers(),
        timeout=60,
    )

    if response.status_code == 200:
        print("Old index deleted successfully.")
    elif response.status_code == 404:
        print("Index did not exist. Proceeding to create a new one.")
    else:
        raise RuntimeError(
            f"Failed to delete index.\nStatus: {response.status_code}\nResponse: {response.text}"
        )


def main():
    create_vectorize.validate_environment()

    # Step 1: Purana index aur uski saari embeddings delete karo
    delete_existing_index()
    print("Waiting 5 seconds for Cloudflare to process deletion...")
    time.sleep(5)

    # Step 2: Naya index aur 'category' metadata index banao
    create_vectorize.create_vectorize_index()
    create_vectorize.create_category_metadata_index()
    print("Waiting 3 seconds before uploading new vectors...")
    time.sleep(3)

    # Step 3: Saari updated .md files ki nayi embeddings upload karo
    ingest.main()


if __name__ == "__main__":
    main()