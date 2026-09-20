
import json
import os
import re

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel


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


if not ACCOUNT_ID:
    raise ValueError(
        "CLOUDFLARE_ACCOUNT_ID is missing from .env"
    )

if not API_TOKEN:
    raise ValueError(
        "CLOUDFLARE_API_TOKEN is missing from .env"
    )


# ============================================================
# CLOUDFLARE CONFIGURATION
# ============================================================

CLOUDFLARE_API_BASE = (
    "https://api.cloudflare.com/client/v4"
)

EMBEDDING_MODEL = (
    "@cf/baai/bge-base-en-v1.5"
)

CHAT_MODEL = (
    "@cf/meta/llama-3.1-8b-instruct"
)

TTS_MODEL = (
    "@cf/deepgram/aura-1"
)

TTS_SPEAKER = "orion"

REQUEST_TIMEOUT = 60


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Talal AI Portfolio Assistant",
    version="3.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str


class TTSRequest(BaseModel):
    text: str


# ============================================================
# CLOUDFLARE HEADERS
# ============================================================

def cloudflare_headers():
    """
    Return authentication headers for Cloudflare APIs.
    """

    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }


# ============================================================
# CREATE QUESTION EMBEDDING
# ============================================================

def create_embedding(text: str):
    """
    Convert a visitor question into a vector embedding.
    """

    url = (
        f"{CLOUDFLARE_API_BASE}"
        f"/accounts/{ACCOUNT_ID}"
        f"/ai/run/{EMBEDDING_MODEL}"
    )

    payload = {
        "text": [text],
    }

    response = requests.post(
        url,
        headers=cloudflare_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Embedding request failed.\n"
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

    embedding_data = result.get("data")

    if not embedding_data:
        raise RuntimeError(
            "Embedding data is missing from Cloudflare response."
        )

    if not isinstance(embedding_data, list):
        raise RuntimeError(
            "Cloudflare embedding data has an unexpected format."
        )

    return embedding_data[0]


# ============================================================
# SEARCH VECTORIZE
# ============================================================

def search_knowledge(
    question_embedding,
    top_k=5,
    metadata_filter=None,
):
    """
    Search the portfolio knowledge stored in Vectorize.

    metadata_filter can be used to restrict retrieval to a
    specific knowledge category.
    """

    url = (
        f"{CLOUDFLARE_API_BASE}"
        f"/accounts/{ACCOUNT_ID}"
        f"/vectorize/v2"
        f"/indexes/{INDEX_NAME}"
        f"/query"
    )

    payload = {
        "vector": question_embedding,
        "topK": top_k,
        "returnMetadata": "all",
    }

    if metadata_filter:
        payload["filter"] = metadata_filter

    response = requests.post(
        url,
        headers=cloudflare_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Vectorize search failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Vectorize returned invalid JSON."
        ) from error

    if not data.get("success"):
        raise RuntimeError(
            "Vectorize returned an error:\n"
            f"{data}"
        )

    return data


# ============================================================
# PROJECT OVERVIEW QUESTION DETECTION
# ============================================================

def is_project_overview_question(
    question: str,
) -> bool:
    """
    Detect questions asking for the overall project list,
    project count, or project overview.
    """

    normalized_question = (
        question.lower()
        .strip()
    )

    keywords = [
        "how many projects",
        "how many project",
        "number of projects",
        "number of project",
        "projects has talal built",
        "projects has talal",
        "projects did talal build",
        "projects did talal create",
        "all projects",
        "all project",
        "list of projects",
        "list of project",
        "list all projects",
        "list all project",
        "what projects",
        "which projects",
        "what are talal's projects",
        "what are talal projects",
        "tell me about talal's projects",
        "tell me about talal projects",
        "tell me your projects",
        "your projects",
    ]

    return any(
        keyword in normalized_question
        for keyword in keywords
    )


# ============================================================
# KNOWLEDGE EXTRACTION
# ============================================================

def extract_context(matches):
    """
    Extract textual knowledge from Vectorize matches.
    """

    context_parts = []

    for match in matches:

        metadata = match.get("metadata") or {}

        text = metadata.get("text")

        if not text:
            continue

        context_parts.append(
            text.strip()
        )

    return context_parts


# ============================================================
# DEBUG VECTOR SEARCH
# ============================================================

def print_search_debug(
    question: str,
    matches,
    search_type: str,
):
    """
    Print retrieval information for development/debugging.
    """

    print()
    print("=" * 60)
    print("VECTORIZE SEARCH")
    print("=" * 60)

    print("Question:", question)
    print("Search type:", search_type)
    print("Matches:", len(matches))

    for index, match in enumerate(
        matches,
        start=1,
    ):

        metadata = (
            match.get("metadata")
            or {}
        )

        print()
        print(f"--- Match {index} ---")

        print(
            "ID:",
            match.get("id"),
        )

        print(
            "Score:",
            match.get("score"),
        )

        print(
            "Category:",
            metadata.get("category"),
        )

        print(
            "Source:",
            metadata.get("source"),
        )

        print(
            "Metadata:",
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
        )

    print("=" * 60)
    print()


# ============================================================
# CLEAN GENERATED ANSWER
# ============================================================

def clean_answer(text: str) -> str:
    """
    Clean the final visitor-facing answer.

    The portfolio assistant should never return the slash
    character because the response may also be spoken aloud.
    """

    if not text:
        return ""

    cleaned = text.strip()

    # Replace slash-separated technology names with natural
    # spoken wording.
    cleaned = re.sub(
        r"\s*/\s*",
        " and ",
        cleaned,
    )

    # Remove accidental repeated whitespace.
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned.strip()


# ============================================================
# GENERATE AI ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str,
):
    """
    Generate a natural portfolio answer using only retrieved
    knowledge.
    """

    url = (
        f"{CLOUDFLARE_API_BASE}"
        f"/accounts/{ACCOUNT_ID}"
        f"/ai/run/{CHAT_MODEL}"
    )

    system_prompt = """
You are Talal AI, the personal AI assistant for Talal's portfolio.

Your job is to answer visitor questions about Talal using ONLY
the portfolio knowledge supplied by the application.

============================================================
CORE RULES
============================================================

1. Answer ONLY the visitor's current question.

2. Use ONLY the provided portfolio knowledge.

3. Never invent information about Talal.

4. Never assume information that is not present in the
   portfolio knowledge.

5. Do not make unsupported claims.

6. Do not invent:
   - jobs
   - companies
   - clients
   - certifications
   - degrees
   - technologies
   - achievements
   - project features
   - work experience

7. Do not repeat the visitor's question.

8. Do not create another question.

9. Do not provide information that is unrelated to the
   visitor's question.

10. Do not use exaggerated marketing language such as:
    flagship, cutting-edge, robust, comprehensive,
    world-class, powerful, exceptional, or outstanding
    unless the supplied portfolio knowledge explicitly
    supports that wording.

============================================================
PROJECT QUESTIONS
============================================================

11. If the visitor asks for the number of projects,
    answer using the project overview knowledge.

12. If the visitor asks for all projects or a project list,
    mention all projects available in the supplied project
    overview knowledge.

13. Do not mention only one project when the visitor asks
    for all projects.

14. If the visitor asks about a specific project, focus on
    that project instead of listing unrelated projects.

15. If the visitor asks a broad question about Talal's
    projects, give a concise overview of the projects rather
    than describing only one project.

============================================================
AMBIGUOUS QUESTIONS
============================================================

16. If a visitor says something like:
    "tell me about your projects"

    interpret it as a request for a general overview of
    Talal's projects.

17. If a visitor asks:
    "tell me about your project"

    and no specific project is named, give a general overview
    of Talal's projects instead of randomly selecting one.

============================================================
SLASH CHARACTER RULE
============================================================

18. NEVER use the "/" character in the answer.

19. Write technology combinations naturally.

Examples:

AI/ML
should become:
AI and machine learning

Python/FastAPI
should become:
Python and FastAPI

React/Django
should become:
React and Django

NLP/AI
should become:
NLP and AI

20. Never say the word "slash" when answering a visitor.

============================================================
UNKNOWN INFORMATION
============================================================

21. If the requested information is not present in the
    supplied portfolio knowledge, respond exactly:

"I don't have that information in Talal's portfolio knowledge."

============================================================
ANSWER STYLE
============================================================

Professional.
Friendly.
Natural.
Clear.
Direct.
Concise.

Simple question:
1 to 2 sentences.

Normal portfolio question:
2 to 4 sentences.

Project list question:
List all requested projects clearly.

Technical question:
Up to 5 to 6 sentences.

Answer the visitor directly.
"""

    user_prompt = f"""
PORTFOLIO KNOWLEDGE:

{context}

VISITOR QUESTION:

{question}

Answer the visitor's question using ONLY the portfolio
knowledge above.
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_tokens": 300,
        "temperature": 0.2,
    }

    response = requests.post(
        url,
        headers=cloudflare_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "AI generation failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Cloudflare AI returned invalid JSON."
        ) from error

    if not data.get("success"):
        raise RuntimeError(
            "Cloudflare AI returned an error:\n"
            f"{data}"
        )

    result = data.get("result")

    if not result:
        raise RuntimeError(
            "Cloudflare AI result is missing."
        )

    answer = result.get("response")

    if not answer:
        raise RuntimeError(
            "Cloudflare AI response is missing."
        )

    return clean_answer(answer)


# ============================================================
# TEXT TO SPEECH
# ============================================================

def generate_speech(text: str):
    """
    Convert the final cleaned answer into speech.

    The answer is cleaned again here as a safety measure so
    that the TTS model never receives slash-separated text.
    """

    speech_text = clean_answer(text)

    if not speech_text:
        raise RuntimeError(
            "Cannot generate speech from empty text."
        )

    url = (
        f"{CLOUDFLARE_API_BASE}"
        f"/accounts/{ACCOUNT_ID}"
        f"/ai/run/{TTS_MODEL}"
    )

    payload = {
        "text": speech_text,
        "speaker": TTS_SPEAKER,
        "encoding": "mp3",
    }

    response = requests.post(
        url,
        headers=cloudflare_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "TTS generation failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    return response.content


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic backend status endpoint.
    """

    return {
        "message": (
            "Talal AI RAG Backend is running"
        )
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    """
    Backend health check.
    """

    return {
        "status": "healthy",
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Main RAG chat endpoint.

    Flow:

    Visitor question
        ↓
    Question embedding
        ↓
    Query classification
        ↓
    Vectorize retrieval
        ↓
    Context construction
        ↓
    LLM answer
        ↓
    Answer cleaning
        ↓
    JSON response
    """

    question = request.message.strip()

    if not question:
        return {
            "answer": (
                "Please ask me something about Talal."
            )
        }

    try:

        # ----------------------------------------------------
        # 1. CREATE QUESTION EMBEDDING
        # ----------------------------------------------------

        question_embedding = create_embedding(
            question
        )

        # ----------------------------------------------------
        # 2. DETERMINE RETRIEVAL STRATEGY
        # ----------------------------------------------------

        if is_project_overview_question(
            question
        ):

            search_type = (
                "project overview"
            )

            search_results = search_knowledge(
                question_embedding,
                top_k=8,
                metadata_filter={
                    "category": "projects_overview"
                },
            )

        else:

            search_type = (
                "semantic portfolio search"
            )

            search_results = search_knowledge(
                question_embedding,
                top_k=5,
            )

        # ----------------------------------------------------
        # 3. EXTRACT MATCHES
        # ----------------------------------------------------

        matches = (
            search_results
            .get("result", {})
            .get("matches", [])
        )

        # ----------------------------------------------------
        # 4. DEBUG RETRIEVAL
        # ----------------------------------------------------

        print_search_debug(
            question=question,
            matches=matches,
            search_type=search_type,
        )

        # ----------------------------------------------------
        # 5. EXTRACT KNOWLEDGE
        # ----------------------------------------------------

        context_parts = extract_context(
            matches
        )

        # ----------------------------------------------------
        # 6. HANDLE MISSING KNOWLEDGE
        # ----------------------------------------------------

        if not context_parts:

            return {
                "answer": (
                    "I don't have that information "
                    "in Talal's portfolio knowledge."
                )
            }

        # ----------------------------------------------------
        # 7. BUILD CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # DEBUG CONTEXT
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("CONTEXT SENT TO LLM")
        print("=" * 60)
        print(context)
        print("=" * 60)
        print()

        # ----------------------------------------------------
        # 8. GENERATE ANSWER
        # ----------------------------------------------------

        answer = generate_answer(
            question=question,
            context=context,
        )

        # ----------------------------------------------------
        # 9. FINAL SAFETY CLEANING
        # ----------------------------------------------------

        answer = clean_answer(
            answer
        )

        # ----------------------------------------------------
        # 10. RETURN RESPONSE
        # ----------------------------------------------------

        return {
            "answer": answer
        }

    except Exception as error:

        print()
        print("=" * 60)
        print("BACKEND ERROR")
        print("=" * 60)
        print(str(error))
        print("=" * 60)
        print()

        return {
            "error": True,
            "details": str(error),
            "answer": (
                "I hit a backend communication issue "
                "while querying Talal's portfolio knowledge."
            ),
        }


# ============================================================
# TEXT TO SPEECH ENDPOINT
# ============================================================

@app.post("/api/tts")
def text_to_speech(
    request: TTSRequest,
):
    """
    Convert an assistant response into MP3 audio.
    """

    text = request.text.strip()

    if not text:
        return {
            "error": True,
            "message": "Text is required.",
        }

    try:

        audio = generate_speech(
            text
        )

        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
            },
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("TTS ERROR")
        print("=" * 60)
        print(str(error))
        print("=" * 60)
        print()

        return {
            "error": True,
            "details": str(error),
        }
