# Talal AI — Portfolio Assistant (AI Portfolio Chatbot, RAG)

The assistant answering questions on this portfolio. A personal project, built and deployed live.

A Retrieval-Augmented Generation (RAG) chatbot that answers visitor questions about Talal's background, skills, and projects. Portfolio content is split into chunks and turned into embeddings. For each question, vector-based semantic search retrieves the most relevant content before the language model generates the answer, which keeps answers grounded in portfolio content and reduces unsupported answers. The answer can also be spoken aloud using text-to-speech.

## Tech stack
Python, FastAPI, embeddings, vector search (Cloudflare Vectorize), and Cloudflare AI (Workers AI).

## What he built
The FastAPI backend for chat requests, retrieval, and response generation, plus tuning of retrieval to stay grounded. It is deployed live on Cloudflare AI infrastructure.

Status: live. Don't state usage numbers, traffic, or accuracy figures — none are confirmed.
