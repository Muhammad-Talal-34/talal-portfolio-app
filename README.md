# Talal Naeem — AI & ML Portfolio

A personal portfolio website showcasing my projects, technical skills, and experience in AI, machine learning, Python development, backend development, and generative AI.

The portfolio also includes **Talal AI**, a RAG-powered AI assistant designed to answer questions about my background, skills, projects, and experience.

## 🚀 Live Portfolio

**AI Assistant:** Available through the portfolio

---

## ✨ Features

- Modern responsive portfolio website
- AI and machine learning project showcase
- Interactive Talal AI assistant
- Retrieval-Augmented Generation (RAG)
- AI-powered question answering
- Text-to-speech responses
- Voice-enabled interaction
- Technical skills and project information
- Responsive design for desktop and mobile devices
- FastAPI backend
- Cloudflare Workers AI integration
- Cloudflare Vectorize knowledge retrieval

---

## 🤖 Talal AI

Talal AI is an AI-powered portfolio assistant built to help visitors learn more about me.

It uses my portfolio knowledge base to answer questions about:

- About me
- Education
- Technical skills
- Projects
- Experience
- Career background
- Contact information

The assistant uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information before generating a response.

## Run locally

1. Install the Python dependencies from the repository root:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Create a `.env` file in the repository root with your Cloudflare credentials:

   ```text
   CLOUDFLARE_ACCOUNT_ID=your-account-id
   CLOUDFLARE_API_TOKEN=your-api-token
   VECTORIZE_INDEX_NAME=talal-portfolio
   ```

3. Start the backend from the repository root:

   ```powershell
   python -m uvicorn backend.main:app --reload --port 8000
   ```

4. Open `http://127.0.0.1:8000`. FastAPI serves `index.html` and the chat sends requests to the same server at `/api/chat`. Check backend health at `http://127.0.0.1:8000/api/health`.

## Update the AI knowledge base

Add or edit a Markdown file in `backend/knowledge`, then run this command from the repository root to re-ingest every knowledge file into Vectorize:

```powershell
python -m backend.ingest
```

The ingestion script locates the knowledge folder relative to its own file, so the command works regardless of the current working directory.

## Connect a deployed frontend

When hosting the frontend separately from the backend, set the `backend-api-url` meta tag in [index.html](./index.html) to the backend's complete chat endpoint, including `/api/chat`:

```html
<meta name="backend-api-url" content="https://your-backend.example.com/api/chat">
```

The backend must allow requests from the published frontend origin. Keep Cloudflare credentials in the backend environment; do not put them in the HTML.
