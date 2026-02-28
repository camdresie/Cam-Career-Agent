# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the production chatbot app
python app.py

# Deploy to HuggingFace Spaces
uv tool install 'huggingface_hub[cli]'
hf auth login --token $HF_TOKEN
uv run gradio deploy
```

## Environment Variables

Required and optional keys go in `.env`:

```
OPENAI_API_KEY       # Required
PUSHOVER_USER        # Phone push notifications
PUSHOVER_TOKEN
HF_TOKEN             # HuggingFace deployment
```

## Architecture

**`app.py`** is a RAG-enhanced Gradio chatbot deployed to HuggingFace Spaces. It impersonates Cam Dresie using context from `me/linkedin.pdf`, `me/Cam_Dresie_Resume_2026_GPM.pdf`, `me/summary.txt`, and `portfolio_data/data.json`, plus inline bio/leadership/timeline content and blog posts fetched from an RSS feed.

It uses OpenAI embeddings + FAISS for semantic retrieval and implements the agentic loop pattern with two tools (`record_user_details`, `record_unknown_question`) and Pushover notifications.

**Core agentic loop pattern:**
```
LLM call → check finish_reason
  "tool_calls" → execute tool → append result → loop
  "stop"       → return response to user
```

**Key files:**
- `app.py` — Production chatbot
- `me/` — Personal documents (LinkedIn PDF, resume, summary)
- `portfolio_data/data.json` — Portfolio project data
- `requirements.txt` — Python dependencies
