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

# Local LLM via Ollama
ollama serve
ollama pull llama3.2
```

Notebooks are run interactively in Cursor/VSCode using the `.venv` kernel (Python 3.12).

## Environment Variables

Required and optional keys go in `.env`:

```
OPENAI_API_KEY       # Required
ANTHROPIC_API_KEY    # Claude models
GOOGLE_API_KEY       # Gemini models
DEEPSEEK_API_KEY
GROQ_API_KEY
PUSHOVER_USER        # Phone push notifications
PUSHOVER_TOKEN
HF_TOKEN             # HuggingFace deployment
```

## Architecture

This is an educational course on agentic AI, structured as 5 progressive Jupyter notebooks plus a production app.

**`app.py`** is the production artifact — a Gradio chatbot deployed to HuggingFace Spaces that impersonates a person using context from `me/linkedin.pdf` and `me/summary.txt`. It implements the full agentic loop pattern with two tools (`record_user_details`, `record_unknown_question`) and Pushover notifications.

**Notebook progression:**
- `1_lab1.ipynb` — Basic OpenAI API setup
- `2_lab2.ipynb` — Multi-model comparison across OpenAI, Anthropic, Gemini, DeepSeek, Groq, Ollama
- `3_lab3.ipynb` — PDF context ingestion, Gradio chat UI, evaluation/retry loop
- `4_lab4.ipynb` — Tool definition (JSON schema), agentic loop, HuggingFace deployment
- `5_extra.ipynb` — Agent loop fundamentals with a todo management example

**Core agentic loop pattern** (used in `app.py`, Lab 4, Lab 5):
```
LLM call → check finish_reason
  "tool_calls" → execute tool → append result → loop
  "stop"       → return response to user
```

**Evaluation/retry pattern** (Lab 3):
```
Generate response → evaluate quality → if fails: add feedback and regenerate
```

`community_contributions/` contains 150+ student implementations showing alternative providers and approaches — treat as reference, not production code.
