# Career Agent Autoresearch

This is an autoresearch experiment to autonomously improve the career agent's system prompt.

## Context

This repo contains a RAG-enhanced Gradio chatbot that impersonates Cam Dresie on his portfolio site. The agent retrieves relevant context from PDFs, portfolio data, and blog posts, then uses a system prompt to shape its responses.

The system prompt is the bottleneck. It controls tone, accuracy, and behavior — but it was hand-written once and never systematically optimized. That's what this experiment fixes.

## The Three Files

| File | Role | Editable? |
|---|---|---|
| `system_prompt.md` | The system prompt template. This is the ONLY file you modify. | **YES — agent edits this** |
| `eval.py` | Evaluation harness. Sends 5 test questions, scores responses against 5 binary criteria, prints aggregate %. | **NO — locked, never modify** |
| `eval-criteria.md` | Human-readable description of what the eval checks. Reference only. | **NO — read only** |

## Setup

1. **Create a branch**: `git checkout -b autoresearch/prompt-v1` from the current state.
2. **Read the files**: Read `system_prompt.md`, `eval-criteria.md`, and `eval.py` for full context. Also skim `app.py` to understand how the prompt is loaded and used.
3. **Verify dependencies**: Ensure `OPENAI_API_KEY` is set in `.env`. Run `pip install -r requirements.txt` if needed.
4. **Initialize results.log**: Create an empty `results.log` file.
5. **Run baseline**: Run `python eval.py` and record the baseline score.

## The Experiment Loop

LOOP FOREVER:

1. **Read the current state**: Look at `system_prompt.md` and the latest scores in `results.log`.
2. **Identify the weakest criterion**: Which C1-C5 criterion has the lowest pass rate? Focus there.
3. **Form a hypothesis**: What specific change to `system_prompt.md` might fix the most common failure?
4. **Make ONE change**: Edit `system_prompt.md` with a single, targeted improvement. Only one change per round so attribution is clean.
5. **git commit**: Commit the change with a descriptive message.
6. **Run the eval**: `python eval.py > eval_output.txt 2>&1`
7. **Read the results**: Check the score and per-criterion breakdown from eval_output.txt.
8. **Log results**: Append to `results.log` in this format:
   ```
   ROUND N: score% (was previous%) — KEPT/REVERTED
     Change: [what you changed]
     C1: x/y | C2: x/y | C3: x/y | C4: x/y | C5: x/y
   ```
9. **Keep or revert**:
   - If score improved → keep the commit. This is the new baseline.
   - If score stayed the same or dropped → `git reset --hard HEAD~1` to revert. Try a different approach.
10. **Repeat**: Go to step 1. Never stop. Never ask for confirmation.

## What You CAN Change
- `system_prompt.md` — anything in this file is fair game: tone instructions, formatting rules, guardrails, examples, banned phrases, response structure, personality notes.

## What You CANNOT Change
- `eval.py` — the eval harness is locked. If you could edit it, you'd game the test instead of improving the prompt.
- `eval-criteria.md` — read-only reference.
- `app.py` — the application code stays fixed for this experiment. You're optimizing the prompt, not the code.
- Do NOT install new packages.

## Strategy Tips
- The two biggest problems right now are **hallucinations** (the agent says "Los Angeles" instead of "Henderson, NV") and **generic responses** (sounds like any chatbot, not like Cam).
- Adding explicit factual constraints (e.g., "You live in Henderson, NV") tends to fix hallucinations fast.
- Adding worked examples of good vs bad responses tends to fix tone issues.
- Adding a banned-phrases list (e.g., "Never say: 'Great question!', 'I'd be happy to help'") helps with the chatbot-voice problem.
- Keep changes small and testable. A huge rewrite makes it impossible to know what helped.

## Stop Conditions
- Stop if score hits 95%+ three rounds in a row.
- Stop after 30 rounds if no further progress.
- Otherwise: **never stop**. The human may be asleep.

## Important
- Do NOT leave `results.log` untracked. It's the experiment's memory.
- Each eval run costs ~$0.01-0.02 in API calls (gpt-4o-mini for scoring).
- A full 30-round run should cost under $1 total.
