# RAG Retrieval Autoresearch

Autonomous optimization of the RAG retrieval pipeline for the career agent.

## Context

The career agent uses OpenAI embeddings + FAISS to retrieve context from PDFs, blog posts, portfolio data, and inline content. The retrieval often misses relevant content — for example, asking "have you written about your career history?" fails to surface the blog post about transitioning to product management.

This experiment optimizes retrieval quality by tuning chunking, query expansion, formatting, and assembly parameters.

## The Three Files

| File | Role | Editable? |
|---|---|---|
| `rag_config.py` | All tunable RAG parameters. This is the ONLY file you modify. | **YES — agent edits this** |
| `rag_eval.py` | Retrieval eval harness. Tests 8 queries, scores against 4 binary criteria per query. | **NO — locked, never modify** |
| `app.py` | The application. Imports values from rag_config.py. | **NO — do not modify** |

## Setup

```bash
git checkout -b autoresearch/rag-v1
touch rag_results.log
python rag_eval.py --verbose    # Run baseline, see what's being retrieved
git add . && git commit -m "RAG autoresearch baseline"
```

## Eval Criteria (what rag_eval.py checks)

- **C1 (Source Hit)**: Does the retrieval include chunks from the expected source type? e.g., blog posts for newsletter questions, leadership content for leadership questions.
- **C2 (Content Hit)**: Do retrieved chunks contain expected keywords? e.g., "Ontra" when asked about Ontra experience.
- **C3 (Diversity)**: Is there variety in retrieved sources? No single source type should dominate >60% of results.
- **C4 (Relevance Floor)**: Is the lowest-scored chunk still meaningfully relevant? (score > 0.15)

## The Experiment Loop

IMPORTANT: After each change to rag_config.py, you MUST delete the `kb_cache/` directory before running the eval. The knowledge base is cached by content hash, but config changes (chunk size, overlap, prefixes) change HOW content is chunked, not the content itself — so the cache won't invalidate automatically.

```bash
rm -rf kb_cache/
```

Now loop:

1. Read `rag_config.py` and `rag_results.log` to understand current state.
2. Run `python rag_eval.py --verbose` to see exactly which chunks are being retrieved for each query.
3. Identify the weakest criterion or test case.
4. Make ONE change to `rag_config.py`. Examples of changes:
   - Adjust CHUNK_SIZE (smaller chunks = more precise, larger = more context)
   - Adjust CHUNK_OVERLAP (more overlap = less boundary information loss)
   - Enable ENABLE_QUERY_EXPANSION (rewrites user queries for better embedding matches)
   - Tune the QUERY_EXPANSION_PROMPT wording
   - Adjust TOP_K (more results = better recall but noisier)
   - Set MIN_SCORE_THRESHOLD to filter out low-relevance chunks
   - Enable REPEAT_TITLE_IN_CHUNKS so blog post titles appear in every chunk from that post
   - Change document prefixes to be more descriptive
   - Enable DEDUPE_BY_SOURCE to force source variety
   - Set MAX_UNIQUE_SOURCES to cap how many sources appear
   - Adjust HISTORY_TURNS_FOR_QUERY
5. Delete cache: `rm -rf kb_cache/`
6. `git commit -am "Round N: [what you changed]"`
7. Run `python rag_eval.py --verbose > rag_eval_output.txt 2>&1`
8. Read results from rag_eval_output.txt.
9. Log to `rag_results.log`:
   ```
   ROUND N: score% (was previous%) — KEPT/REVERTED
     Change: [what you changed]
     C1: x/y | C2: x/y | C3: x/y | C4: x/y
   ```
10. If score improved → keep. If not → `git reset --hard HEAD~1` and try something different.
11. Repeat.

## Strategy Tips

The biggest known problem is blog post retrieval. "Have you written about your career history?" should surface the "My Journey to Product Management" post but currently doesn't. Likely causes:

1. **No query expansion**: The raw question "have you written about your career history" doesn't embed close to blog post text about transitioning from law to product management. Enabling ENABLE_QUERY_EXPANSION and writing a good expansion prompt is probably the highest-leverage change.

2. **Blog chunks lack title context**: Once a blog post is chunked, later chunks lose the title. Enabling REPEAT_TITLE_IN_CHUNKS would keep the title visible in every chunk.

3. **Chunk size may be too large**: At 400 words, blog post chunks might be so big that the embedding averages out the signal. Trying 200-250 word chunks could help precision.

4. **top_k may be too low**: With 8 results and lots of resume/LinkedIn content competing, blog posts get crowded out. Trying 12-15 might help.

## Stop Conditions

- Stop at 95%+ for 3 consecutive rounds
- Stop after 30 rounds
- Each round takes ~5-10 seconds (embedding calls only, no LLM chat) so this is very fast

## Critical Reminders

- ALWAYS `rm -rf kb_cache/` after changing rag_config.py
- Never modify rag_eval.py
- Never modify app.py
- One change per round
