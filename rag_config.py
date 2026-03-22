"""
RAG Configuration — the ONLY file modified during autoresearch.
All tunable retrieval parameters live here.
"""

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 400          # Max words per chunk
CHUNK_OVERLAP = 80        # Overlap words between consecutive chunks
REPEAT_TITLE_IN_CHUNKS = False  # Prepend document title to every chunk

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 8                 # Number of chunks to retrieve
MIN_SCORE_THRESHOLD = 0.0  # Minimum cosine similarity to include a chunk
DEDUPE_BY_SOURCE = False   # Limit chunks per source for diversity
MAX_PER_SOURCE = 3         # Max chunks from one source (if DEDUPE_BY_SOURCE)

# ---------------------------------------------------------------------------
# Query Expansion
# ---------------------------------------------------------------------------
ENABLE_QUERY_EXPANSION = False
QUERY_EXPANSION_PROMPT = """Rewrite the following user question to improve semantic search retrieval.
Make it more specific and include likely keywords that relevant documents would contain.
Keep it as a single short paragraph. Do not answer the question.

User question: {question}

Rewritten query:"""

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
HISTORY_TURNS_FOR_QUERY = 2  # Number of recent conversation turns to include in retrieval query
