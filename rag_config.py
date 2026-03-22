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
ENABLE_QUERY_EXPANSION = True
QUERY_EXPANSION_PROMPT = """You are a search query optimizer for a portfolio website knowledge base about Cam Dresie, a product manager.
The knowledge base contains: resume, LinkedIn profile, career timeline, leadership philosophy, engineering projects, PM projects, blog posts, and a personal bio/summary.

Rewrite the user's question into a search query that will match the most relevant documents.
Include specific keywords the documents would contain. Be concrete — mention job titles, tools, company names, or topics that would appear in the matching content.
Output ONLY the rewritten query, nothing else.

User question: {question}

Rewritten query:"""

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
HISTORY_TURNS_FOR_QUERY = 2  # Number of recent conversation turns to include in retrieval query
