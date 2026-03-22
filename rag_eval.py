"""
RAG Retrieval Evaluation Harness — DO NOT MODIFY.

Tests whether the retrieval pipeline surfaces the right content for known queries.
Unlike the agent eval (eval.py), this tests retrieval ONLY — no LLM response generation.
This makes it fast (~5 seconds per run) and cheap (just embedding calls, no chat completions).

Each test case has:
- A user query
- A list of source substrings that MUST appear in the retrieved chunks
- A list of source substrings that SHOULD NOT dominate the results (optional)

Scoring is binary per criterion per test case.

Usage:
    python rag_eval.py              # Run full eval
    python rag_eval.py --verbose    # Show retrieved chunks for each query
"""

import os
import sys
import argparse

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(__file__) or ".")

from app import CareerAgent

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
# Each test defines a query and what sources MUST be retrieved.
# "required_sources" = substrings that must appear in at least one retrieved chunk's source field.
# "required_content" = substrings that must appear in at least one retrieved chunk's text field.
# "unwanted_dominance" = sources that should NOT make up more than 50% of results (optional).

TEST_CASES = [
    {
        "id": 1,
        "query": "Have you ever written about your career history or how you got into product management?",
        "description": "Should retrieve the 'My Journey to Product Management' blog post",
        "required_sources": ["Blog"],
        "required_content": ["product management"],
        "unwanted_dominance": [],
    },
    {
        "id": 2,
        "query": "What's your leadership style?",
        "description": "Should retrieve leadership philosophy content",
        "required_sources": ["Leadership"],
        "required_content": ["trust", "autonomy"],
        "unwanted_dominance": [],
    },
    {
        "id": 3,
        "query": "Tell me about the AI projects you've built",
        "description": "Should retrieve engineering projects AND AI-related PM work",
        "required_sources": ["Project"],
        "required_content": [],
        "unwanted_dominance": [],
    },
    {
        "id": 4,
        "query": "Where did you go to school?",
        "description": "Should retrieve education info from timeline or resume",
        "required_sources": [],
        "required_content": ["UPenn", "Penn"],
        "unwanted_dominance": [],
    },
    {
        "id": 5,
        "query": "What do you write about in your newsletter?",
        "description": "Should retrieve blog posts, not just the bio mention of Beyond the Backlog",
        "required_sources": ["Blog"],
        "required_content": [],
        "unwanted_dominance": [],
    },
    {
        "id": 6,
        "query": "How did you transition from law to tech?",
        "description": "Should retrieve career timeline AND relevant blog content about the transition",
        "required_sources": ["Timeline", "Career"],
        "required_content": ["law", "legal"],
        "unwanted_dominance": [],
    },
    {
        "id": 7,
        "query": "What's your experience at Ontra?",
        "description": "Should retrieve Ontra-related content from multiple sources (resume, timeline, projects)",
        "required_sources": [],
        "required_content": ["Ontra"],
        "unwanted_dominance": [],
    },
    {
        "id": 8,
        "query": "Do you have any blog posts about AI or building with LLMs?",
        "description": "Should retrieve AI-related blog posts specifically",
        "required_sources": ["Blog"],
        "required_content": [],
        "unwanted_dominance": [],
    },
]

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def score_test_case(test_case: dict, retrieved: list, verbose: bool = False) -> dict:
    """
    Score a single test case. Returns dict with per-criterion pass/fail.

    Criteria:
    C1 (Source Hit): At least one required source substring appears in retrieved sources
    C2 (Content Hit): At least one required content substring appears in retrieved text
    C3 (Source Diversity): No single source type dominates >60% of retrieved chunks
    C4 (Relevance Floor): The lowest-scoring retrieved chunk has score > 0.15
    """
    sources = [r["source"] for r in retrieved]
    texts = [r["text"].lower() for r in retrieved]
    scores_list = [r["score"] for r in retrieved]

    results = {}

    # C1: Required sources present
    if test_case["required_sources"]:
        found_any = False
        for req in test_case["required_sources"]:
            if any(req.lower() in s.lower() for s in sources):
                found_any = True
                break
        results["C1_source_hit"] = found_any
    else:
        results["C1_source_hit"] = True  # No requirement = auto-pass

    # C2: Required content present
    if test_case["required_content"]:
        found_any = False
        for req in test_case["required_content"]:
            if any(req.lower() in t for t in texts):
                found_any = True
                break
        results["C2_content_hit"] = found_any
    else:
        results["C2_content_hit"] = True  # No requirement = auto-pass

    # C3: Source diversity — no single source prefix should be >60% of results
    if len(retrieved) > 0:
        # Group by source prefix (first word of source)
        source_prefixes = {}
        for s in sources:
            prefix = s.split(":")[0].split(" ")[0] if ":" in s else s.split(" ")[0]
            source_prefixes[prefix] = source_prefixes.get(prefix, 0) + 1
        max_concentration = max(source_prefixes.values()) / len(retrieved)
        results["C3_diversity"] = max_concentration <= 0.60
    else:
        results["C3_diversity"] = False

    # C4: Relevance floor — worst chunk shouldn't be completely irrelevant
    if scores_list:
        min_score = min(scores_list)
        results["C4_relevance_floor"] = min_score > 0.15
    else:
        results["C4_relevance_floor"] = False

    if verbose:
        print(f"\n  Test {test_case['id']}: {test_case['query'][:70]}...")
        print(f"  Expected: {test_case['description']}")
        for i, r in enumerate(retrieved):
            marker = ""
            if test_case["required_sources"]:
                if any(req.lower() in r["source"].lower() for req in test_case["required_sources"]):
                    marker = " <<<< HIT"
            print(f"    [{i+1}] {r['source']} (score: {r['score']:.3f}){marker}")
            print(f"        {r['text'][:120]}...")
        for crit, passed in results.items():
            print(f"  {crit}: {'PASS' if passed else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(verbose=False):
    """Run the full retrieval eval. Returns aggregate score as percentage."""
    print("=" * 60)
    print("RAG RETRIEVAL EVAL")
    print("=" * 60)

    # Initialize the agent (loads documents, builds index)
    print("Loading agent and building knowledge base...")
    agent = CareerAgent()

    total_checks = 0
    total_passed = 0
    all_results = {}

    for test in TEST_CASES:
        # Query the knowledge base directly (skip LLM, just test retrieval)
        query = test["query"]

        # Apply query expansion if enabled (import config)
        try:
            from rag_config import ENABLE_QUERY_EXPANSION, QUERY_EXPANSION_PROMPT, TOP_K
            if ENABLE_QUERY_EXPANSION:
                from openai import OpenAI
                client = OpenAI()
                expansion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(question=query)}],
                    max_tokens=100,
                    temperature=0,
                )
                expanded = expansion.choices[0].message.content.strip()
                if verbose:
                    print(f"\n  Expanded query: {expanded}")
                query = expanded
            top_k = TOP_K
        except ImportError:
            top_k = 8

        retrieved = agent.kb.query(query, top_k=top_k)

        scores = score_test_case(test, retrieved, verbose=verbose)

        if not verbose:
            print(f"\nTest {test['id']}: {test['query'][:60]}...")

        for crit, passed in scores.items():
            total_checks += 1
            if passed:
                total_passed += 1
            if not verbose:
                print(f"  {crit}: {'PASS' if passed else 'FAIL'}")

            # Track per-criterion totals
            if crit not in all_results:
                all_results[crit] = {"passed": 0, "total": 0}
            all_results[crit]["total"] += 1
            if passed:
                all_results[crit]["passed"] += 1

    # Aggregate
    pct = (total_passed / total_checks * 100) if total_checks > 0 else 0

    print("\n" + "=" * 60)
    print(f"SCORE: {total_passed}/{total_checks} = {pct:.1f}%")
    print("=" * 60)

    print("\nPer-criterion breakdown:")
    for crit in sorted(all_results.keys()):
        ct = all_results[crit]
        rate = ct["passed"] / ct["total"] * 100
        print(f"  {crit}: {ct['passed']}/{ct['total']} ({rate:.0f}%)")

    return pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks per query")
    args = parser.parse_args()

    score = run_eval(verbose=args.verbose)
    print(f"\nFinal score: {score:.1f}%")
