"""
Evaluation harness for the Career Agent — autoresearch edition.
DO NOT MODIFY THIS FILE. This is the locked eval (like prepare.py in Karpathy's repo).

Sends test questions to the agent, scores each response against binary criteria,
and prints a single aggregate score. The agent loop uses this to decide keep/revert.

Usage:
    python eval.py                # Run full eval, print score
    python eval.py --verbose      # Also print per-question breakdowns
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Test inputs — these simulate real visitors to the portfolio site
# ---------------------------------------------------------------------------

TEST_INPUTS = [
    {
        "id": 1,
        "question": "Where do you live and what do you do outside of work?",
        "applicable_criteria": ["C1", "C2", "C3"],
        "notes": "Must NOT say Los Angeles. Should mention Henderson NV, hiking, running, reading, Koda, husband, concerts.",
    },
    {
        "id": 2,
        "question": "Tell me about your experience with AI — what have you actually built?",
        "applicable_criteria": ["C1", "C2", "C3"],
        "notes": "Should cite specific tools (LangGraph, LangChain, Claude Agents, RAG, FAISS, etc.) and specific projects. Must not invent projects.",
    },
    {
        "id": 3,
        "question": "What's your management style? Can you give me a specific example?",
        "applicable_criteria": ["C1", "C2", "C3"],
        "notes": "Should reference leadership principles (high-trust, outcome-focused, psychological safety, etc.) with specifics, not just generic management platitudes.",
    },
    {
        "id": 4,
        "question": "I'm hiring a GPM for a legal tech company. Are you interested?",
        "applicable_criteria": ["C1", "C3", "C5"],
        "notes": "Should steer toward sharing email or LinkedIn. Should express genuine openness. Must not hallucinate interest in a specific company not in context.",
    },
    {
        "id": 5,
        "question": "What's your opinion on the latest iPhone release?",
        "applicable_criteria": ["C1", "C3", "C4"],
        "notes": "Out of scope. Should stay in character, not fabricate an opinion, and redirect naturally.",
    },
]

# ---------------------------------------------------------------------------
# Criteria definitions for the scoring LLM
# ---------------------------------------------------------------------------

CRITERIA = {
    "C1": {
        "name": "No hallucinated facts",
        "prompt": "Does the response ONLY contain claims verifiable from the provided bio context? If it invents a city (e.g. says 'Los Angeles' instead of 'Henderson, NV'), fabricates a project, makes up a metric, or attributes an experience the person never had, answer FAIL. If all facts are grounded in the context or are reasonable inferences, answer PASS.",
    },
    "C2": {
        "name": "Cites specific detail",
        "prompt": "Does the response reference at least one specific project name, company, tool, metric, or concrete experience — not just vague generalities? 'I have experience with AI' is FAIL. 'I built agents using LangGraph and Claude Agents at Ontra' is PASS.",
    },
    "C3": {
        "name": "Sounds like a real person",
        "prompt": "Does the response avoid generic AI-assistant phrasing like 'I'd be happy to help', 'As an AI', 'Great question!', or robotic bullet-point lists of qualifications? It should read like a confident professional talking naturally in first person. FAIL if it sounds like a customer service bot.",
    },
    "C4": {
        "name": "Handles out-of-scope gracefully",
        "prompt": "When asked something outside the person's professional context, does the agent stay in character, acknowledge the boundary naturally, and redirect — WITHOUT making up an answer? FAIL if it fabricates an opinion or breaks character. Only score this on out-of-scope questions.",
    },
    "C5": {
        "name": "Steers toward contact",
        "prompt": "When the conversation has a hiring or opportunity signal, does the response encourage the user to share their email or connect on LinkedIn? FAIL if it completely misses the opportunity to facilitate a connection. Only score this on hiring-signal questions.",
    },
}

# ---------------------------------------------------------------------------
# Known facts for grounding the scorer
# ---------------------------------------------------------------------------

GROUND_TRUTH_CONTEXT = """
Known facts about Cam Dresie (use these to check for hallucinations):
- Lives in Henderson, NV (NOT Los Angeles, NOT San Francisco, NOT New York)
- Group Product Manager at Ontra (flagship product: Contract Automation)
- Previously: Staff PM, Senior PM, PM, Associate PM at Ontra; TPM/SWE at Stack Moxie; GTA at UPenn
- Education: MCIT from UPenn, JD from WashU Law, BA from Truman State
- Builds with: LangGraph, LangChain, Claude Agents, Claude Code, OpenAI Agents SDK, RAG, FAISS
- Hobbies: hiking, running, reading (loves Walter Isaacson's Leonardo Da Vinci bio), concerts, husband, Labradoodle named Koda
- Leadership style: high-trust, outcome-focused, psychological safety, empowered autonomy
- LinkedIn: https://www.linkedin.com/in/camdresie/
- Writes the newsletter "Beyond the Backlog" about product management
"""


def get_agent_response(question: str) -> str:
    """Get a response from the career agent for a test question."""
    # Import the agent here so eval.py stays decoupled
    sys.path.insert(0, os.path.dirname(__file__))
    from app import CareerAgent

    agent = CareerAgent()
    history = []  # Fresh conversation each time
    return agent.chat(question, history)


def score_response(question: str, response: str, criterion_id: str, notes: str) -> dict:
    """Use an LLM to score a single response against a single criterion. Returns PASS/FAIL."""
    client = OpenAI()
    criterion = CRITERIA[criterion_id]

    scoring_prompt = f"""You are an eval scorer. You will be given:
1. A question that was asked to a portfolio chatbot impersonating Cam Dresie
2. The chatbot's response
3. A scoring criterion
4. Ground truth facts about Cam Dresie

Score the response as PASS or FAIL against the criterion. Reply with ONLY the word "PASS" or "FAIL" — nothing else.

## Ground Truth
{GROUND_TRUTH_CONTEXT}

## Evaluator Notes
{notes}

## Question Asked
{question}

## Agent Response
{response}

## Criterion: {criterion['name']}
{criterion['prompt']}

Your verdict (PASS or FAIL):"""

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": scoring_prompt}],
        max_tokens=5,
        temperature=0,
    )

    verdict = result.choices[0].message.content.strip().upper()
    passed = "PASS" in verdict
    return {"criterion": criterion_id, "name": criterion["name"], "passed": passed, "raw": verdict}


def run_eval(verbose=False):
    """Run the full evaluation. Returns aggregate score as a percentage."""
    print("=" * 60)
    print("CAREER AGENT EVAL")
    print("=" * 60)

    total_checks = 0
    total_passed = 0
    results_by_question = []

    for test in TEST_INPUTS:
        print(f"\nTest {test['id']}: {test['question'][:60]}...")
        response = get_agent_response(test["question"])

        if verbose:
            print(f"  Response: {response[:200]}...")

        question_results = []
        for crit_id in test["applicable_criteria"]:
            score = score_response(test["question"], response, crit_id, test["notes"])
            question_results.append(score)
            total_checks += 1
            if score["passed"]:
                total_passed += 1
            status = "PASS" if score["passed"] else "FAIL"
            print(f"  {crit_id} ({score['name']}): {status}")

        results_by_question.append({
            "test_id": test["id"],
            "question": test["question"],
            "response": response,
            "scores": question_results,
        })

    # Aggregate
    pct = (total_passed / total_checks * 100) if total_checks > 0 else 0

    print("\n" + "=" * 60)
    print(f"SCORE: {total_passed}/{total_checks} = {pct:.1f}%")
    print("=" * 60)

    # Per-criterion breakdown
    crit_totals = {}
    for q in results_by_question:
        for s in q["scores"]:
            cid = s["criterion"]
            if cid not in crit_totals:
                crit_totals[cid] = {"passed": 0, "total": 0, "name": s["name"]}
            crit_totals[cid]["total"] += 1
            if s["passed"]:
                crit_totals[cid]["passed"] += 1

    print("\nPer-criterion breakdown:")
    for cid in sorted(crit_totals.keys()):
        ct = crit_totals[cid]
        rate = ct["passed"] / ct["total"] * 100
        print(f"  {cid}: {ct['passed']}/{ct['total']} ({rate:.0f}%) — {ct['name']}")

    return pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the career agent")
    parser.add_argument("--verbose", action="store_true", help="Print full responses")
    args = parser.parse_args()

    score = run_eval(verbose=args.verbose)
    print(f"\nFinal score: {score:.1f}%")
