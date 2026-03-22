"""
Autonomous autoresearch loop for the Career Agent system prompt.
Replaces the need for a human-in-the-loop coding agent.

This script IS the agent. It reads the current prompt, runs the eval,
asks an LLM what to change, makes the edit, re-runs the eval, and
keeps or reverts — all in a loop with zero human interaction.

Usage:
    python autoresearch.py                    # Run 30 rounds (default)
    python autoresearch.py --max-rounds 10    # Run 10 rounds
    python autoresearch.py --dry-run          # Show what would happen without writing

Requires: OPENAI_API_KEY in environment or .env file
"""

import os
import sys
import subprocess
import json
import argparse
import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPT_FILE = os.path.join(os.path.dirname(__file__) or ".", "system_prompt.md")
RESULTS_LOG = os.path.join(os.path.dirname(__file__) or ".", "results.log")
EVAL_SCRIPT = os.path.join(os.path.dirname(__file__) or ".", "eval.py")
STOP_SCORE = 95.0       # Stop if we hit this 3x in a row
MAX_CONSECUTIVE = 3      # How many times at STOP_SCORE before we stop
MAX_REVERTS_IN_A_ROW = 5 # Give up if stuck


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True,
        cwd=os.path.dirname(__file__) or "."
    )
    if result.returncode != 0 and "fatal" in result.stderr.lower():
        log(f"  git warning: {result.stderr.strip()}")
    return result.stdout.strip()


def git_commit(message: str):
    git("add", "system_prompt.md", "results.log")
    git("commit", "-m", message)


def git_revert():
    git("checkout", "HEAD~1", "--", "system_prompt.md")
    git("commit", "-m", "Revert: score did not improve")


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval() -> dict:
    """
    Run eval.py and parse the output.
    Returns {"score": float, "breakdown": {"C1": "3/4", ...}, "raw": str}
    """
    log("  Running eval...")
    result = subprocess.run(
        [sys.executable, EVAL_SCRIPT],
        capture_output=True, text=True,
        cwd=os.path.dirname(__file__) or ".",
        timeout=300,  # 5 min max
    )
    output = result.stdout + result.stderr

    # Parse score from output like "SCORE: 12/14 = 85.7%"
    score = 0.0
    breakdown = {}
    for line in output.split("\n"):
        if line.strip().startswith("SCORE:"):
            try:
                pct_part = line.split("=")[1].strip().replace("%", "")
                score = float(pct_part)
            except (IndexError, ValueError):
                pass
        # Parse per-criterion lines like "  C1: 4/5 (80%) — No hallucinated facts"
        for cid in ["C1", "C2", "C3", "C4", "C5"]:
            if line.strip().startswith(f"{cid}:"):
                try:
                    parts = line.strip().split("—")[0].strip()
                    # e.g. "C1: 4/5 (80%)"
                    fraction = parts.split(":")[1].strip().split("(")[0].strip()
                    breakdown[cid] = fraction
                except (IndexError, ValueError):
                    pass

    return {"score": score, "breakdown": breakdown, "raw": output}


# ---------------------------------------------------------------------------
# LLM-powered prompt improver
# ---------------------------------------------------------------------------

def suggest_change(current_prompt: str, results_history: str, eval_output: str) -> str:
    """
    Ask an LLM to suggest ONE specific edit to the system prompt.
    Returns the complete new prompt text.
    """
    client = OpenAI()

    meta_prompt = f"""You are an autoresearch agent optimizing a chatbot's system prompt.

## Your job
Look at the eval results and the current prompt. Identify the weakest criterion (lowest pass rate).
Make ONE targeted change to improve it. Return the COMPLETE updated prompt — not a diff, not an explanation,
just the full new prompt text ready to be written to system_prompt.md.

## Rules
- Make exactly ONE change per round. Small, targeted edits only.
- Never remove the {{name}} or {{retrieved_context}} template variables — they get replaced at runtime.
- Never remove the MANDATORY FACT OVERRIDES section.
- Focus on the criterion with the lowest pass rate.
- If a criterion is at 100%, don't touch the instructions that drive it.
- Common fixes: add explicit instructions, add worked examples, add banned phrases, reword vague guidance.
- Do NOT add markdown code fences or any wrapper around your response. Just output the raw prompt text.

## Current system prompt (system_prompt.md):
{current_prompt}

## Latest eval output:
{eval_output}

## Previous rounds (results.log):
{results_history if results_history else "(No previous rounds — this is the first iteration.)"}

Now return the complete updated system_prompt.md content with your ONE improvement:"""

    response = client.chat.completions.create(
        model="gpt-4o",  # Use the stronger model for the "researcher" role
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.7,  # Some creativity in suggestions
        max_tokens=4000,
    )

    new_prompt = response.choices[0].message.content.strip()

    # Strip markdown code fences if the LLM wraps them anyway
    if new_prompt.startswith("```"):
        lines = new_prompt.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        new_prompt = "\n".join(lines)

    return new_prompt


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(max_rounds: int = 30, dry_run: bool = False):
    log(f"Starting autoresearch loop (max {max_rounds} rounds)")
    log(f"Target file: {PROMPT_FILE}")
    log(f"Eval script: {EVAL_SCRIPT}")

    # Ensure we're on a feature branch
    branch = git("branch", "--show-current")
    if branch in ("main", "master"):
        new_branch = f"autoresearch/prompt-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        log(f"On {branch}, creating branch: {new_branch}")
        if not dry_run:
            git("checkout", "-b", new_branch)
    else:
        log(f"On branch: {branch}")

    # --- Phase 1: Baseline ---
    log("=" * 60)
    log("PHASE 1: BASELINE")
    log("=" * 60)

    baseline = run_eval()
    log(f"Baseline score: {baseline['score']:.1f}%")
    for cid, frac in sorted(baseline["breakdown"].items()):
        log(f"  {cid}: {frac}")

    # Initialize results log
    if not dry_run:
        with open(RESULTS_LOG, "a") as f:
            f.write(f"BASELINE: {baseline['score']:.1f}%\n")
            for cid, frac in sorted(baseline["breakdown"].items()):
                f.write(f"  {cid}: {frac}\n")
            f.write("\n")
        git("add", "results.log")
        git("commit", "-m", f"Autoresearch baseline: {baseline['score']:.1f}%", "--allow-empty")

    # --- Phase 2: Improvement loop ---
    log("=" * 60)
    log("PHASE 2: AUTONOMOUS IMPROVEMENT LOOP")
    log("=" * 60)

    prev_score = baseline["score"]
    consecutive_at_target = 0
    consecutive_reverts = 0

    for round_num in range(1, max_rounds + 1):
        log(f"\n--- Round {round_num}/{max_rounds} ---")

        # Read current state
        with open(PROMPT_FILE, "r") as f:
            current_prompt = f.read()
        results_history = ""
        if os.path.exists(RESULTS_LOG):
            with open(RESULTS_LOG, "r") as f:
                results_history = f.read()

        # Get LLM suggestion
        log("  Asking LLM for improvement suggestion...")
        new_prompt = suggest_change(current_prompt, results_history, baseline["raw"] if round_num == 1 else last_eval_output)

        if new_prompt.strip() == current_prompt.strip():
            log("  LLM returned identical prompt. Skipping round.")
            continue

        if dry_run:
            log("  [DRY RUN] Would write new prompt and run eval. Skipping.")
            continue

        # Write the change
        with open(PROMPT_FILE, "w") as f:
            f.write(new_prompt)

        # Describe what changed (first 200 chars of diff)
        diff_output = git("diff", "system_prompt.md")
        change_summary = diff_output[:500] if diff_output else "(no visible diff)"
        log(f"  Change applied. Running eval...")

        # Commit before eval so we can revert cleanly
        git_commit(f"Round {round_num}: prompt improvement attempt")

        # Run eval
        eval_result = run_eval()
        new_score = eval_result["score"]
        last_eval_output = eval_result["raw"]

        log(f"  Score: {new_score:.1f}% (was {prev_score:.1f}%)")
        for cid, frac in sorted(eval_result["breakdown"].items()):
            log(f"    {cid}: {frac}")

        # Keep or revert
        if new_score > prev_score:
            verdict = "KEPT"
            log(f"  >> KEPT (improved by {new_score - prev_score:.1f}pp)")
            prev_score = new_score
            consecutive_reverts = 0
        elif new_score == prev_score:
            # Tie goes to revert — we want clear improvement
            verdict = "REVERTED (no improvement)"
            log(f"  >> REVERTED (score unchanged)")
            git_revert()
            consecutive_reverts += 1
        else:
            verdict = "REVERTED (regression)"
            log(f"  >> REVERTED (dropped by {prev_score - new_score:.1f}pp)")
            git_revert()
            consecutive_reverts += 1

        # Log results
        with open(RESULTS_LOG, "a") as f:
            breakdown_str = " | ".join(f"{cid}: {frac}" for cid, frac in sorted(eval_result["breakdown"].items()))
            f.write(f"ROUND {round_num}: {new_score:.1f}% (was {prev_score:.1f}%) — {verdict}\n")
            f.write(f"  {breakdown_str}\n\n")
        git("add", "results.log")
        git("commit", "-m", f"Round {round_num} results: {new_score:.1f}% — {verdict}")

        # Check stop conditions
        if prev_score >= STOP_SCORE:
            consecutive_at_target += 1
            if consecutive_at_target >= MAX_CONSECUTIVE:
                log(f"\n>> STOPPING: Hit {STOP_SCORE}%+ for {MAX_CONSECUTIVE} consecutive rounds!")
                break
        else:
            consecutive_at_target = 0

        if consecutive_reverts >= MAX_REVERTS_IN_A_ROW:
            log(f"\n>> STOPPING: {MAX_REVERTS_IN_A_ROW} consecutive reverts. Possibly stuck.")
            break

    # --- Phase 3: Summary ---
    log("\n" + "=" * 60)
    log("PHASE 3: SUMMARY")
    log("=" * 60)
    log(f"Started at: {baseline['score']:.1f}%")
    log(f"Ended at:   {prev_score:.1f}%")
    log(f"Rounds run: {round_num}")
    log(f"Results saved to: {RESULTS_LOG}")

    # Save final improved prompt as a separate file for easy comparison
    final_path = os.path.join(os.path.dirname(__file__) or ".", "system_prompt_improved.md")
    with open(PROMPT_FILE, "r") as f:
        final_prompt = f.read()
    with open(final_path, "w") as f:
        f.write(final_prompt)
    log(f"Final prompt also saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous autoresearch loop")
    parser.add_argument("--max-rounds", type=int, default=30, help="Max improvement rounds")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing")
    args = parser.parse_args()

    run(max_rounds=args.max_rounds, dry_run=args.dry_run)
