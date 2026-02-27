import os
import json
import time
import traceback
import statistics
from typing import Union

from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# ------------------------------------------------------------------
# 1. SETUP: API Key and Client
# ------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ------------------------------------------------------------------
# 2. DEFINITIONS: The System Prompt (The Logic Engine)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert AI Talent Assessor and Recruitment Analyst. Your task is to process a bulk list of candidates against a single Job Description (JD) using the 4-Stage Agent Check process.

### THE 4-STAGE AGENT CHECK PROCESS
For *each* candidate provided in the input list, you must simulate these agents:

1. **Agent 1: Resume Analysis (20%):** Extract core data (skills, experience, tenure).
2. **Agent 2: Skills & Job Alignment (40%):** Compare candidate skills and details against JD requirements.
3. **Agent 3: Project & Growth Potential (25%):** Analyze project depth and learning agility and the relevance to the role.
4. **Agent 4: HR Evaluation Summary (15%):** Assess overall suitability and cultural fit.

### SCORING RULES
* **Overall Match Score (0-100):** >85 (Excellent), 70-84 (Good), 50-69 (Moderate), <50 (Poor).
* Scores MUST be calibrated against an ideal candidate for this JD — do NOT inflate scores just because a candidate is the best in this batch.
* **Highlights:** Identify unique, high-value traits (e.g., patents, open-source maintainer).

### INPUT & OUTPUT FORMAT
You will receive a JSON input containing a `job_description` and a list of `candidates`.
You must return a JSON object with a single key `bulk_evaluations` containing an array of evaluation objects.

**Required Output JSON Structure:**

{
  "bulk_evaluations": [
    {
      "candidate_id": "Refers to ID from input",
      "candidate_name": "Extracted Name",
      "overall_match_score": 0,
      "phone": "Extracted Phone Number",
      "email": "Extracted Email",
      "experience_in_skills": {
        "Skill A": "3 years 0 months",
        "Skill B": "2 years 6 months"
      },
      "total_work_experience_years": "10 years 0 months",
      "relevant_experience_years": "8 years 0 months",
      "scoring_tier": "Excellent/Good/Moderate/Poor",
      "agent_checks": {
        "agent_1_summary": "Brief extraction summary.",
        "agent_2_skills_gap": {
          "missing_critical": ["Skill A"],
          "strong_matches": ["Skill B"]
        },
        "agent_3_project_depth": "High/Medium/Low assessment along with a brief explanation.",
        "agent_4_recommendation": {
          "recommendation": "Interview/Reject",
          "reasoning": "Brief explanation for recommendation"
        }
      },
      "strengths": ["Point 1", "Point 2"],
      "weaknesses": ["Point 1", "Point 2"],
      "special_highlights": ["Unique trait 1"]
    }
  ]
}
"""

# ------------------------------------------------------------------
# 3. TOKEN COUNTING SETUP
# ------------------------------------------------------------------
# gpt-4o uses o200k_base encoding; gpt-4-turbo uses cl100k_base
_ENC = tiktoken.get_encoding("o200k_base")

def count_tokens(text: str) -> int:
    """Returns the real token count for a given string."""
    return len(_ENC.encode(text))


# ------------------------------------------------------------------
# 4. TPM BUDGET CONSTANTS
# ------------------------------------------------------------------
TPM_LIMIT             = 30_000
SYSTEM_PROMPT_TOKENS  = count_tokens(SYSTEM_PROMPT)   # exact, not estimated
JD_SKILLS_TOKENS      = 1_000   # estimated; overridden dynamically per call
OUTPUT_TOKENS_EACH    = 500     # estimated output tokens per candidate
BATCH_OVERHEAD        = SYSTEM_PROMPT_TOKENS + JD_SKILLS_TOKENS


# ------------------------------------------------------------------
# 5. TOKEN-AWARE BATCH BUILDER
# ------------------------------------------------------------------
def build_token_aware_batches(candidates_data: list[dict]) -> list[list[dict]]:
    """
    Groups candidates into batches so each batch's total token usage
    (input + expected output) stays within the 30k TPM limit.

    Budget per batch:
        available_input = TPM_LIMIT
                          - SYSTEM_PROMPT_TOKENS
                          - JD_SKILLS_TOKENS
                          - (projected_n_candidates × OUTPUT_TOKENS_EACH)
    """
    batches        = []
    current_batch  = []
    current_tokens = 0

    for cand in candidates_data:
        resume_tokens = cand["_token_count"]

        # How many candidates if we add this one?
        projected_n   = len(current_batch) + 1
        output_budget = projected_n * OUTPUT_TOKENS_EACH
        available     = TPM_LIMIT - BATCH_OVERHEAD - output_budget

        if current_batch and (current_tokens + resume_tokens) > available:
            # Seal the current batch and start fresh
            batches.append(current_batch)
            current_batch  = []
            current_tokens = 0

        current_batch.append(cand)
        current_tokens += resume_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


# ------------------------------------------------------------------
# 6. SINGLE-BATCH LLM CALL WITH RETRY
# ------------------------------------------------------------------
def call_llm_with_retry(
    job_description: str,
    required_skills: list,
    batch: list[dict],
    batch_num: int,
    total_batches: int,
    max_retries: int = 3
) -> list[dict]:
    """
    Sends one batch to GPT-4o and returns its bulk_evaluations list.
    Returns [] on unrecoverable failure.
    """
    # Strip internal fields before sending to API
    api_batch = [{"id": c["id"], "resume_text": c["resume_text"]} for c in batch]

    user_message_payload = {
        "job_description": job_description,
        "required_skills": required_skills,
        "candidates": api_batch
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(user_message_payload)}
                ],
                temperature=0.1
            )

            result_json = json.loads(response.choices[0].message.content)

            if "bulk_evaluations" in result_json:
                results = result_json["bulk_evaluations"]
                print(f"✅ Batch {batch_num}/{total_batches} — {len(results)} candidate(s) evaluated")
                return results
            else:
                print(f"❌ Batch {batch_num}: Response missing 'bulk_evaluations' key")
                return []

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                wait = 2 ** (attempt + 2)   # 4 s → 8 s → 16 s
                print(f"⚠️  Rate limit on batch {batch_num}, retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"❌ Batch {batch_num} failed after {attempt + 1} attempt(s): {e}")
                traceback.print_exc()
                return []

    return []


# ------------------------------------------------------------------
# 7. CROSS-BATCH CALIBRATION & GLOBAL RANKING
# ------------------------------------------------------------------
def calibrate_and_rank(all_results: list[dict]) -> list[dict]:
    """
    Produces a global rank 1…N across all batches.

    Default strategy: direct sort by overall_match_score (scores are already
    anchored to the JD ideal via the system prompt, not relative to batch peers).

    Optional Z-score normalization is included but commented out — enable it
    if you observe systematic score inflation/deflation between batches.
    """
    if not all_results:
        return []

    # ── Option A (default): direct sort ──────────────────────────────────────
    ranked = sorted(
        all_results,
        key=lambda x: x.get("overall_match_score", 0),
        reverse=True
    )

    # ── Option B: Z-score normalization per batch (uncomment to enable) ───────
    # Useful when you notice batch 1 clusters around 80 and batch 2 around 60.
    #
    # batch_groups = {}
    # for r in all_results:
    #     bid = r.pop("_batch_id", 0)
    #     batch_groups.setdefault(bid, []).append(r)
    #
    # normalized = []
    # for bid, group in batch_groups.items():
    #     scores = [g["overall_match_score"] for g in group]
    #     mu     = statistics.mean(scores)
    #     sigma  = statistics.stdev(scores) if len(scores) > 1 else 1
    #     for g in group:
    #         g["_normalized_score"] = (g["overall_match_score"] - mu) / (sigma or 1)
    #         normalized.append(g)
    #
    # ranked = sorted(normalized,
    #                 key=lambda x: x.get("_normalized_score", 0),
    #                 reverse=True)
    # for r in ranked:
    #     r.pop("_normalized_score", None)

    # Assign global rank and clean up internal fields
    for rank, candidate in enumerate(ranked, start=1):
        candidate["global_rank"] = rank
        candidate.pop("_batch_id", None)

    return ranked


# ------------------------------------------------------------------
# 8. MAIN BULK EVALUATION FUNCTION
# ------------------------------------------------------------------
def evaluate_candidates_bulk(
    candidate_resumes: Union[list, str],
    job_description: str,
    required_skills: list,
) -> list[dict]:
    """
    Evaluate multiple candidates against a job description and return a
    globally ranked list (global_rank=1 is the best fit).

    Batching is driven by real-time tiktoken counts so each API call stays
    comfortably within GPT-4o's 30,000 TPM limit.

    Parameters
    ──────────
    candidate_resumes : list[str] | str
        Either a list of resume strings, or a single string with resumes
        separated by "\\n\\n--- RESUME SEPARATOR ---\\n\\n".
    job_description   : str
    required_skills   : list[str]

    Returns
    ───────
    list[dict] — each dict contains all fields from the system prompt output,
                 plus a `global_rank` field (1 = best fit).
    """
    try:
        # ── 1. Normalise input ────────────────────────────────────────────────
        if isinstance(candidate_resumes, list):
            raw_resumes = candidate_resumes
        else:
            raw_resumes = [
                r for r in candidate_resumes.split("\n\n--- RESUME SEPARATOR ---\n\n")
                if r.strip()
            ]

        candidates_data = [
            {"id": f"cand_{i+1}", "resume_text": resume}
            for i, resume in enumerate(raw_resumes)
        ]

        print(f"📋 Total candidates: {len(candidates_data)}")

        # ── 2. Count tokens per resume ────────────────────────────────────────
        for cand in candidates_data:
            cand["_token_count"] = count_tokens(cand["resume_text"])

        total_resume_tokens = sum(c["_token_count"] for c in candidates_data)
        print(f"🔢 Total resume tokens: {total_resume_tokens:,}  |  "
              f"System prompt: {SYSTEM_PROMPT_TOKENS} tokens")

        # ── 3. Build token-aware batches ──────────────────────────────────────
        batches       = build_token_aware_batches(candidates_data)
        total_batches = len(batches)
        print(f"📦 Batches created: {total_batches}")
        for i, b in enumerate(batches, 1):
            batch_tokens = sum(c["_token_count"] for c in b)
            print(f"   Batch {i}: {len(b)} candidate(s), ~{batch_tokens:,} resume tokens")

        # ── 4. Evaluate each batch ────────────────────────────────────────────
        all_results = []

        for batch_idx, batch in enumerate(batches):
            batch_num = batch_idx + 1
            print(f"\n🚀 Processing batch {batch_num}/{total_batches} "
                  f"({len(batch)} candidate(s))…")

            results = call_llm_with_retry(
                job_description, required_skills,
                batch, batch_num, total_batches
            )

            # Tag with batch id for optional calibration
            for r in results:
                r["_batch_id"] = batch_num

            all_results.extend(results)

            # ── Throttle: wait long enough for this batch's tokens to replenish
            if batch_num < total_batches:
                batch_input_tokens = sum(c["_token_count"] for c in batch)
                batch_total_tokens = (
                    BATCH_OVERHEAD
                    + batch_input_tokens
                    + len(batch) * OUTPUT_TOKENS_EACH
                )
                # seconds = (tokens_used / TPM_LIMIT) * 60s + 2s buffer
                wait_seconds = max(5, int((batch_total_tokens / TPM_LIMIT) * 60) + 2)
                print(f"⏳ Waiting {wait_seconds}s before next batch "
                      f"(used ~{batch_total_tokens:,} tokens this batch)…")
                time.sleep(wait_seconds)

        # ── 5. Global ranking ─────────────────────────────────────────────────
        print(f"\n🏆 Ranking {len(all_results)} candidates globally…")
        ranked_results = calibrate_and_rank(all_results)

        # ── 6. Print summary table ────────────────────────────────────────────
        print(f"\n✅ Done — {len(ranked_results)} candidates ranked.\n")
        print(f"{'Rank':<6} {'ID':<12} {'Score':<8} {'Tier':<12} Name")
        print("─" * 65)
        for r in ranked_results:
            print(
                f"{r['global_rank']:<6} "
                f"{r.get('candidate_id', r.get('id', '?')):<12} "
                f"{r.get('overall_match_score', '?'):<8} "
                f"{r.get('scoring_tier', '?'):<12} "
                f"{r.get('candidate_name', 'Unknown')}"
            )
        print(ranked_results)
        return ranked_results

    except Exception as e:
        print(f"❌ Error in evaluate_candidates_bulk: {e}")
        traceback.print_exc()
        return []


# ------------------------------------------------------------------
# 9. SIMPLE SINGLE-CALL EVALUATION (original run_evaluation)
# ------------------------------------------------------------------
job_description_text = """
We are looking for a Senior Backend Engineer to join our FinTech team.
Must Haves: 5+ years Python, Experience with Django/FastAPI, PostgreSQL scaling, and AWS (Lambda, EC2).
Nice to Haves: Experience with Kubernetes, Blockchain knowledge, and previous mentorship experience.
"""

_sample_candidates = [
    {
        "id": "cand_101",
        "name": "Alex Johnson",
        "resume_text": (
            "Alex Johnson. 6 years of experience in Python. Expert in Flask and Django. "
            "Built scalable payment systems using PostgreSQL. Certified AWS Solutions Architect. "
            "I have never used Kubernetes."
        )
    },
    {
        "id": "cand_102",
        "name": "Sam Smith",
        "resume_text": (
            "Sam Smith. Junior Developer. 1 year experience in Java and C++. "
            "Passionate about crypto. Fast learner. Willing to learn Python."
        )
    }
]

def run_evaluation():
    """Quick single-call evaluation for the two sample candidates above."""
    try:
        print("Sending request to OpenAI…")

        user_message_payload = {
            "job_description": job_description_text,
            "candidates": _sample_candidates
        }

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(user_message_payload)}
            ],
            temperature=0.1
        )

        result_json = json.loads(response.choices[0].message.content)
        return result_json

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# ------------------------------------------------------------------
# 10. MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    # ── A) Quick single-call demo ─────────────────────────────────────────────
    results = run_evaluation()

    if results:
        print("\n=== EVALUATION RESULTS ===")
        print(json.dumps(results, indent=2))

        print("\n=== SUMMARY ===")
        for eval_item in results["bulk_evaluations"]:
            print(
                f"Candidate: {eval_item['candidate_name']} | "
                f"Score: {eval_item['overall_match_score']} "
                f"({eval_item['scoring_tier']})"
            )

    # ── B) Bulk evaluation with token-aware batching + global ranking ─────────
    # Uncomment below to run the full bulk pipeline on many resumes:
    #
    # sample_resumes = [c["resume_text"] for c in _sample_candidates]
    # ranked = evaluate_candidates_bulk(
    #     candidate_resumes=sample_resumes,
    #     job_description=job_description_text,
    #     required_skills=["Python", "Django", "FastAPI", "PostgreSQL", "AWS"]
    # )
    # print("\n=== FULL RANKED OUTPUT ===")
    # print(json.dumps(ranked, indent=2))