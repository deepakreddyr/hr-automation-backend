import os
import json
import time
import traceback
import statistics
import math
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
# 3. TOURNAMENT SYSTEM PROMPT
# ------------------------------------------------------------------
TOURNAMENT_SYSTEM_PROMPT = """
You are an expert AI Recruitment Analyst performing a final head-to-head ranking of shortlisted candidates.

You will receive:
- A job description
- A list of finalist candidates with their resume snippet and initial evaluation scores

Your task is to carefully compare ALL candidates against each other and against the JD requirements,
then return a definitive ranked order from best to worst fit.

### RANKING RULES
- Rank purely on fit for the JD: skills match, experience depth, and growth potential
- Do NOT be influenced by the initial scores; use them only as a reference point
- Consider subtle differentiators: project complexity, leadership signals, domain relevance
- Every candidate must appear in your ranked output exactly once

### OUTPUT FORMAT
Return a JSON object with a single key `tournament_ranking` containing an array ordered from best (rank 1) to worst:

{
  "tournament_ranking": [
    {
      "candidate_id": "cand_X",
      "final_rank": 1,
      "final_score": 91,
      "ranking_rationale": "Brief explanation of why this candidate ranks here relative to others"
    }
  ]
}
"""

# ------------------------------------------------------------------
# 4. TOKEN COUNTING SETUP
# ------------------------------------------------------------------
_ENC = tiktoken.get_encoding("o200k_base")  # gpt-4o encoding

def count_tokens(text: str) -> int:
    """Returns the real token count for a given string."""
    return len(_ENC.encode(text))


# ------------------------------------------------------------------
# 5. TPM BUDGET CONSTANTS
# ------------------------------------------------------------------
TPM_LIMIT             = 30_000
SYSTEM_PROMPT_TOKENS  = count_tokens(SYSTEM_PROMPT)
JD_SKILLS_TOKENS      = 1_000   # estimated; covers job_description + required_skills
OUTPUT_TOKENS_EACH    = 500     # estimated output tokens per candidate
BATCH_OVERHEAD        = SYSTEM_PROMPT_TOKENS + JD_SKILLS_TOKENS

# Tournament-specific budget
TOURNAMENT_SYSTEM_TOKENS  = count_tokens(TOURNAMENT_SYSTEM_PROMPT)
TOURNAMENT_OUTPUT_TOKENS  = 300   # per finalist: id + rank + score + rationale
TOURNAMENT_CANDIDATE_SIZE = 800   # tokens per finalist summary sent to tournament


# ------------------------------------------------------------------
# 6. TOKEN-AWARE BATCH BUILDER
# ------------------------------------------------------------------
def build_token_aware_batches(candidates_data: list[dict]) -> list[list[dict]]:
    """
    Groups candidates into batches so each batch's total token usage
    (input + expected output) stays within the 30k TPM limit.
    """
    batches        = []
    current_batch  = []
    current_tokens = 0

    for cand in candidates_data:
        resume_tokens = cand["_token_count"]
        projected_n   = len(current_batch) + 1
        output_budget = projected_n * OUTPUT_TOKENS_EACH
        available     = TPM_LIMIT - BATCH_OVERHEAD - output_budget

        if current_batch and (current_tokens + resume_tokens) > available:
            batches.append(current_batch)
            current_batch  = []
            current_tokens = 0

        current_batch.append(cand)
        current_tokens += resume_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


# ------------------------------------------------------------------
# 7. SINGLE-BATCH LLM CALL WITH RETRY
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
                wait = 2 ** (attempt + 2)
                print(f"⚠️  Rate limit on batch {batch_num}, retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"❌ Batch {batch_num} failed after {attempt + 1} attempt(s): {e}")
                traceback.print_exc()
                return []

    return []


# ------------------------------------------------------------------
# 8. TOURNAMENT RANKING
# ------------------------------------------------------------------

def _build_tournament_batches(finalists: list[dict], job_description: str) -> list[list[dict]]:
    """
    Splits finalists into sub-groups that each fit within the TPM limit
    for the tournament prompt.
    """
    available = (
        TPM_LIMIT
        - TOURNAMENT_SYSTEM_TOKENS
        - count_tokens(job_description)
        - 200  # buffer for wrapper JSON keys
    )

    batches        = []
    current_batch  = []
    current_tokens = 0

    for f in finalists:
        candidate_tokens = TOURNAMENT_CANDIDATE_SIZE + TOURNAMENT_OUTPUT_TOKENS

        if current_batch and (current_tokens + candidate_tokens) > available:
            batches.append(current_batch)
            current_batch  = []
            current_tokens = 0

        current_batch.append(f)
        current_tokens += candidate_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _run_tournament_round(
    finalists: list[dict],
    job_description: str,
    resume_lookup: dict[str, str],
    round_label: str,
    max_retries: int = 3
) -> list[dict]:
    """
    Sends a group of finalists to the tournament LLM for head-to-head ranking.
    Includes actual resume snippets so the LLM can compare candidates properly.
    Returns the tournament_ranking list, or [] on failure.
    """
    tournament_candidates = []
    for f in finalists:
        cid = f.get("candidate_id", f.get("id", "unknown"))
        tournament_candidates.append({
            "candidate_id"        : cid,
            "candidate_name"      : f.get("candidate_name", "Unknown"),
            "initial_score"       : f.get("overall_match_score", 0),
            "scoring_tier"        : f.get("scoring_tier", ""),
            "strengths"           : f.get("strengths", []),
            "weaknesses"          : f.get("weaknesses", []),
            "relevant_experience" : f.get("relevant_experience_years", ""),
            "resume_snippet"      : resume_lookup.get(cid, "")[:2000]  # ~500 tokens
        })

    user_payload = {
        "job_description" : job_description,
        "finalists"       : tournament_candidates
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": TOURNAMENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(user_payload)}
                ],
                temperature=0.0   # deterministic for ranking
            )

            result_json = json.loads(response.choices[0].message.content)

            if "tournament_ranking" in result_json:
                print(f"✅ Tournament {round_label} — {len(result_json['tournament_ranking'])} candidates ranked")
                return result_json["tournament_ranking"]
            else:
                print(f"❌ Tournament {round_label}: Response missing 'tournament_ranking' key")
                return []

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                wait = 2 ** (attempt + 2)
                print(f"⚠️  Rate limit on tournament {round_label}, retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"❌ Tournament {round_label} failed: {e}")
                traceback.print_exc()
                return []

    return []


def run_tournament(
    all_results: list[dict],
    job_description: str,
    resume_lookup: dict[str, str],
    top_pct: float = 0.20,
    min_finalists: int = 5,
    max_finalists: int = 20
) -> list[dict]:
    """
    3-phase tournament ranking:

    Phase 1 — Score-based pre-sort
        Sort ALL candidates by initial overall_match_score. This gives a
        reasonable baseline. The bottom (100 - top_pct)% skip the tournament
        and keep their score-based rank.

    Phase 2 — Finalist head-to-head tournament
        Take top N% (clamped between min_finalists and max_finalists) and run
        them through the tournament LLM for true head-to-head comparison.

        If finalists exceed token budget for one call, they are split into
        sub-groups. Each sub-group is ranked internally, then group winners
        compete in a grand final to determine the overall top order.

    Phase 3 — Merge & assign global_rank
        Tournament-ranked finalists → ranks 1…F  (method: "tournament")
        Remaining candidates       → ranks F+1…N (method: "score_based")

    Parameters
    ──────────
    all_results    : flat list of all evaluation dicts from batch scoring
    job_description: original JD string
    resume_lookup  : dict mapping candidate_id → original resume_text
    top_pct        : fraction of candidates to enter tournament (default 0.20)
    min_finalists  : minimum finalist pool size (default 5)
    max_finalists  : cap on finalist pool to control cost (default 20)
    """
    if not all_results:
        return []

    n_total = len(all_results)

    # ── Phase 1: Pre-sort everyone by initial score ───────────────────────────
    pre_sorted = sorted(
        all_results,
        key=lambda x: x.get("overall_match_score", 0),
        reverse=True
    )

    # ── Determine finalist pool size ──────────────────────────────────────────
    n_finalists = max(min_finalists, math.ceil(n_total * top_pct))
    n_finalists = min(n_finalists, max_finalists, n_total)

    finalists     = pre_sorted[:n_finalists]
    non_finalists = pre_sorted[n_finalists:]

    print(f"\n🏟️  Tournament: {n_finalists} finalists from {n_total} total "
          f"(top {top_pct*100:.0f}%, min={min_finalists}, max={max_finalists})")

    # ── Phase 2: Head-to-head tournament ─────────────────────────────────────
    tournament_batches   = _build_tournament_batches(finalists, job_description)
    n_tournament_batches = len(tournament_batches)
    print(f"   Tournament split into {n_tournament_batches} sub-group(s)")

    if n_tournament_batches == 1:
        # ── Simple case: all finalists fit in one call ────────────────────────
        final_tournament_ranking = _run_tournament_round(
            finalists, job_description, resume_lookup,
            round_label="Round 1 (Final)"
        )

    else:
        # ── Multi-group: rank within groups, then grand final between winners ─
        sub_group_winners     = []
        sub_group_all_ranked  = []

        for grp_idx, grp in enumerate(tournament_batches):
            grp_label = f"Group {grp_idx + 1}/{n_tournament_batches}"
            print(f"   Running tournament {grp_label}…")

            if grp_idx > 0:
                time.sleep(8)   # throttle between sub-group calls

            grp_ranking = _run_tournament_round(
                grp, job_description, resume_lookup, round_label=grp_label
            )

            if grp_ranking:
                grp_sorted = sorted(grp_ranking, key=lambda x: x.get("final_rank", 999))
                sub_group_all_ranked.append(grp_sorted)
                sub_group_winners.append(grp_sorted[0])
            else:
                # Fallback: use score order if tournament call failed
                grp_score_sorted = sorted(grp, key=lambda x: x.get("overall_match_score", 0), reverse=True)
                fallback_ranking = [
                    {
                        "candidate_id"     : c.get("candidate_id", c.get("id")),
                        "final_rank"       : i + 1,
                        "final_score"      : c.get("overall_match_score", 0),
                        "ranking_rationale": "fallback (API error)"
                    }
                    for i, c in enumerate(grp_score_sorted)
                ]
                sub_group_all_ranked.append(fallback_ranking)
                sub_group_winners.append(fallback_ranking[0])

        # ── Grand final: tournament between group winners ─────────────────────
        print(f"\n   🏆 Grand Final: {len(sub_group_winners)} group winner(s)…")
        time.sleep(5)

        eval_lookup = {
            r.get("candidate_id", r.get("id")): r for r in all_results
        }
        grand_final_candidates = [
            eval_lookup[w["candidate_id"]]
            for w in sub_group_winners
            if w["candidate_id"] in eval_lookup
        ]

        grand_final_ranking = _run_tournament_round(
            grand_final_candidates, job_description, resume_lookup,
            round_label="Grand Final"
        )

        if not grand_final_ranking:
            grand_final_ranking = [
                {
                    "candidate_id"     : w["candidate_id"],
                    "final_rank"       : i + 1,
                    "final_score"      : w.get("final_score", 0),
                    "ranking_rationale": "fallback (API error)"
                }
                for i, w in enumerate(sub_group_winners)
            ]

        # Grand final order → group runners-up in sub-group rank order
        grand_final_ids = [
            r["candidate_id"]
            for r in sorted(grand_final_ranking, key=lambda x: x["final_rank"])
        ]

        remaining_ranked_ids = []
        for grp_results in sub_group_all_ranked:
            for r in grp_results:
                if r["candidate_id"] not in grand_final_ids:
                    remaining_ranked_ids.append(r["candidate_id"])

        # Build the merged final ranking list
        final_tournament_ranking = [
            {
                "candidate_id"     : cid,
                "final_rank"       : i + 1,
                "final_score"      : 0,
                "ranking_rationale": "grand final winner"
            }
            for i, cid in enumerate(grand_final_ids)
        ] + [
            {
                "candidate_id"     : cid,
                "final_rank"       : len(grand_final_ids) + i + 1,
                "final_score"      : 0,
                "ranking_rationale": "group stage"
            }
            for i, cid in enumerate(remaining_ranked_ids)
        ]

    # ── Phase 3: Merge tournament results with non-finalists ─────────────────
    tournament_lookup = {r["candidate_id"]: r for r in final_tournament_ranking}
    eval_lookup       = {r.get("candidate_id", r.get("id")): r for r in all_results}

    finalist_ids_by_rank = [
        r["candidate_id"]
        for r in sorted(final_tournament_ranking, key=lambda x: x.get("final_rank", 999))
    ]

    final_ranked = []

    # Tournament-verified finalists → ranks 1…F
    for rank, cid in enumerate(finalist_ids_by_rank, start=1):
        if cid in eval_lookup:
            candidate = eval_lookup[cid].copy()
            t_result  = tournament_lookup.get(cid, {})
            candidate["global_rank"]            = rank
            candidate["ranking_method"]         = "tournament"
            candidate["tournament_final_score"] = t_result.get("final_score", candidate.get("overall_match_score", 0))
            candidate["tournament_rationale"]   = t_result.get("ranking_rationale", "")
            final_ranked.append(candidate)

    # Score-based non-finalists → ranks F+1…N
    finalist_set = set(finalist_ids_by_rank)
    offset       = len(finalist_ids_by_rank)

    for rank_offset, candidate in enumerate(non_finalists, start=1):
        cid = candidate.get("candidate_id", candidate.get("id"))
        if cid not in finalist_set:
            c = candidate.copy()
            c["global_rank"]    = offset + rank_offset
            c["ranking_method"] = "score_based"
            final_ranked.append(c)

    # Clean up internal fields
    for r in final_ranked:
        r.pop("_batch_id", None)

    return final_ranked


# ------------------------------------------------------------------
# 9. MAIN BULK EVALUATION FUNCTION
# ------------------------------------------------------------------
def evaluate_candidates_bulk(
    candidate_resumes: Union[list, str],
    job_description: str,
    required_skills: list,
    tournament_top_pct: float = 0.20,
    tournament_min_finalists: int = 5,
    tournament_max_finalists: int = 20
) -> list[dict]:
    """
    Evaluate multiple candidates against a job description and return a
    globally ranked list (global_rank=1 is the best fit).

    Pipeline:
        1. Tokenize resumes with tiktoken (real counts, not estimates)
        2. Build token-aware batches (each call stays under 30k TPM)
        3. Score all candidates across batches via GPT-4o
        4. Run tournament on top N% for true head-to-head comparison
        5. Merge and return final ranked list 1…N

    Parameters
    ──────────
    candidate_resumes         : list[str] | str (separated by RESUME SEPARATOR)
    job_description           : str
    required_skills           : list[str]
    tournament_top_pct        : fraction entering tournament (default 0.20)
    tournament_min_finalists  : always at least this many in tournament (default 5)
    tournament_max_finalists  : cap finalists to control API cost (default 20)

    Returns
    ───────
    list[dict] — full evaluation dicts with:
        global_rank            : 1 = best fit
        ranking_method         : "tournament" | "score_based"
        tournament_final_score : refined score (tournament candidates only)
        tournament_rationale   : LLM explanation of rank (tournament candidates only)
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

        # Build resume lookup for tournament phase
        resume_lookup = {c["id"]: c["resume_text"] for c in candidates_data}

        # ── 3. Build token-aware batches ──────────────────────────────────────
        batches       = build_token_aware_batches(candidates_data)
        total_batches = len(batches)
        print(f"📦 Scoring batches: {total_batches}")
        for i, b in enumerate(batches, 1):
            batch_tokens = sum(c["_token_count"] for c in b)
            print(f"   Batch {i}: {len(b)} candidate(s), ~{batch_tokens:,} resume tokens")

        # ── 4. Score all batches ──────────────────────────────────────────────
        all_results = []

        for batch_idx, batch in enumerate(batches):
            batch_num = batch_idx + 1
            print(f"\n🚀 Scoring batch {batch_num}/{total_batches} "
                  f"({len(batch)} candidate(s))…")

            results = call_llm_with_retry(
                job_description, required_skills,
                batch, batch_num, total_batches
            )

            for r in results:
                r["_batch_id"] = batch_num

            all_results.extend(results)

            # Throttle between batches
            if batch_num < total_batches:
                batch_input_tokens = sum(c["_token_count"] for c in batch)
                batch_total_tokens = (
                    BATCH_OVERHEAD
                    + batch_input_tokens
                    + len(batch) * OUTPUT_TOKENS_EACH
                )
                wait_seconds = max(5, int((batch_total_tokens / TPM_LIMIT) * 60) + 2)
                print(f"⏳ Waiting {wait_seconds}s (used ~{batch_total_tokens:,} tokens)…")
                time.sleep(wait_seconds)

        print(f"\n📊 Scoring complete — {len(all_results)} candidates evaluated")

        # ── 5. Build enriched resume_lookup keyed by candidate_id from LLM ───
        # The LLM echoes back the "id" we sent as "candidate_id"
        enriched_resume_lookup = {}
        for r in all_results:
            cid = r.get("candidate_id", r.get("id", ""))
            if cid in resume_lookup:
                enriched_resume_lookup[cid] = resume_lookup[cid]

        # ── 6. Tournament ranking ─────────────────────────────────────────────
        final_ranked = run_tournament(
            all_results     = all_results,
            job_description = job_description,
            resume_lookup   = enriched_resume_lookup,
            top_pct         = tournament_top_pct,
            min_finalists   = tournament_min_finalists,
            max_finalists   = tournament_max_finalists
        )

        # ── 7. Print summary table ────────────────────────────────────────────
        print(f"\n✅ Final ranking complete — {len(final_ranked)} candidates\n")
        print(f"{'Rank':<6} {'Method':<12} {'Init':<6} {'T-Score':<9} {'Tier':<12} Name")
        print("─" * 72)
        for r in final_ranked:
            t_score = r.get("tournament_final_score", "—")
            print(
                f"{r['global_rank']:<6} "
                f"{r.get('ranking_method', '?'):<12} "
                f"{r.get('overall_match_score', '?'):<6} "
                f"{str(t_score):<9} "
                f"{r.get('scoring_tier', '?'):<12} "
                f"{r.get('candidate_name', 'Unknown')}"
            )

        return final_ranked

    except Exception as e:
        print(f"❌ Error in evaluate_candidates_bulk: {e}")
        traceback.print_exc()
        return []


# ------------------------------------------------------------------
# 10. SIMPLE SINGLE-CALL EVALUATION (original run_evaluation)
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
# 11. MAIN
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

    # ── B) Full bulk pipeline: token-aware batching + tournament ranking ───────
    # Uncomment to run on many resumes:
    #
    # sample_resumes = [c["resume_text"] for c in _sample_candidates]
    # ranked = evaluate_candidates_bulk(
    #     candidate_resumes         = sample_resumes,
    #     job_description           = job_description_text,
    #     required_skills           = ["Python", "Django", "FastAPI", "PostgreSQL", "AWS"],
    #     tournament_top_pct        = 0.20,  # top 20% go to tournament
    #     tournament_min_finalists  = 5,     # always at least 5 in tournament
    #     tournament_max_finalists  = 20     # cap to control cost
    # )
    # print("\n=== FULL RANKED OUTPUT ===")
    # print(json.dumps(ranked, indent=2))