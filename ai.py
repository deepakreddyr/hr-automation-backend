import os
import json
from openai import OpenAI
from dotenv import load_dotenv

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


{
  "email": "sandy17santhosh@outlook.com",
  "phone": "+91 9894414277",
  "strengths": [
    "Strong experience in API and network integration",
    "Proficient in Swift and SwiftUI"
  ],
  "weaknesses": [
    "Limited experience in Objective-C",
    "No formal education in computer science"
  ],
  "agent_checks": {
    "agent_1_summary": "Santhosh has 6 years of experience with 2 years in iOS development using Swift and related technologies.",
    "agent_2_skills_gap": {
      "strong_matches": [
        "Swift",
        "SwiftUI",
        "URLSession",
        "Alamofire",
        "Google Maps",
        "Remote Config"
      ],
      "missing_critical": []
    },
    "agent_3_project_depth": "Medium assessment. Santhosh has worked on various iOS projects with a focus on API integration and third-party tools.",
    "agent_4_recommendation": {
      "reasoning": "Santhosh has strong technical skills in iOS development and relevant experience, making him a good fit for the role.",
      "recommendation": "Interview"
    }
  },
  "candidate_id": "cand_1",
  "scoring_tier": "Good",
  "candidate_name": "Santhosh V",
  "special_highlights": [
    "Led deployment of apps to App Store"
  ],
  "overall_match_score": 82,
  "experience_in_skills": {
    "Git": "3 years 0 months",
    "Swift": "2 years 0 months",
    "UIKit": "1 year 0 months",
    "Combine": "1 year 0 months",
    "SwiftUI": "2 years 0 months",
    "Alamofire": "2 years 0 months",
    "Core Data": "1 year 0 months",
    "URLSession": "2 years 0 months",
    "Google Maps": "2 years 0 months",
    "Objective-C": "1 year 0 months",
    "Remote Config": "2 years 0 months",
    "Core Animation": "1 year 0 months"
  },
  "relevant_experience_years": "2 years 0 months",
  "total_work_experience_years": "6 years 0 months"
}

# ------------------------------------------------------------------
# 3. INPUT DATA: The Variable Fields (JD + Candidates)
# ------------------------------------------------------------------
# In a real app, this data would come from your database or file uploads.

job_description_text = """
We are looking for a Senior Backend Engineer to join our FinTech team.
Must Haves: 5+ years Python, Experience with Django/FastAPI, PostgreSQL scaling, and AWS (Lambda, EC2).
Nice to Haves: Experience with Kubernetes, Blockchain knowledge, and previous mentorship experience.
"""

candidates_data = [
    {
        "id": "cand_101",
        "name": "Alex Johnson",
        "resume_text": "Alex Johnson. 6 years of experience in Python. Expert in Flask and Django. Built scalable payment systems using PostgreSQL. Certified AWS Solutions Architect. I have never used Kubernetes."
    },
    {
        "id": "cand_102",
        "name": "Sam Smith",
        "resume_text": "Sam Smith. Junior Developer. 1 year experience in Java and C++. Passionate about crypto. Fast learner. Willing to learn Python."
    }
]

# Construct the final User Message
user_message_payload = {
    "job_description": job_description_text,
    "candidates": candidates_data
}

# ------------------------------------------------------------------
# 4. EXECUTION: Running the Model
# ------------------------------------------------------------------
def run_evaluation():
    try:
        print("Sending request to OpenAI...")
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Recommended for complex logic/JSON
            response_format={"type": "json_object"}, # CRITICAL: Enforces JSON output
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_message_payload)}
            ],
            temperature=0.1 # Low temp for consistent, analytical results
        )

        # Parse the JSON string from the response
        result_json = json.loads(response.choices[0].message.content)
        
        return result_json

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# ------------------------------------------------------------------
# 5. BULK EVALUATION FUNCTION FOR MAIN.PY
# ------------------------------------------------------------------
def evaluate_candidates_bulk(candidate_resumes, job_description, required_skills):
    """
    Evaluate multiple candidates against a job description.
    
    Args:
        candidate_resumes: List of resume text strings or single concatenated string
        job_description: Job description text
        required_skills: Comma-separated string or list of required skills
    
    Returns:
        List of candidate evaluation dictionaries matching the bulk_evaluations structure
    """
    try:
        # Handle both list and concatenated string inputs
        if isinstance(candidate_resumes, list):
            candidates_data = [
                {
                    "id": f"cand_{i+1}",
                    "resume_text": resume
                }
                for i, resume in enumerate(candidate_resumes)
            ]
        else:
            # Split concatenated resumes
            resume_texts = candidate_resumes.split("\n\n--- RESUME SEPARATOR ---\n\n")
            candidates_data = [
                {
                    "id": f"cand_{i+1}",
                    "resume_text": resume
                }
                for i, resume in enumerate(resume_texts) if resume.strip()
            ]
        
        # Construct the user message payload
        user_message_payload = {
            "job_description": job_description,
            "required_skills": required_skills,
            "candidates": candidates_data
        }
        
        print(f"Evaluating {len(candidates_data)} candidates...")
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o for complex logic/JSON
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_message_payload)}
            ],
            temperature=0.1
        )
        
        # Parse the JSON response
        result_json = json.loads(response.choices[0].message.content)
        
        # Return the bulk_evaluations array
        if "bulk_evaluations" in result_json:
            print(f"✅ Successfully evaluated {len(result_json['bulk_evaluations'])} candidates")
            return result_json["bulk_evaluations"]
        else:
            print("❌ ERROR: Response missing 'bulk_evaluations' key")
            return []
            
    except Exception as e:
        print(f"❌ Error in evaluate_candidates_bulk: {e}")
        import traceback
        traceback.print_exc()
        return []

# ------------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    results = run_evaluation()
    
    if results:
        print("\n=== EVALUATION RESULTS ===")
        print(json.dumps(results, indent=2))
        
        # Example: Loop through results to show how to use them
        print("\n=== SUMMARY ===")
        for eval in results['bulk_evaluations']:
            print(f"Candidate: {eval['candidate_name']} | Score: {eval['overall_match_score']} ({eval['scoring_tier']})")