import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is not set. Please configure it in your environment.")

client = OpenAI(api_key=api_key)


# ----------------------------
# Candidate Evaluation
# ----------------------------
def get_candidate_details(data, jd1, skills):
    system_prompt = f"""
    Evaluate a batch of candidate resumes against a given job description (JD) to identify and shortlist qualified candidates. For each candidate, calculate a match score from 0 to 100 based on relevance to the JD. Only candidates with a score above 70 should be shortlisted. Provide output in structured JSON format for each candidate, with detailed fields for shortlisted ones and a rejection reason for others.

- Match Score Calculation: Analyze each candidate's resume against the JD to calculate a match score in the range of 0 to 100.
- Data Extraction: For candidates with a match score greater than 70, extract key fields from their resumes.
- Conditional Output: Vary the structure of the JSON output based on the match score for each resume.

# Steps

1. **Calculate Match Score**: 
   - Compare the candidates' resumes with the {jd1} to determine a match score within 0-100 for each resume.
   
2. **Conditional Logic**:
   - If a match score is greater than 70, proceed to extract detailed information from the resume.
   - If a match score is 70 or below, skip data extraction and generate a rejection reason.

3. **Data Extraction** (for match scores > 70):
   - Extract and include the following fields in the JSON output for each candidate with a high score:
     - `match_score`
     - `name`
     - `phone`
     - `email`
     - `job_summary`
     - `experience in each skill` as mentioned in {skills}
     - `total_experience` in years and months
     - `relevant_experience` in specified skills in years and months

4. **Rejection Cases** (for match scores ≤ 70):
   - Output a JSON containing:
     - `match_score`
     - `reason_to_reject`

5. **The number of candidates resume results should strictly be equal to the number of candidates resume input(IMPORTANT).**
# Output Format

Produce the output in JSON format for each resume:
- **If match_score > 70 for a resume**:
  {{
    "match_score": [value],
    "name": "[Name]",
    "phone": "[Phone Number]",
    "email": "[Email]",
    "job_summary": "[Summary]",
    "experience_in_skills": {{
      "[Skill1]": "[Duration]",
      "[Skill2]": "[Duration]"
    }},
    "total_experience": "[Years and Months]",
    "relevant_experience": "[Years and Months]"
  }}
"""
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"{system_prompt}\n\nCandidate Resumes:\n{data}",
            temperature=0.5,
            max_output_tokens=2048,
            top_p=1,
        )

        raw_output = response.output[0].content[0].text.strip()

        if not raw_output:
            print("❌ ERROR: Empty model output")
            return []

        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            print("⚠️ WARNING: Output is not valid JSON")
            print("🔎 Raw output:", raw_output)
            return []

        if isinstance(parsed_output, dict):
            parsed_output = [parsed_output]

        if not isinstance(parsed_output, list):
            print("❌ ERROR: Parsed output is not a list")
            return []

        print("✅ Parsed JSON:", parsed_output)
        return parsed_output

    except Exception as e:
        print("❌ ERROR calling OpenAI:", e)
        return []


# ----------------------------
# HR Interview Questions
# ----------------------------
def get_questions(jd):
    prompt = f"""Generate HR-level interview questions from a provided job description to confirm a candidate's experience and qualifications. Focus on evaluating demonstrated work related to the key skills and experiences mentioned within the job description.\n\n[Frame questions based on the provided job description template.]\n\n# Steps\n\n1. Analyze the provided job description to extract key skills, experience, and attributes required for the role.\n2. Formulate questions aimed at confirming the candidate's direct experience and work related to these specific criteria from the job description.\n3. Ensure questions are specific to the job description details, yet open enough to evoke detailed responses.\n\n# Input\n\n- {jd}: A structured job description from which key skills, experiences, and qualifications will be identified.\n\n# Output Format\n\nThe output should be a structured list of HR-level interview questions.\n\n- Example Questions:\n  - \"Can you detail your experience working with {{key_skill_1}} as described in the job description?\"\n  - \"The job description mentions experience in {{key_area}}. Can you share a specific project in which you utilized these skills?\"\n  - \"Can you provide examples of how you have demonstrated {{required_attribute}} in previous roles?\"\n\n# Notes\n\n- Tailor questions to verify the candidate's past work and experience directly related to the job description provided.\n- The aim is to assess the candidate's specific capabilities aligning with the job requirements described. (Do not generate more than 3 questions and also make sure that the questions are shorter which will also have shorter answers)"""

    try:
        response = client.responses.create(
            model="gpt-4.1-nano",
            input=prompt,
            temperature=1,
            max_output_tokens=512,
            top_p=1,
        )

        raw_output = response.output[0].content[0].text.strip()
        print("✅ HR Questions:", raw_output)
        return raw_output

    except Exception as e:
        print("❌ ERROR generating questions:", e)
        return []
