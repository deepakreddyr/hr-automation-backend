import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import openai
from dateutil import parser
from datetime import datetime, timedelta
import re
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is not set. Please configure it in your environment.")

client = OpenAI(api_key=api_key)

# --- Function to get JD embedding ---
def get_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
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

Return a valid JSON array containing objects for each resume. Do not include any explanatory text, warnings, or content outside the JSON array.

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

- **If match_score ≤ 70 for a resume**:
  {{
    "match_score": [value],
    "reason_to_reject": "[Reason]"
  }}

Example output structure:
[
  {{
    "match_score": 85,
    "name": "John Doe",
    "phone": "+91-9876543210",
    "email": "john.doe@example.com",
    "job_summary": "Experienced data scientist with 3 years in ML",
    "experience_in_skills": {{
      "Python": "3 years 0 months",
      "Machine Learning": "2 years 6 months"
    }},
    "total_experience": "3 years 2 months",
    "relevant_experience": "2 years 8 months"
  }},
  {{
    "match_score": 65,
    "reason_to_reject": "Lacks required Python experience"
  }}
]

Respond with ONLY the JSON array, no additional text.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"{system_prompt}\n\nCandidate Resumes:\n{data}",
            temperature=0.3,  # Reduced for more consistent output
            max_output_tokens=4096,  # Increased for larger batches
            top_p=1,
        )

        raw_output = response.output[0].content[0].text.strip()
        print(raw_output)
        if not raw_output:
            print("❌ ERROR: Empty model output")
            return []

        # Clean the output to extract JSON
        cleaned_output = clean_ai_response(raw_output)
        
        if not cleaned_output:
            print("❌ ERROR: Could not extract valid JSON from response")
            print("🔎 Raw output:", raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
            return []

        try:
            parsed_output = json.loads(cleaned_output)
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON Parse Error: {json_error}")
            print("🔎 Cleaned output:", cleaned_output[:300] + "..." if len(cleaned_output) > 300 else cleaned_output)
            return []

        # Ensure output is a list
        if isinstance(parsed_output, dict):
            parsed_output = [parsed_output]

        if not isinstance(parsed_output, list):
            print("❌ ERROR: Parsed output is not a list")
            return []

        # Validate each candidate object
        valid_candidates = []
        for candidate in parsed_output:
            if validate_candidate_structure(candidate):
                valid_candidates.append(candidate)
            else:
                print(f"⚠️ WARNING: Invalid candidate structure: {candidate}")

        print(f"✅ Successfully parsed {len(valid_candidates)} valid candidates")
        return valid_candidates

    except Exception as e:
        print(f"❌ ERROR calling AI model: {e}")
        import traceback
        traceback.print_exc()
        return []


def clean_ai_response(raw_output):
    """
    Extract valid JSON from AI response, handling various formatting issues
    """
    try:
        # Remove any leading/trailing whitespace
        cleaned = raw_output.strip()
        
        # Remove common prefixes that might interfere with JSON parsing
        prefixes_to_remove = [
            "⚠️ WARNING: Output is not valid JSON",
            "🔎 Raw output:",
            "Here's the JSON output:",
            "```json",
            "```"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove trailing markdown or extra content
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        
        # Find JSON array bounds
        start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = cleaned[start_idx:end_idx + 1]
        else:
            # Try to find a single JSON object and wrap it in array
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = '[' + cleaned[start_idx:end_idx + 1] + ']'
            else:
                return None
        
        # Basic JSON validation - try to parse
        json.loads(json_content)
        return json_content
        
    except Exception as e:
        print(f"❌ Error cleaning AI response: {e}")
        return None


def validate_candidate_structure(candidate):
    """
    Validate that candidate object has required structure
    """
    if not isinstance(candidate, dict):
        return False
    
    # Check for required match_score
    if 'match_score' not in candidate:
        return False
    
    try:
        match_score = float(candidate['match_score'])
        if not (0 <= match_score <= 100):
            return False
    except (ValueError, TypeError):
        return False
    
    # For high-scoring candidates, check required fields
    if match_score > 70:
        required_fields = ['name', 'email']
        for field in required_fields:
            if field not in candidate or not candidate[field]:
                return False
    else:
        # For rejected candidates, check for rejection reason
        if 'reason_to_reject' not in candidate or not candidate['reason_to_reject']:
            return False
    
    return True

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


res = []
prof = []

def shortlist_candidates(candidates, required_skills, noc, jd_text):
    prompt = f"""
        You are an expert technical recruiter.  

        ### Task
        From the scraped candidate data, identify the **top {noc} candidates** that STRICTLY MATCH THE GIVEN KEY SKILLS and BEST MATCH THE GIVEN JOB DESCRIPTION.  
        Each candidate entry is a list containing their name, experience, location, current role, education, key skills, and other details.  

        ### Key Skills
        {required_skills}

        ### Job Description
        {jd_text}

        ### Scraped Candidate Data
        {candidates}

        ### Output Format
        Return the result strictly as a **raw JSON array** of objects (no markdown, no code fences), each containing:
        - "index": <0-based index of candidate in scraped_data>
        - "name": <candidate's name>
        - "email": <email if found, otherwise "N/A">

        ### Rules
        - Only return up to {noc} best matches.
        - If fewer than {noc} good matches exist, return fewer.
        - The output must be valid JSON, directly parsable with `json.loads()`.
        - Do not include markdown, code fences, or any explanation outside the JSON.
    """

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.5,
            max_output_tokens=2048,
            top_p=1,
        )

        raw_output = response.output[0].content[0].text.strip()

        if not raw_output:
            print("❌ ERROR: Empty model output")
            return []

        # 🚀 Strip markdown code fences if present
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            if raw_output.lower().startswith("json"):
                raw_output = raw_output[4:].strip()

        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            print("⚠️ WARNING: Output is not valid JSON")
            print("🔎 Raw output:", raw_output)
            print("Error:", e)
            return []

        # Normalize output
        if isinstance(parsed_output, dict):
            parsed_output = [parsed_output]

        if not isinstance(parsed_output, list):
            print("❌ ERROR: Parsed output is not a list")
            return []

        print("✅ Parsed JSON:", parsed_output)
        return parsed_output

    except Exception as e:
        print("❌ ERROR generating candidates:", e)
        return []

def scrape(data):
    global res, prof
    res = []  # Reset global list

    for i in data.split("\n"):
        res.append(i.strip())

    sliced_data = res[46:len(res)-34]

    final = []
    temp = []

    for i in sliced_data:
        if "active" in i.lower():
            final.append(temp)
            temp = []
        elif i in ["View phone number", "Call candidate", "Verified phone & email", "\n", ""]:
            continue
        else:
            temp.append(i)

    return final

def parse_reschedule_time(time_string):
    """
    Convert natural language time strings to timestamp
    Try regex first (fast), fallback to OpenAI for complex cases
    """
    if not time_string or time_string.strip() == "":
        return None
    
    try:
        time_string_lower = time_string.lower().strip()
        now = datetime.now()
        
        # List of patterns we can handle with regex
        simple_patterns = [
            r'in\s+',           # "in X min/hr"
            r'after\s+',        # "after X hours"
            r'o\'?clock',       # "2 o'clock"
            r'tomorrow',        # "tomorrow"
            r'next day',        # "next day"
            r'\d+\s*:\s*\d+',   # "5:30"
            r'\d+\s*(am|pm)',   # "3pm", "10am"
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday',  # day names
        ]
        
        # Check if it's a simple pattern we can handle
        is_simple = any(re.search(pattern, time_string_lower) for pattern in simple_patterns)
        
        if is_simple:
            # Use regex logic
            result = parse_with_regex(time_string_lower, now)
            if result:
                return result
            else:
                # Regex failed, try OpenAI
                print(f"Regex parsing failed for '{time_string}', trying OpenAI")
                return parse_with_openai(time_string)
        else:
            # Complex/ambiguous case - use OpenAI
            print(f"Complex time expression detected: '{time_string}', using OpenAI")
            return parse_with_openai(time_string)
            
    except Exception as e:
        print(f"Error parsing time '{time_string}': {e}")
        # Default fallback
        default_time = datetime.now() + timedelta(days=1)
        default_time = default_time.replace(hour=10, minute=0, second=0, microsecond=0)
        return default_time.isoformat()


def parse_with_regex(time_string, now):
    """Parse time using regex patterns"""
    try:
        # Handle relative time formats (e.g., "in 2 hr", "in 5 min", "in another 30 minutes")
        # Remove "another" to normalize the string
        normalized_time = time_string.replace("another", "").strip()
        
        if "in " in normalized_time or time_string.startswith("another"):
            # Match patterns like "in 2 hr", "in 5 minutes", "in 1 hour", "another 5 min"
            hours_match = re.search(r'(\d+)\s*(hr|hour|hours)', normalized_time)
            minutes_match = re.search(r'(\d+)\s*(min|minute|minutes)', normalized_time)
            
            delta = timedelta()
            
            if hours_match:
                hours = int(hours_match.group(1))
                delta += timedelta(hours=hours)
            
            if minutes_match:
                minutes = int(minutes_match.group(1))
                delta += timedelta(minutes=minutes)
            
            # If we found either hours or minutes
            if hours_match or minutes_match:
                scheduled_time = now + delta
                return scheduled_time.isoformat()
        
        # Handle "after X hours/minutes" format (including "after another X")
        after_match = re.search(r'after\s+(?:another\s+)?(\d+)\s*(hr|hour|hours|min|minute|minutes)', time_string)
        if after_match:
            value = int(after_match.group(1))
            unit = after_match.group(2)
            
            if 'hr' in unit or 'hour' in unit:
                scheduled_time = now + timedelta(hours=value)
            else:  # minutes
                scheduled_time = now + timedelta(minutes=value)
            
            return scheduled_time.isoformat()
        
        # Handle "o'clock" format (e.g., "2 o'clock")
        if "o'clock" in time_string:
            hour_match = re.search(r'(\d+)\s*o\'?clock', time_string)
            if hour_match:
                hour = int(hour_match.group(1))
                # Assume PM if hour is between 1-7, AM if 8-12
                if 1 <= hour <= 7:
                    hour += 12
                elif hour == 12:
                    hour = 12
                
                scheduled_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # If the time has passed today, schedule for tomorrow
                if scheduled_time <= now:
                    scheduled_time += timedelta(days=1)
                
                return scheduled_time.isoformat()
        
        # Handle "tomorrow" keyword
        if "tomorrow" in time_string:
            time_part = time_string.replace("tomorrow", "").replace("at", "").strip()
            base_date = now + timedelta(days=1)
            
            if time_part:
                # Try to parse the time part
                try:
                    parsed_time = parser.parse(time_part, fuzzy=True)
                    scheduled_time = base_date.replace(
                        hour=parsed_time.hour,
                        minute=parsed_time.minute,
                        second=0,
                        microsecond=0
                    )
                    return scheduled_time.isoformat()
                except:
                    # Default to 10 AM tomorrow if parsing fails
                    scheduled_time = base_date.replace(hour=10, minute=0, second=0, microsecond=0)
                    return scheduled_time.isoformat()
        
        # Handle "next day" or "the next day"
        if "next day" in time_string or "the next day" in time_string:
            time_part = re.sub(r'(next day|the next day)', '', time_string).replace("at", "").strip()
            base_date = now + timedelta(days=1)
            
            if time_part:
                try:
                    parsed_time = parser.parse(time_part, fuzzy=True)
                    scheduled_time = base_date.replace(
                        hour=parsed_time.hour,
                        minute=parsed_time.minute,
                        second=0,
                        microsecond=0
                    )
                    return scheduled_time.isoformat()
                except:
                    scheduled_time = base_date.replace(hour=10, minute=0, second=0, microsecond=0)
                    return scheduled_time.isoformat()
        
        # Handle day of week (e.g., "Monday at 3pm", "next Friday")
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in weekdays:
            if day in time_string:
                target_day = weekdays.index(day)
                current_day = now.weekday()
                days_ahead = target_day - current_day
                
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                
                base_date = now + timedelta(days=days_ahead)
                time_part = time_string.replace(day, "").replace("next", "").replace("at", "").strip()
                
                if time_part:
                    try:
                        parsed_time = parser.parse(time_part, fuzzy=True)
                        scheduled_time = base_date.replace(
                            hour=parsed_time.hour,
                            minute=parsed_time.minute,
                            second=0,
                            microsecond=0
                        )
                        return scheduled_time.isoformat()
                    except:
                        pass
                
                # Default to 10 AM if no time specified
                scheduled_time = base_date.replace(hour=10, minute=0, second=0, microsecond=0)
                return scheduled_time.isoformat()
        
        # Try general parsing with dateutil (handles "3pm", "5:30pm", etc.)
        parsed = parser.parse(time_string, fuzzy=True)
        
        # If only time was provided (no date), use today or tomorrow
        if parsed.date() == datetime.now().date():
            scheduled_time = now.replace(
                hour=parsed.hour,
                minute=parsed.minute,
                second=0,
                microsecond=0
            )
            
            # If time has passed, schedule for tomorrow
            if scheduled_time <= now:
                scheduled_time += timedelta(days=1)
        else:
            scheduled_time = parsed
        
        return scheduled_time.isoformat()
        
    except Exception as e:
        print(f"Regex parsing error: {e}")
        return None


def parse_with_openai(time_string):
    """Fallback for complex expressions using OpenAI"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        current_time = datetime.now()
        prompt = f"""Convert this time expression to an ISO 8601 timestamp.

Current date and time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ({current_time.strftime('%A')})
Time zone: Assume local time (no timezone conversion needed)
User said: "{time_string}"

Rules:
- Return ONLY the ISO timestamp in format: YYYY-MM-DDTHH:MM:SS
- If the time is ambiguous (like "evening"), use reasonable defaults (evening = 6 PM)
- If no date is mentioned and the time has passed today, assume tomorrow
- Do not include any explanation, just the timestamp

Example outputs:
- "2025-11-06T14:30:00"
- "2025-11-10T18:00:00"
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper and faster
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50
        )
        
        timestamp_str = response.choices[0].message.content.strip()
        
        # Validate it's a proper timestamp
        parsed_dt = datetime.fromisoformat(timestamp_str)
        
        # Additional validation: make sure it's not in the past (unless very recent)
        if parsed_dt < datetime.now() - timedelta(minutes=5):
            print(f"Warning: OpenAI returned past timestamp: {timestamp_str}")
            # Use tomorrow at same time instead
            parsed_dt = parsed_dt + timedelta(days=1)
            return parsed_dt.isoformat()
        
        return timestamp_str
        
    except Exception as e:
        print(f"OpenAI parsing failed: {e}")
        # Return default: tomorrow at 10 AM
        default_time = datetime.now() + timedelta(days=1)
        default_time = default_time.replace(hour=10, minute=0, second=0, microsecond=0)
        return default_time.isoformat()
