import os
from dotenv import load_dotenv
from functools import wraps
import json
from flask import (
    Flask, render_template, request,
    redirect, url_for, session, jsonify, flash, make_response
)
from dashboard_data import (
    get_todays_searches,
    get_todays_candidates,
    get_new_joinees,
    get_creds_used,
    get_user_name,
    get_people_called,
    get_weekly_activity
)
import fitz  
from flask_cors import CORS
import requests
from aiparser import shortlist_candidates,scrape,get_candidate_details,get_questions
from supabase import create_client, Client
import threading
import ast
import secrets

# ─── App & Config ─────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", template_folder="templates")
load_dotenv()

# Environment detection
IS_PRODUCTION = os.getenv("FLASK_ENV") == "production" or os.getenv("ENVIRONMENT") == "production"

# ─── Unified CORS Configuration (Works for Both Localhost & Vercel) ─────────────

# Unified Origins - Works for both scenarios
ALLOWED_ORIGINS = [
    # Localhost development
    "http://localhost:3000",
    "http://localhost:8080", 
    "http://localhost:5173",    # Vite default
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
    
    # Production deployments
    "https://hr-frontend-one.vercel.app",
    "https://hr-frontend-x9j2.onrender.com",
    
    # Custom frontend URL from environment
    os.getenv("FRONTEND_URL", ""),
]

PRODUCTION_ORIGINS = [
    os.getenv("FRONTEND_URL", "https://yourdomain.com"),
    "https://hr-frontend-one.vercel.app",
    "https://hr-frontend-x9j2.onrender.com",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

# Debug output
print("=== UNIFIED CORS CONFIGURATION ===")
print(f"Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}")
print(f"Allowed origins: {ALLOWED_ORIGINS}")
print("==================================")

# Unified CORS Configuration
CORS(
    app,
    supports_credentials=True,
    origins=ALLOWED_ORIGINS,
    allow_headers=[
        "Content-Type", 
        "Authorization", 
        "X-Requested-With",
        "Accept",
        "Origin"
    ],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    expose_headers=["Set-Cookie"],
    max_age=86400  # Cache preflight requests for 24 hours
)

# Secure secret key
app.secret_key = os.getenv("SECRET_KEY", "xJ7vK9mQ2nR8pL6wE4tY1uI0oP3aS5dF7gH9jK2lM6nB8vC1xZ4qW7eR3tY6uI9o")

# Unified Session Configuration - Works for both localhost and Vercel
app.config.update(
    SESSION_COOKIE_NAME="session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,   # Must be HTTPS in production
    SESSION_COOKIE_SAMESITE="Lax",  # Required for cross-site cookies
    PERMANENT_SESSION_LIFETIME=86400,
    SESSION_COOKIE_DOMAIN=None,             # Don't restrict domain
    SESSION_COOKIE_PATH="/",                # Available for all paths
)

print(f"Session config: SameSite=None, Secure=False, HttpOnly=False, Domain=None")

# ─── Environment Variables ─────────────────────────────────────────────────────

VAPI_URL               = os.getenv("VAPI_URL")
VAPI_API_KEY           = os.getenv("VAPI_API_KEY")
AGENT_ID               = os.getenv("AGENT_ID")
VAPI_PHONE_NUMBER_ID   = os.getenv("VAPI_PHONE_NUMBER_ID")

url: str = os.getenv("SUPABASE_URL")
anon_key: str = os.getenv("SUPABASE_KEY")
service_key: str = os.getenv("SERVICE_ROLE_KEY")

# Validate required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Required environment variable {var} is not set")

supabase: Client = create_client(url, anon_key)

# ─── Helper Functions ─────────────────────────────────────────────────────────

def debug_session():
    """Helper function to debug session state"""
    print(f"Session data: {dict(session)}")
    print(f"Session keys: {list(session.keys())}")
    print(f"Has user: {'user' in session}")
    print(f"Has user_id: {'user_id' in session}")
    print(f"Request origin: {request.headers.get('Origin')}")
    print(f"Request cookies: {dict(request.cookies)}")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({"status": "healthy", "environment": "production" if IS_PRODUCTION else "development"})

@app.route("/login", methods=["GET","POST"])
def login():
    try:
        # Handle both form data and JSON data
        if request.is_json:
            data = request.get_json()
            email = data.get("email", "").lower()
            password = data.get("password", "")
        else:
            email = request.form.get("email", "").lower()
            password = request.form.get("password", "")

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password required"}), 400

        # Authenticate with Supabase
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        # Check if authentication failed
        if not auth_response or not auth_response.user:
            return jsonify({"success": False, "message": "Invalid email or password"}), 401

        # Fetch user record from DB
        user_data = supabase.table("users").select("id, email").eq("email", email).execute()
        if not user_data.data:
            return jsonify({"success": False, "message": "User not found in database"}), 404

        # Store session data
        session.permanent = True
        session["user"] = user_data.data[0]["email"]
        session["user_id"] = user_data.data[0]["id"]
        
        print("After login - Session set:")
        debug_session()
        
        response = make_response(jsonify({"success": True, "message": "Login successful"}))
        return response

    except Exception as e:
        # Log error
        print(f"Login error: {str(e)}")
        return jsonify({"success": False, "message": "Login failed"}), 500

# Add a session check endpoint for debugging
@app.route("/api/session-check", methods=["GET"])
def session_check():
    debug_session()
    return jsonify({
        "has_session": "user" in session and "user_id" in session,
        "user": session.get("user"),
        "user_id": session.get("user_id"),
        "session_keys": list(session.keys()),
        "request_origin": request.headers.get('Origin'),
        "cookies_received": dict(request.cookies)
    })

@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})
# ------------------------------
# Password reset
# ------------------------------

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        email = request.form.get("email", "").lower()
        try:
            supabase.auth.reset_password_for_email(email)
            flash("✅ Password reset email sent. Check your inbox.", "success")
        except Exception as e:
            flash(f"❌ Error: {str(e)}", "danger")
    return render_template("reset_password.html")

TRANSCRIPT={}

@app.route("/404")
def error():
    return render_template("404.html")

# ─── Utility: Login Required Decorator ────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route('/api/search-status/<int:search_id>')
@login_required
def get_search_status(search_id):
    response = supabase.table("search").select("status").eq("id", search_id).single().execute()
    if response.data:
        return jsonify({"status": response.data.get("status")})
    else:
        return jsonify({"error": "Search not found"}), 404


@app.route("/api/create-search", methods=["POST"])
@login_required
def create_search():
    response = supabase.table("search").insert({
        "user_id": session["user_id"],
        "processed": False,
        "remote_work": False,
        "contract_hiring": False,
        "key_skills": "",
        "job_role": "",
        "raw_data": "",
        "shortlisted_index": "[]",
        "noc": 0,
        "job_description": "",
        "status":"shortlist"
    }).execute()

    history_resp=supabase.table("history").insert({
        "user_id": session["user_id"],
        "creds": 0
        }).execute()  
    if not history_resp.data:
            print("❌ Failed to insert history")
            return "Error inserting history", 500

    history_id = history_resp.data[0]["id"]
    session["history_id"] = history_id

    if response.data:
        return jsonify({"search_id": response.data[0]["id"]})
    else:
        return jsonify({"error": "Could not create search"}), 500

@app.route("/api/shortlist/<int:search_id>", methods=["GET", "POST"])
@login_required
def shortlist(search_id):
    if request.method == "POST":
        candidate_data = request.form.get("candidateData", "").strip()
        skills = request.form.get("skills", "").strip()
        job_role = request.form.get("jobRole", "").strip()
        jd_file = request.files.get("jdFile")
        noc = int(request.form.get("numCandidates", "0"))
        search_name = request.form.get("searchName","").strip()
        errors = []
        if not candidate_data:
            errors.append("Candidate data is required.")
        if not skills:
            errors.append("Required skills cannot be empty.")
        if not job_role:
            errors.append("Job role is required.")
        if noc <= 0:
            errors.append("Number of candidates must be greater than 0.")

        if errors:
            return jsonify(success=False, errors=errors), 400

        # ✅ Extract JD text from uploaded PDF
        jd_text = ""
        if jd_file and jd_file.filename.endswith(".pdf"):
            try:
                doc = fitz.open(stream=jd_file.read(), filetype="pdf")
                jd_text = "\n".join([page.get_text() for page in doc])
            except Exception as e:
                return jsonify(success=False, error=f"Error reading JD PDF: {e}"), 500

        # ✅ Extract, shortlist, update
        final_candidates = scrape(candidate_data)
        shortlisted_indices = shortlist_candidates(final_candidates, skills)

        # ✅ Save to session
        session["shortlisted_indices"] = shortlisted_indices
        session["shortlisted_count"] = len(shortlisted_indices)
        

        # ✅ Update the search entry
        supabase.table("search").update({
            "raw_data": candidate_data,
            "key_skills": skills,
            "job_role": job_role,
            "user_id": session["user_id"],
            "shortlisted_index": json.dumps(shortlisted_indices),
            "processed": True,
            "remote_work": False,
            "contract_hiring": False,
            "noc": noc,
            "job_description": jd_text,
            "history_id":session["history_id"],
            "search_name": search_name,
            "status":"process"
        }).eq("id", search_id).execute()
        deduct_credits(session["user_id"], "search", reference_id=None)
        return jsonify(success=True)

@app.route("/api/process/<int:search_id>", methods=["GET", "POST"])
@login_required
def process(search_id):
    # print("\n==== Entered /api/process/<search_id> route ====")
    # print("Session BEFORE reset:", dict(session))

    # === Reset session only if search_id changed ===
    if session.get("search_id") != search_id:
        print("Resetting session for new search_id")
        session["search_id"] = search_id
        session["collected_resumes"] = []
        session["resume_dict"] = {}
        session["right_fields"] = None
        session["sl_count"]=0

        try:
            result = supabase.table("search").select("shortlisted_index").eq("id", search_id).single().execute()
            shortlisted = result.data.get("shortlisted_index", [])
            if isinstance(shortlisted, str):
                try:
                    shortlisted = ast.literal_eval(shortlisted)
                except Exception:
                    shortlisted = []
            session["shortlisted_indices"] = shortlisted
            session["shortlisted_count"] = len(shortlisted)
            # print("Loaded shortlisted_indices:", shortlisted)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # Ensure keys exist
    session.setdefault("collected_resumes", [])
    session.setdefault("resume_dict", {})
    session.setdefault("right_fields", None)
    session.setdefault("shortlisted_indices", [])
    session.setdefault("shortlisted_count", 0)

    shortlisted_indices = session["shortlisted_indices"]
    target = session["shortlisted_count"]
    submitted = len(session["collected_resumes"])
    candidate_index = shortlisted_indices[submitted] if submitted < target else None
    is_last = (submitted == target - 1)

    # === POST (Form Submission) ===
    if request.method == "POST":
        resume_text = request.form.get("resumeText", "").strip()
        session["sl_count"]+=1
        hiring_company = request.form.get("hiringCompany", "").strip()
        company_location = request.form.get("companyLocation", "").strip()
        hr_company = request.form.get("hrCompany", "").strip()
        notice_period = request.form.get("noticePeriod", "").strip()
        remote_work = request.form.get("remoteWork") == "on"
        contract_h = request.form.get("contractH") == "on"
        custom_question = request.form.get("customQuestion", "").strip()

        errors = []
        if not resume_text:
            errors.append("Resume text is required.")
        if not session["right_fields"]:
            if not hiring_company:
                errors.append("Hiring Company is required.")
            if not company_location:
                errors.append("Company Location is required.")
            if not hr_company:
                errors.append("HR Company is required.")
            if not notice_period:
                errors.append("Notice Period is required.")

        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        # Handle optional CSV file
        csv_file = request.files.get("csvFile")
        if csv_file:
            print("Received CSV file:", csv_file.filename)
            # Optional CSV parsing logic here

        # === Store resume data ===
        session["resume_dict"][f"candidate_{candidate_index}"] = resume_text
        session["collected_resumes"].append({
            "short_idx": candidate_index,
            "resumeText": resume_text
        })
        print(session["resume_dict"])
        # === Store right_fields only once ===
        if not session["right_fields"]:
            session["right_fields"] = {
                "hiringCompany": hiring_company,
                "companyLocation": company_location,
                "hrCompany": hr_company,
                "noticePeriod": notice_period,
                "remoteWork": remote_work,
                "contractH": contract_h,
            }

            try:
                supabase.table("search").update({
                    "rc_name": hiring_company,
                    "company_location": company_location,
                    "hc_name": hr_company,
                    "notice_period": notice_period,
                    "remote_work": remote_work,
                    "contract_hiring": contract_h,
                    "user_id": session["user_id"]
                }).eq("id", session["search_id"]).execute()
            except Exception as e:
                print("Failed to update job details:", str(e))

        # === Save custom question if it's the last submission ===
        submitted = len(session["collected_resumes"])
        if submitted == target and custom_question:
            try:
                supabase.table("search").update({
                    "custom_question": custom_question
                }).eq("id", session["search_id"]).execute()
            except Exception as e:
                print("Failed to save custom question:", str(e))

        # === FINAL RESPONSE ===
        
        print(session["sl_count"],len(shortlisted_indices))
        if session["sl_count"] == len(shortlisted_indices):
            supabase.table("search").update({"status":"results"}).eq("id", session["search_id"]).execute()
            deduct_credits(session["user_id"], "process_candidate", reference_id=None)
            return jsonify({
                "success": True,
                "submitted": submitted,
                "target": target,
                "candidateIndex": candidate_index,
                "isLast": True,
                "right_fields": session["right_fields"],
                "next": False,
                "redirect": "/loading"
            })

        # NEXT candidate
        next_index = shortlisted_indices[submitted]
        return jsonify({
            "success": True,
            "submitted": submitted,
            "target": target,
            "candidateIndex": next_index,
            "isLast": (submitted == target - 1),
            "right_fields": session["right_fields"],
            "next": True
        })

    # === GET (Initial Load) ===
    return jsonify({
        "success": True,
        "submitted": submitted,
        "target": target,
        "candidateIndex": candidate_index,
        "isLast": is_last,
        "shortlisted_indices": shortlisted_indices,
        "right_fields": session["right_fields"] or {},
        "next": True
    })

@app.route("/api/get-questions", methods=["GET"])
def generate_questions():
    search_id = session.get("search_id")
    if not search_id:
        return jsonify({"error": "No active search session"}), 400

    result = supabase.table("search").select("job_description").eq("id", search_id).execute().data
    if not result or not result[0].get("job_description"):
        return jsonify({"error": "No job description found"}), 404

    jd = result[0]["job_description"]
    print(len(jd))
    raw_questions = get_questions(jd)

    if isinstance(raw_questions, str):
        questions = [q.strip("1234567890.- ") for q in raw_questions.strip().split("\n") if q.strip()]
    else:
        questions = raw_questions  # Already a list
    print(questions)
    return jsonify({"questions": questions})


@app.route("/api/loading", methods=["GET"])
@login_required
def loading():
    def process_candidates_async(combined_resumes, jd, skills, search_id, user_id):
        candidate_data = get_candidate_details(combined_resumes, jd, skills)
        
        for candidate in candidate_data:
            if candidate.get("match_score", 0) > 70:
                name = candidate["name"]
                email = candidate["email"]
                phone_raw = candidate.get("phone", "")
                phone_digits = ''.join(filter(str.isdigit, phone_raw))[-10:]
                phone = int(phone_digits) if len(phone_digits) == 10 else None

                supabase.table("candidates").insert({
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "skills": ", ".join(candidate["experience_in_skills"].keys()),
                    "summary": candidate["job_summary"],
                    "skills_experience": candidate["experience_in_skills"],
                    "search_id": search_id,
                    "history_id": search_id,
                    "user_id": user_id,
                    "total_experience": candidate["total_experience"],
                    "relevant_work_experience": candidate["relevant_experience"],
                    "match_score": candidate["match_score"]
                }).execute()

        # ✅ Mark search as processed
        supabase.table("search").update({"processed": True}).eq("id", search_id).execute()

    # Prepare combined resumes
    resume_dict = session.get("resume_dict", {})
    print(len(resume_dict))
    combined_resumes = "\n".join(resume_dict.values())

    # Get JD and skills
    search_id = session["search_id"]
    user_id = session["user_id"]

    jd_resp = supabase.table("search").select("job_description", "key_skills").eq("id", search_id).single().execute()
    jd = jd_resp.data["job_description"]
    skills = jd_resp.data["key_skills"]

    # Start background processing thread
    thread = threading.Thread(
        target=process_candidates_async,
        args=(combined_resumes, jd, skills, search_id, user_id)
    )
    thread.start()

    # ✅ Return a JSON response immediately
    return jsonify({"message": "Processing started"}), 200


@app.route("/api/check-processing", methods=["GET"])
@login_required
def check_processing_status():
    search_id = session.get("search_id")
    if not search_id:
        return jsonify({"error": "Search ID not found"}), 400

    result = supabase.table("search").select("processed").eq("id", search_id).single().execute()
    is_done = result.data.get("processed", False)

    return jsonify({"processed": is_done,"search_id": search_id})

@app.route('/api/results', methods=["GET"])
@login_required
def results():
    if request.args.get('searchID'):
        session["search_id"] = int(request.args.get('searchID'))
        print("Search ID:", session["search_id"])
    
    response = supabase.table("candidates").select("*").eq("search_id", session["search_id"]).execute()
    data = response.data or []

    shortlisted_candidates = []
    calls_scheduled = 0
    rescheduled_calls = 0

    for person in data:
        call_status = person.get("call_status") or "not_called"
        if call_status == "CALLED" or call_status == "Not Answered" or call_status == "Re-schedule":
            calls_scheduled += 1
        if call_status == "Re-schedule":
            rescheduled_calls += 1

        candidate = {
            "id": person.get("id"),  # used as key in React
            "name": person.get("name", "Unknown"),
            "phone": str(person.get("phone", "0000000000")),
            "skills": person.get("skills", ""),
            "email": person.get("email", "noemail@example.com"),
            "summary": person.get("summary"),
            "skills_experience": person.get("skills_experience"),  
            "totalExp": person.get("total_experience"),
            "relevantExp": person.get("relevant_work_experience"),
            "call_status": call_status,
            "match_score": person.get("match_score") or 0,
            "liked": person.get("liked") or False,
            "join_status": person.get("join_status") or False,
            "status": person.get("status") or "pending"
        }
        shortlisted_candidates.append(candidate)
    print(shortlist_candidates)
    return jsonify({
        "candidates": shortlisted_candidates,
        "total": len(shortlisted_candidates),
        "calls_scheduled": calls_scheduled,
        "rescheduled_calls": rescheduled_calls
    })

@app.route("/api/like-candidate", methods=["POST"])
def like_candidate():
    data = request.get_json()
    candidate_id = data.get("candidate_id")
    print(candidate_id)
    liked = data.get("liked", False)

    if not candidate_id:
        return jsonify({"error": "Missing candidate ID"}), 400

    # Update 'liked' column in Supabase
    supabase.table("candidates").update({"liked": liked}).eq("id", candidate_id).execute()
    return jsonify({"success": True}), 200



@app.route("/api/unlike-candidate", methods=["POST"])
def unlike_candidate():
    data = request.get_json()
    candidate_id = data.get("candidate_id")
    liked = False

    if not candidate_id:
        return jsonify({"error": "Missing candidate ID"}), 400

    # Update 'liked' column in Supabase
    supabase.table("candidates").update({"liked": liked}).eq("id", candidate_id).execute()
    return jsonify({"success": True}), 200


@app.route('/add-final-select', methods=['POST'])
def mark_final_selects():
    data = request.json
    candidate_ids = data.get('candidate_ids', [])
    for cid in candidate_ids:
        supabase.table('candidates').update({
            "hiring_status": True
        }).eq('id', cid).execute()
    return jsonify({"message": "Updated successfully"}), 200

@app.route("/api/final-selects", methods=["GET", "POST"])
def final_selects():
    if request.method == "POST":
        data = request.get_json()

        # Toggle joined status
        joined_updates = data.get("joined", [])
        for update in joined_updates:
            candidate_id = update.get("candidate_id")
            new_status = update.get("joined", True)
            supabase.table("candidates").update({"join_status": new_status}).eq("id", candidate_id).execute()

        # Remove candidates from final list
        remove_ids = data.get("remove_from_final", [])
        for candidate_id in remove_ids:
            supabase.table("candidates").update({"hiring_status": False}).eq("id", candidate_id).execute()

        return jsonify({"status": "success"})

    # GET request — return all candidates marked as final selects
    response = supabase.table("candidates").select("*").eq("hiring_status", True).execute()
    data = response.data or []

    final_candidates = []
    for person in data:
        final_candidates.append({
            "id": person.get("id"),
            "name": person.get("name", "Unknown"),
            "phone": str(person.get("phone", "0000000000")),
            "skills": person.get("skills", ""),
            "email": person.get("email", "noemail@example.com"),
            "summary": person.get("summary", ""),
            "totalExp": person.get("total_experience", "0"),
            "relevantExp": person.get("relevant_work_experience", "0"),
            "matchScore": person.get("match_score", 0),
            "status": person.get("call_status") or "pending",
            "joined": person.get("join_status", False),
            "liked": person.get("liked", False),
        })

    return jsonify({"candidates": final_candidates})


@app.route("/api/remove-final-select", methods=["POST"])
def remove_final_select():
    data = request.get_json()
    candidate_id = data.get("candidate_id")

    if not candidate_id:
        return jsonify({"error": "Candidate ID missing"}), 400

    # Update the hiring_status field to False
    supabase.table("candidates").update({"hiring_status": False}).eq("id", candidate_id).execute()

    return jsonify({"status": "success"})

# ─── Call Initiation Logic ───────────────────────────────────────────────────

def call_candidate(name, phone_number, skills, candidate_id, employer, hr, work_location, notice_period,custom_question,contracth,remotew):
    print("Call candidate works ")
    print(f"Candidate ID {candidate_id}")
    headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}
    payload = {
        "assistantId": AGENT_ID,
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,  # Now dynamic
        "assistantOverrides": {
            "variableValues": {
                "name": name,
                "skills": skills,
                "employer": employer,
                "hr": hr,
                "work_location": work_location,
                "days":notice_period,
                "candidate_id":str(candidate_id),
                "custom_question":custom_question,
                "contract_hiring":contracth,
                "remote_work":remotew
            }
        },
        "customer": {
            "number": f"+91{phone_number}",
            "name": name
        }
    }
    try:
        print("works here")
        resp = requests.post(VAPI_URL, headers=headers, json=payload)
        print(resp.json())
        return {"name": name, "status": resp.status_code}
    except Exception as err:
        return {"name": name, "error": str(err)}

@app.route('/call-single', methods=['POST'])
@login_required
def call_single():
    data = request.get_json()
    print("works")
    print(data)
    if not data or 'name' not in data or 'phone' not in data:
        return jsonify({'error': 'Missing candidate data'}), 400

    name = data['name']
    phone = data['phone']
    skills = data.get('skills', [])
    company = data.get('company', '')
    candidate_id= data.get('candidate_id')

    da=supabase.from_('search')\
    .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id')\
    .eq('id',session['search_id'])\
    .execute()

    call_data=da.data[0]

    contracth=None
    remotew=None
    if call_data["contract_hiring"]== True:
        contracth="Inform the candidate that this is a contract hiring."
    else:
        contracth=""
    if call_data["remote_work"] == True:
        remotew="This is a remote work."
    else:
        remotew="7.⁠ ⁠Ask for the candidate's current location and preferred location 8.⁠ ⁠If the candidate's current location is not the same as {{work_location}}, then ask if the candidate is ready to relocate, else skip this question. skip if its a remote work"

    custom_question=None
    if call_data["custom_question"]:
        custom_question=call_data["custom_question"]
    else:
        custom_question=""

    result = call_candidate(name, phone, skills=call_data["key_skills"], employer=call_data["rc_name"], hr=call_data["hc_name"],candidate_id=candidate_id,work_location=call_data["company_location"],notice_period=call_data["notice_period"],custom_question=custom_question,contracth=contracth,remotew=remotew)
    return jsonify({'status': 'success', 'result': result}), 200

@app.route("/call", methods=["POST"])
@login_required
def initiate_call():
    data = request.get_json()
    candidates = data.get("candidates", [])

    results = []
    da=supabase.from_('search')\
    .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id')\
    .eq('id',session['search_id'])\
    .execute()

    call_data=da.data[0]

    contracth=None
    remotew=None
    if call_data["contract_hiring"]== True:
        contracth="Inform the candidate that this is a contract hiring."
    else:
        contracth="Inform the candidate that this is not a contract hiring."
    if call_data["remote_work"] == True:
        remotew="This is a remote work."
    else:
        remotew="This is not a remote work."

    custom_question=None
    if call_data["custom_question"]:
        custom_question=call_data["custom_question"]
    else:
        custom_question=""

    
    for c in candidates:
        name = c.get("name")
        phone = c.get("phone")
        skills = c.get("skills", [])
        company = c.get("company", "")
        candidate_id = c.get("candidate_id")

        # You can now use candidate_id or store logs, etc.
        res = call_candidate(name, phone, skills=call_data["key_skills"], employer=call_data["rc_name"], hr=call_data["hc_name"],candidate_id=candidate_id,work_location=call_data["company_location"],notice_period=call_data["notice_period"],custom_question=custom_question,contracth=contracth,remotew=remotew)
    return jsonify({"message": "Calls initiated", "results": results})


def add_call_data(transcript,summary,structuredData,call_status,success_eval,phone,durationMinutes,name,candidate_id,):

    supabase.table("calls").insert({
        "candidate_id": candidate_id,
        "transcript": transcript,
        "call_summary": summary,
        "structured_call_data": json.dumps(structuredData),
        "status": call_status,
        "evaluation": success_eval,
        "name":name,
        "phone":phone,
        "call_duration":durationMinutes
        # "search_id": session["search_id"],
        # "user_id": session["user_id"]
    }).execute()

    supabase.table("candidates").update({
        "call_status":call_status
    }).eq('id', candidate_id).execute()

    print("Added data to call")


# ─── Webhook Receiver ────────────────────────────────────────────────────────
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    print(data)

    message = data.get("message", data)
    end_call_status=message.get("status","")
    call=message.get("call")
    assistantOverrides=call.get("assistantOverrides")
    variableValues=assistantOverrides.get("variableValues")
    candidate_id=variableValues.get("candidate_id")
    customer = call.get("customer", "")
    name=customer.get("name","")
    phone = int(customer.get("number", "")[-10:]) if customer.get("number") else None
    transcript = message.get("transcript", "")
    analysis = message.get("analysis", {})
    summary = analysis.get("summary", "")
    structuredData = analysis.get("structuredData", {})
    status=structuredData.get("re-schedule","")
    success_eval = analysis.get("successEvaluation", "")


   
    
    call_status=None
    if end_call_status == "ended":
        call_status = "Not Answered"
        supabase.table("candidates").update({
        "call_status":call_status
        }).eq('id', candidate_id).execute()
    if status == "yes":
        durationMinutes=message.get("durationMinutes","")
        call_status = "Re-schedule"
        add_call_data(transcript,summary,structuredData,call_status,success_eval,phone,durationMinutes,name,candidate_id=int(candidate_id))
        deduct_credits(session["user_id"], "rescheduled_call", reference_id=None)
    elif status == "no":
        call_status = "Called & Answered"
        add_call_data(transcript,summary,structuredData,call_status,success_eval,phone,durationMinutes,name,candidate_id=int(candidate_id))
        deduct_credits(session["user_id"], "ai_call", reference_id=None)
    
    print(f"Transcript : {transcript}")
    print(f"Summary : {summary}")
    print(f"Structured data : {structuredData}")
    print(f"Call Status : {call_status}")
    print(f"Evaluation : {success_eval}")
    print(f"Phone : {phone}")
    print(candidate_id)
    print("✅ Inserted into calls table")
    return jsonify({"status": "received"}), 200

@app.route("/api/transcript/<int:candidate_id>")
@login_required
def api_transcript(candidate_id):
    try:
        import json

        # Fetch candidate
        candidate_res = supabase.table("candidates").select("*").eq("id", candidate_id).execute()
        if not candidate_res.data:
            return jsonify({"error": "Candidate not found"}), 404
        candidate = candidate_res.data[0]

        # Fetch candidate's call
        call_res = supabase.table("calls").select("*").eq("candidate_id", candidate_id).execute()
        calls = call_res.data or []
        call = calls[0] if calls else {}

        # Parse structured_call_data
        structured_data = call.get("structured_call_data")
        if isinstance(structured_data, str):
            try:
                structured_data = json.loads(structured_data)
            except json.JSONDecodeError:
                structured_data = {}

        # Parse transcript
        transcript_text = call.get("transcript", "")
        transcript = []
        for line in transcript_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("AI:"):
                transcript.append({"speaker": "ai", "message": line[3:].strip(), "timestamp": ""})
            elif line.startswith("User:") or line.startswith("You:"):
                transcript.append({"speaker": "candidate", "message": line.split(":", 1)[1].strip(), "timestamp": ""})

        # Parse evaluation
        eval_raw = call.get("evaluation")
        evaluation = {}
        if eval_raw:
            try:
                evaluation = json.loads(eval_raw)
            except:
                evaluation = {"summary": eval_raw}

        # Final structured JSON response
        return jsonify({
            "candidate": {
                "name": candidate.get("name"),
                "phone": candidate.get("phone"),
                "email": candidate.get("email"),
                "skills": candidate.get("skills"),
                "matchScore": candidate.get("match_score"),
                "callStatus": candidate.get("call_status"),
                "totalExperience": candidate.get("total_experience"),
                "relevantExperience": candidate.get("relevant_work_experience"),
                "summary": candidate.get("summary"),
                "liked": candidate.get("liked"),
                "hiringStatus": candidate.get("hiring_status"),
                "joinStatus": candidate.get("join_status")
            },
            "transcript": transcript,
            "evaluation": evaluation,
            "structured": structured_data,
            "call": {
                "duration": call.get("call_duration"),
                "status": call.get("status"),
                "summary": call.get("call_summary", "")
            }
        })

    except Exception as e:
        print("Transcript API error:", e)
        return jsonify({"error": "Could not fetch transcript"}), 500

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    # Debug session before checking
    print("Dashboard endpoint - Session check:")
    debug_session()
    
    if "user" not in session or "user_id" not in session:
        print("Unauthorized access attempt - no session data")
        return jsonify({"error": "Unauthorized - Please login again"}), 401

    try:
        user_id = session["user_id"]
        print(f"Dashboard access granted for user_id: {user_id}")

        dashboard_data = {
            "todays_searches": get_todays_searches(user_id),
            "todays_candidates": get_todays_candidates(user_id),
            "new_joinees": get_new_joinees(user_id),
            "creds_used": get_creds_used(user_id),
            "user_name": get_user_name(user_id),
            "people_called": get_people_called(user_id),
            "weekly_activity": get_weekly_activity(user_id)
        }
        
        return make_response(jsonify(dashboard_data))
        
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ------ to fetch and render search table data in frontend -------------------------------

@app.route("/api/searches", methods=["GET"])
def get_searches():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    try:
        user_data = supabase.table("users").select("id").eq("email", session["user"]).execute()
        user_id = user_data.data[0]["id"]
        session["user_id"] = user_id  
        # print(user_id)

        response = supabase.table("search").select("*").eq("user_id", user_id).execute()
        # print(response.data)
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear_search_session", methods=["POST"])
def clear_search_session():
    # Keys to keep
    keys_to_keep = {"user", "user_id"}

    # Remove all other keys except user and user_id
    keys_to_remove = [key for key in session.keys() if key not in keys_to_keep]
    for key in keys_to_remove:
        session.pop(key, None)

    return jsonify({"message": "Search session cleared, user session intact"})

@app.route("/api/get-liked-candiates", methods=["GET"])
def get_liked_candidates():
    
    response = supabase.table("candidates") \
        .select("*") \
        .eq("liked", True) \
        .execute()

    candidates = response.data or []
    return jsonify(candidates)

@app.route("/api/user-profile", methods=["GET"])
@login_required
def get_user_profile():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    response = supabase.table("users").select("*").eq("id", user_id).single().execute()
    user = response.data
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "name": user.get("name"),
        "email": user.get("email"),
        "organization": user.get("organization"),
        "creds": user.get("creds"),
        "id": user.get("id")
    })

@app.route("/api/billing-data", methods=["GET"])
def get_billing_data():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    try:
        # Fetch recent credit logs
        credit_logs_resp = supabase.table("credit_logs") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        credit_logs = credit_logs_resp.data or []

        # Fetch payment history
        payment_history_resp = supabase.table("payment_history") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        payment_history = payment_history_resp.data or []

        # Fetch plan_id from account_preferences
        account_resp = supabase.table("account_preferences") \
            .select("plan_id") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
        account_data = account_resp.data[0] if account_resp.data else None
        plan_id = account_data["plan_id"] if account_data else None

        # Fetch start_creds from plan
        start_creds = 0
        if plan_id:
            plan_resp = supabase.table("plan") \
                .select("start_creds") \
                .eq("id", plan_id) \
                .limit(1) \
                .execute()
            plan_data = plan_resp.data[0] if plan_resp.data else None
            start_creds = plan_data["start_creds"] if plan_data else 0

        # Calculate total credits used
        total_deductions = sum(log["deductions"] for log in credit_logs if "deductions" in log)

        current_credits = max(0, start_creds - total_deductions)

        return jsonify({
            "current_credits": current_credits,
            "credit_logs": credit_logs,
            "payment_history": payment_history
        })

    except Exception as e:
        print(f"Billing API error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/api/settings", methods=["GET", "POST"])
def user_settings():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    if request.method == "GET":
        try:
            response = supabase.table("account_preferences") \
                .select("dark_theme, credit_warnings, weekly_reports, email_alerts") \
                .eq("user_id", user_id) \
                .limit(1) \
                .execute()

            data = response.data[0] if response.data else {}

            return jsonify({
                "creditWarnings": data.get("credit_warnings", True),
                "weeklyReports": data.get("weekly_reports", False),
                "emailAlerts": data.get("email_alerts", True),
                "darkTheme": data.get("dark_theme", False)
            })

        except Exception as e:
            print("Error fetching settings:", e)
            return jsonify({"error": "Internal server error"}), 500

    elif request.method == "POST":
        data = request.get_json()

        try:
            # Check if a row exists for this user
            existing = supabase.table("account_preferences") \
                .select("id") \
                .eq("user_id", user_id) \
                .limit(1) \
                .execute()

            update_data = {
                "user_id": user_id,
                "dark_theme": data.get("darkTheme", False),
                "credit_warnings": data.get("creditWarnings", True),
                "weekly_reports": data.get("weeklyReports", False),
                "email_alerts": data.get("emailAlerts", True)
            }

            if existing.data:
                pref_id = existing.data[0]["id"]
                supabase.table("account_preferences").update(update_data).eq("id", pref_id).execute()
            else:
                supabase.table("account_preferences").insert(update_data).execute()

            return jsonify({"message": "Settings updated successfully"})

        except Exception as e:
            print("Error updating settings:", e)
            return jsonify({"error": "Failed to update settings"}), 500


def deduct_credits(user_id, action_type, reference_id=None):
    credit_cost_map = {
        "search": 5,
        "process_candidate": 2,
        "ai_call": 13,
        "rescheduled_call": 5,
    }

    cost = credit_cost_map.get(action_type)
    if cost is None:
        raise ValueError("Invalid action type")

    user = supabase.table("users").select("*").eq("id", user_id).single().execute().data
    if user["creds"] < cost:
        raise Exception("Not enough credits") #show pop up component on frontend 

    # Deduct credits
    new_balance = user["creds"] - cost
    supabase.table("users").update({"creds": new_balance}).eq("id", user_id).execute()

    # Log it
    supabase.table("credits_logs").insert({
        "user_id": user_id,
        "action": action_type,
        "credits_used": cost,
        "reference_id": reference_id
    }).execute()


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    if IS_PRODUCTION:
        # Production server configuration
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
    else:
        # Development server configuration
        app.run(debug=True, host="127.0.0.1", port=5000)
