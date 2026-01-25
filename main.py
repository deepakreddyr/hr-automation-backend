import os
from dotenv import load_dotenv
from functools import wraps
import json
import jwt
from datetime import datetime, timedelta, timezone
from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, flash, make_response
)
from helpers import correct_shortlisted_indices
from dashboard_data import (
    get_dashboard_data,
    get_total_credits_used,
    get_user_name
)
from generate_embeddings import MemoryOptimizedResumeExtractor
import fitz  
from flask_cors import CORS
import requests
from aiparser import shortlist_candidates,scrape,get_candidate_details,get_questions,get_embedding,parse_reschedule_time
from supabase import create_client, Client
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import atexit
# ─── App & Config ─────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", template_folder="templates")
load_dotenv()

# Environment detection
IS_PRODUCTION = os.getenv("FLASK_ENV") == "production" or os.getenv("ENVIRONMENT") == "production"

# ─── JWT Configuration ─────────────────────────────────────────────────────

# JWT Secret Key - Use a strong secret key
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=690)  # Access token expires in 24 hours
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=800)  # Refresh token expires in 30 days

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
    "https://www.thehireai.in",
    "https://app.thehireai.in",
    "https://hr-frontend-nine-theta.vercel.app",
    
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



# Initialize APScheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Shutdown scheduler gracefully when app closes
atexit.register(lambda: scheduler.shutdown())
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

# Secure secret key (still needed for Flask operations)
app.secret_key = os.getenv("SECRET_KEY", "xJ7vK9mQ2nR8pL6wE4tY1uI0oP3aS5dF7gH9jK2lM6nB8vC1xZ4qW7eR3tY6uI9o")

print(f"JWT Authentication configured with {JWT_ACCESS_TOKEN_EXPIRES} access token expiry")

# ─── Environment Variables ─────────────────────────────────────────────────────

VAPI_URL               = os.getenv("VAPI_URL")
VAPI_API_KEY           = os.getenv("VAPI_API_KEY")
AGENT_ID               = os.getenv("AGENT_ID")
VAPI_PHONE_NUMBER_ID   = os.getenv("VAPI_PHONE_NUMBER_ID")

RINGG_API_KEY = os.getenv("RINGG_API_KEY")
RINGG_AGENT_ID = os.getenv("ASSISTANT_ID")
RINGG_NUMBER_ID = os.getenv("NUMBER_ID")
RINGG_URL = os.getenv("RINGG_URL")

url: str = os.getenv("SUPABASE_URL")
anon_key: str = os.getenv("SUPABASE_KEY")
service_key: str = os.getenv("SERVICE_ROLE_KEY")

# Validate required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Required environment variable {var} is not set")

supabase: Client = create_client(url, anon_key)
supabase_admin: Client = create_client(url, service_key) 
extractor = MemoryOptimizedResumeExtractor(
    max_workers=3,
    batch_size=50,
    max_memory_mb=512
)


# ─── JWT Helper Functions ─────────────────────────────────────────────────────

def generate_access_token(user_id, email, org_id):
    """Generate JWT access token"""
    payload = {
        'user_id': str(user_id),  
        'email': str(email),      
        'org_id': str(org_id),    # ✅ Add org_id
        'exp': datetime.now(timezone.utc) + JWT_ACCESS_TOKEN_EXPIRES,
        'iat': datetime.now(timezone.utc),
        'type': 'access'
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def generate_refresh_token(user_id, email, org_id):
    """Generate JWT refresh token"""
    payload = {
        'user_id': str(user_id),
        'email': str(email),
        'org_id': str(org_id),    # ✅ Add org_id
        'exp': datetime.now(timezone.utc) + JWT_REFRESH_TOKEN_EXPIRES,
        'iat': datetime.now(timezone.utc),
        'type': 'refresh'
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token):
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return {"error": "Token has expired"}
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}

def get_token_from_header():
    """Extract JWT token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            # Expected format: "Bearer <token>"
            token = auth_header.split(' ')[1]
            return token
        except IndexError:
            return None
    return None

def get_current_user():
    """Get current user from JWT token"""
    token = get_token_from_header()
    if not token:
        return None
    
    payload = decode_token(token)
    if 'error' in payload:
        return None
    
    if payload.get('type') != 'access':
        return None
    
    return {
        'user_id': payload.get('user_id'),
        'email': payload.get('email'),
        'org_id': payload.get('org_id')
    }

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
        user_data = supabase.table("users").select("id, email, org_id, role").eq("email", email).execute()
        if not user_data.data:
            return jsonify({"success": False, "message": "User not found in database"}), 404

        user_id = user_data.data[0]["id"]
        user_email = user_data.data[0]["email"]
        org_id= user_data.data[0]["org_id"]
        # Ensure user_id and email are strings and add debug logging
        user_id_str = str(user_id)
        user_email_str = str(user_email)
        org_id_str = str(org_id)

        print(f"Creating JWT tokens for user_id: {user_id_str} ({type(user_id)}), email: {user_email_str}, org_id: {org_id_str}")

        # Generate JWT tokens
        access_token = generate_access_token(user_id_str, user_email_str,org_id_str)
        refresh_token = generate_refresh_token(user_id_str, user_email_str,org_id_str)
        
        print(f"JWT tokens generated for user: {user_email}")
        
        return jsonify({
            "success": True, 
            "message": "Login successful",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": user_id_str,
                "email": user_email_str,
                "role": user_data.data[0]["role"]
            }
        })

    except Exception as e:
        # Log error
        print(f"Login error: {str(e)}")
        return jsonify({"success": False, "message": "Login failed"}), 500

@app.route("/refresh", methods=["POST"])
def refresh_token():
    """Refresh JWT access token using refresh token"""
    try:
        data = request.get_json()
        refresh_token = data.get("refresh_token")
        
        if not refresh_token:
            return jsonify({"success": False, "message": "Refresh token required"}), 400
        
        # Decode refresh token
        payload = decode_token(refresh_token)
        if 'error' in payload:
            return jsonify({"success": False, "message": payload['error']}), 401
        
        if payload.get('type') != 'refresh':
            return jsonify({"success": False, "message": "Invalid token type"}), 401
        
        # Generate new access token
        new_access_token = generate_access_token(payload['user_id'], payload['email'], payload['org_id'])
        
        return jsonify({
            "success": True,
            "access_token": new_access_token
        })
        
    except Exception as e:
        print(f"Refresh token error: {str(e)}")
        return jsonify({"success": False, "message": "Token refresh failed"}), 500

@app.route("/api/user-info", methods=["GET"])
def get_user_info():
    """Get current user information from JWT token"""
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "message": "Authentication required"}), 401
    
    return jsonify({
        "success": True,
        "user": user
    })

@app.route("/logout", methods=["GET", "POST"])
def logout():
    """Logout endpoint (client-side token removal)"""
    # With JWT, logout is primarily handled client-side by removing tokens
    # You could implement a token blacklist here if needed
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route("/404")
def error():
    return jsonify({"message": "Unknown API"}), 404

# ─── Utility: JWT Authentication Decorator ────────────────────────────────────

def jwt_required(f):
    """JWT authentication decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            # For API endpoints, return JSON error
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({"success": False, "message": "Authentication required"}), 401
            # For web pages, redirect to login
            return redirect(url_for("login"))
        
        # Add user info to request context
        request.current_user = user
        return f(*args, **kwargs)
    return decorated

def optional_jwt(f):
    """Optional JWT authentication decorator - doesn't fail if no token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        request.current_user = user  # Will be None if not authenticated
        return f(*args, **kwargs)
    return decorated

@app.route('/api/search-status/<int:search_id>')
@jwt_required
def get_search_status(search_id):
    response = supabase.table("search").select("status").eq("id", search_id).single().execute()
    if response.data:
        return jsonify({"status": response.data.get("status")})
    else:
        return jsonify({"error": "Search not found"}), 404

@app.route("/api/shortlist", methods=["POST"])
@jwt_required
def shortlist():
    """Create new shortlist or update existing one"""
    user = request.current_user
    user_id = user["user_id"]
    org_id = user.get("org_id")
    
    # Extract form data
    candidate_data = request.form.get("candidateData", "").strip()
    skills = request.form.get("requiredSkills", "").strip()
    job_role = request.form.get("jobRole", "").strip()
    search_name = request.form.get("searchName", "").strip()
    jd_file = request.files.get("jdFile")
    noc = int(request.form.get("numberOfCandidates", "0"))
    
    # Get search_id from form data (if updating)
    search_id = request.form.get("search_id")
    if search_id:
        try:
            search_id = int(search_id)
        except (ValueError, TypeError):
            search_id = None
    
    # New fields
    hiring_company = request.form.get("hiringCompany", "").strip()
    company_location = request.form.get("companyLocation", "").strip()
    hr_company = request.form.get("hrCompany", "").strip()
    notice_period = request.form.get("noticePeriod", "").strip()
    remote_work_available = request.form.get("remoteWorkAvailable", "").strip()
    contract_hiring = request.form.get("contractHiring", "").strip()
    
    # Validation
    errors = []
    if not candidate_data:
        errors.append("Candidate data is required.")
    if not skills:
        errors.append("Required skills cannot be empty.")
    if not job_role:
        errors.append("Job role is required.")
    if not search_name:
        errors.append("Search name is required.")
    if noc <= 0:
        errors.append("Number of candidates must be greater than 0.")
    if not jd_file or not jd_file.filename.endswith(".pdf"):
        errors.append("Job Description PDF is required.")
    
    # Validate new fields
    if not hiring_company:
        errors.append("Hiring company is required.")
    if not company_location:
        errors.append("Company location is required.")
    if not hr_company:
        errors.append("HR company is required.")
    if not notice_period:
        errors.append("Notice period is required.")
    if not remote_work_available:
        errors.append("Remote work available is required.")
    if not contract_hiring:
        errors.append("Contract hiring is required.")
    
    if errors:
        return jsonify(success=False, errors=errors), 400

    # Extract JD text from uploaded PDF
    jd_text = ""
    if jd_file and jd_file.filename.endswith(".pdf"):
        try:
            doc = fitz.open(stream=jd_file.read(), filetype="pdf")
            jd_text = "\n".join([page.get_text() for page in doc])
        except Exception as e:
            return jsonify(success=False, error=f"Error reading JD PDF: {e}"), 500

    # Process shortlisting before database operation
    final_candidates = scrape(candidate_data)
    shortlisted_indices = shortlist_candidates(final_candidates, skills, noc, jd_text)
    corrected_shortlist = correct_shortlisted_indices(shortlisted_indices, final_candidates)
    
    if corrected_shortlist == []:  # nothing matched        
        return jsonify({
            "success": False,
            "message": "No candidates matched the required skills.",
            "search_id": search_id
        }), 200

    # Convert remote_work and contract_hiring to boolean for database
    remote_work_bool = remote_work_available.lower() in ['yes', 'hybrid']
    contract_hiring_bool = contract_hiring.lower() == 'yes'

    # Prepare data for database operation
    search_data = {
        "user_id": user_id,
        "org_id": org_id,
        "processed": False,
        "remote_work": remote_work_bool,
        "contract_hiring": contract_hiring_bool,
        "key_skills": skills,
        "job_role": job_role,
        "raw_data": candidate_data,
        "shortlisted_index": json.dumps(corrected_shortlist),
        "noc": noc,
        "job_description": jd_text,
        "status": "process",
        "search_name": search_name,
        "search_type": "Naukri",
        # New fields
        "rc_name": hiring_company,
        "company_location": company_location,
        "hc_name": hr_company,
        "notice_period": notice_period
    }

    try:
        if search_id:
            # Update existing search
            # First check if search exists and belongs to user
            existing_search = supabase.table("search").select("*").eq("id", search_id).eq("user_id", user_id).execute()
            
            if not existing_search.data:
                return jsonify({
                    "success": False,
                    "message": "Search not found or access denied."
                }), 404
            
            # Get existing data
            existing_data = existing_search.data[0]
            existing_shortlist = existing_data.get("shortlisted_index", "[]")
            existing_process_state = existing_data.get("process_state", "{}")
            
            # Parse existing data
            if isinstance(existing_shortlist, str):
                existing_shortlist = json.loads(existing_shortlist)
            if isinstance(existing_process_state, str):
                existing_process_state = json.loads(existing_process_state)
            
            # Get existing resume_dict (already submitted resumes)
            existing_resume_dict = existing_process_state.get("resume_dict", {})
            
            # Merge old and new candidates
            # Keep old candidates that have resumes submitted
            old_candidates_with_resumes = []
            for candidate in existing_shortlist:
                candidate_key = f"candidate_{candidate['index']}"
                if candidate_key in existing_resume_dict:
                    old_candidates_with_resumes.append(candidate)
            
            # Find the maximum index from old candidates to avoid conflicts
            max_old_index = -1
            if old_candidates_with_resumes:
                max_old_index = max(c['index'] for c in old_candidates_with_resumes)
            
            # Remap new candidate indices to avoid conflicts with old ones
            remapped_new_candidates = []
            new_resume_mapping = {}  # Map old index to new index for updating resume_dict
            
            for idx, new_candidate in enumerate(corrected_shortlist):
                old_index = new_candidate['index']
                new_index = max_old_index + 1 + idx
                
                # Create remapped candidate
                remapped_candidate = new_candidate.copy()
                remapped_candidate['index'] = new_index
                remapped_candidate['original_index'] = old_index  # Keep track of original
                remapped_new_candidates.append(remapped_candidate)
                
                new_resume_mapping[old_index] = new_index
            
            # Combine: old candidates first, then new ones with remapped indices
            merged_shortlist = old_candidates_with_resumes + remapped_new_candidates
            
            # Update process_state to preserve old resumes
            updated_process_state = {
                "resume_dict": existing_resume_dict,
                "old_candidate_count": len(old_candidates_with_resumes)
            }
            
            # Debug logging
            print(f"Update shortlist - Old candidates: {len(old_candidates_with_resumes)}, New candidates: {len(remapped_new_candidates)}")
            print(f"Old candidate indices: {[c['index'] for c in old_candidates_with_resumes]}")
            print(f"New candidate indices: {[c['index'] for c in remapped_new_candidates]}")
            print(f"Merged shortlist total: {len(merged_shortlist)}")
            
            # Update search_data with merged list
            search_data["shortlisted_index"] = json.dumps(merged_shortlist)
            search_data["process_state"] = json.dumps(updated_process_state)
            search_data["status"] = "process"  # Reset to process status
            
            # Update the existing search
            search_resp = supabase.table("search").update(search_data).eq("id", search_id).execute()
            
            if not search_resp.data:
                return jsonify({"success": False, "error": "Error updating search"}), 500
                
            final_search_id = search_id
            operation_message = f"Shortlist updated successfully. {len(old_candidates_with_resumes)} old candidates preserved, {len(remapped_new_candidates)} new candidates added."
            
        else:
            # Create new search
            search_resp = supabase.table("search").insert(search_data).execute()
            
            if not search_resp.data:
                return jsonify({"success": False, "error": "Error creating search"}), 500
            
            final_search_id = search_resp.data[0]["id"]
            operation_message = "Shortlist created successfully"
            
            # Only deduct credits for new searches, not updates
            deduct_credits(user_id, org_id, "search", reference_id=final_search_id)

        return jsonify({
            "success": True,
            "message": operation_message,
            "search_id": final_search_id,
            "shortlist": corrected_shortlist if not search_id else merged_shortlist,
            "is_update": bool(search_id),
            "old_count": len(old_candidates_with_resumes) if search_id else 0,
            "new_count": len(corrected_shortlist)
        })
        
    except Exception as e:
        print(f"Error in shortlist operation: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An error occurred while processing your request."
        }), 500
    
@app.route("/api/shortlist/simple", methods=["POST"])
@jwt_required
def create_shortlist():
    try:
        user = request.current_user
        user_id = user["user_id"]
        org_id = user.get("org_id")

        # === Get form fields ===
        search_name = request.form.get("searchName")
        job_role = request.form.get("jobRole")
        skills = request.form.get("skills")
        num_candidates = request.form.get("numCandidates")
        resume_link = request.form.get("resumeLink")
        jd_file = request.files.get("jdFile")

        # Extra fields from wizard
        hiring_company = request.form.get("hiringCompany")
        hr_company = request.form.get("hrCompany")
        company_location = request.form.get("companyLocation")
        notice_period = request.form.get("noticePeriod")
        remote_work = request.form.get("remoteWork", "false").lower() == "true"
        contract_hiring = request.form.get("contractHiring", "false").lower() == "true"

        if not search_name or not job_role or not skills or not resume_link or not user_id:
            return jsonify({"success": False, "message": "Missing required fields"}), 400

        # === STEP 1: Extract JD text ===
        jd_text = ""
        if jd_file and jd_file.filename.endswith(".pdf"):
            doc = fitz.open(stream=jd_file.read(), filetype="pdf")
            jd_text = "\n".join([page.get_text() for page in doc])

        # === STEP 2: Insert search record ===
        response = supabase.table("search").insert({
            "user_id": user_id,
            "org_id": org_id,
            "search_name": search_name,
            "job_role": job_role,
            "key_skills": skills,
            "job_description": jd_text,
            "resume_link": resume_link,
            "noc": int(num_candidates) if num_candidates else None,
            "rc_name": hiring_company,
            "hc_name": hr_company,
            "company_location": company_location,
            "notice_period": notice_period,
            "remote_work": remote_work,
            "contract_hiring": contract_hiring,
            "status": "results",
            "processed": True,
            "search_type": "Simple"
        }).execute()
        search_id = response.data[0]["id"]

        # === STEP 3: Extract resumes ===
        raw_resumes = []
        for batch in extractor.process_drive_spreadsheet_optimized(resume_link, user_id, search_id):
            raw_resumes.extend(batch)

        if not raw_resumes:
            return jsonify({"success": False, "message": "No resumes processed"}), 500

        # === STEP 4: Embedding-based shortlist ===
        jd_embedding = get_embedding(jd_text)

        matches = supabase.rpc("match_resumes", {
            "query_embedding": jd_embedding,
            "similarity_threshold": 0.3,
            "match_count": int(num_candidates) if num_candidates else 5,
            "input_search_id": search_id
        }).execute()

        shortlisted = matches.data or []
        print(f"{shortlisted} : shortlisted")
        shortlisted_texts = [c["resume_text"] for c in shortlisted if "resume_text" in c]

        if not shortlisted_texts:
            return jsonify({"success": False, "message": "No candidates shortlisted"}), 200

        # === STEP 5: Pass shortlisted resumes into AI scoring ===

        ai_results = get_candidate_details(shortlisted_texts, jd_text, skills)

        if not ai_results:
            return jsonify({"success": False, "message": "No candidates after AI scoring"}), 200

        # === STEP 6: Insert AI-enriched candidates into table ===
        inserts = []
        for cand in ai_results:
            insert_obj = {
                "user_id": user_id,
                "org_id": org_id,
                "search_id": search_id,
                "match_score": cand.get("match_score"),
            }

            if cand.get("match_score", 0) > 70:
                # --- sanitize phone number ---
                phone_raw = cand.get("phone")
                phone_clean = None
                if phone_raw:
                    import re
                    digits = re.sub(r"\D", "", str(phone_raw))
                    if digits:
                        try:
                            phone_clean = int(digits)
                        except Exception:
                            phone_clean = None

                insert_obj.update({
                    "name": cand.get("name"),
                    "phone": phone_clean,  # stored as bigint-safe
                    "email": cand.get("email"),
                    "summary": cand.get("job_summary"),
                    "skills_experience": cand.get("experience_in_skills"),
                    "total_experience": cand.get("total_experience"),
                    "relevant_work_experience": cand.get("relevant_experience"),
                })
            else:
                insert_obj["summary"] = cand.get("reason_to_reject")

            inserts.append(insert_obj)

        if inserts:
            supabase.table("candidates").insert(inserts).execute()
            deduct_credits(user_id, org_id, "search", reference_id=search_id)
            deduct_credits(user_id,org_id, "process_candidate", reference_id=None)

        return jsonify({
            "success": True,
            "message": "Shortlist created successfully",
            "search_id": search_id,
            "resumes_processed": len(raw_resumes),
            "shortlisted_candidates": ai_results,
        }), 200

    except Exception as e:
        print(f"Error in /api/shortlist/simple: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/process/<int:search_id>", methods=["GET", "POST"]) 
@jwt_required 
def process(search_id):
    user = request.current_user     
    user_id = user["user_id"]      

    # === Load search ===     
    result = supabase.table("search").select(
        "shortlisted_index, process_state, job_description, key_skills, rc_name, company_location, hc_name, notice_period, remote_work, contract_hiring"
    ).eq("id", search_id).single().execute()     

    if not result.data:         
        return jsonify(success=False, error="Search not found"), 404      

    # Load shortlisted data (now full JSON, not just indices)
    shortlisted = result.data.get("shortlisted_index") or []     
    if isinstance(shortlisted, str):         
        shortlisted = json.loads(shortlisted or "[]")      

    process_state = result.data.get("process_state") or {}     
    jd = result.data.get("job_description")     
    skills = result.data.get("key_skills")      

    if isinstance(process_state, str):         
        try:             
            process_state = json.loads(process_state or "{}")         
        except json.JSONDecodeError:             
            process_state = {}      

    # Get existing resumes and old candidate count
    resume_dict = process_state.get("resume_dict", {})
    old_candidate_count = process_state.get("old_candidate_count", 0)
    
    # Calculate how many candidates have resumes (both old and newly added)
    candidates_with_resumes = len(resume_dict)
    target = len(shortlisted)
    
    # Find current candidate (first one without resume)
    current_candidate = None
    current_index_in_list = None
    
    for idx, candidate in enumerate(shortlisted):
        candidate_key = f"candidate_{candidate['index']}"
        if candidate_key not in resume_dict:
            current_candidate = candidate
            current_index_in_list = idx
            break
    
    is_last = (candidates_with_resumes == target - 1)
    
    # Separate old and new candidates for display
    old_candidates = shortlisted[:old_candidate_count] if old_candidate_count > 0 else []
    new_candidates = shortlisted[old_candidate_count:] if old_candidate_count > 0 else shortlisted

    # === POST submission ===
    if request.method == "POST":         
        if not current_candidate:
            return jsonify(success=False, errors=["No more candidates to process."]), 400
            
        resume_text = request.form.get("resumeText", "").strip()
        resume_file = request.files.get("resumeFile")
        
        # Extract text from file if provided
        if resume_file:
            try:
                filename = resume_file.filename.lower()
                
                if filename.endswith('.pdf'):
                    import fitz
                    pdf_doc = fitz.open(stream=resume_file.read(), filetype="pdf")
                    resume_text = "\n".join([page.get_text() for page in pdf_doc])
                    pdf_doc.close()
                    
                elif filename.endswith('.txt'):
                    resume_text = resume_file.read().decode('utf-8')
                    
                elif filename.endswith('.csv'):
                    resume_text = resume_file.read().decode('utf-8')
                    
                else:
                    return jsonify(success=False, errors=["Unsupported file format. Please upload PDF, TXT, or CSV."]), 400
                    
            except Exception as e:
                return jsonify(success=False, errors=[f"Error reading file: {str(e)}"]), 500

        # Validation
        errors = []         
        if not resume_text:             
            errors.append("Resume text or file is required.")         

        if errors:             
            return jsonify(success=False, errors=errors), 400          

        # === Save resume ===         
        resume_dict[f"candidate_{current_candidate['index']}"] = resume_text
        process_state["resume_dict"] = resume_dict
        
        # Keep old_candidate_count
        if old_candidate_count > 0:
            process_state["old_candidate_count"] = old_candidate_count

        # === Update process state ===
        supabase.table("search").update({             
            "process_state": json.dumps(process_state)         
        }).eq("id", search_id).execute()          

        candidates_with_resumes = len(resume_dict)
        
        # === Final submission ===
        if candidates_with_resumes == target:
            supabase.table("search").update({
                "status": "processing",
                "processed": False
            }).eq("id", search_id).execute()
            
            return jsonify({
                "success": True,
                "submitted": candidates_with_resumes,
                "target": target,
                "candidate": current_candidate,
                "isLast": True,
                "next": False,
                "processing": True,
                "search_id": search_id
            })

        # Find next candidate without resume
        next_candidate = None
        for candidate in shortlisted:
            candidate_key = f"candidate_{candidate['index']}"
            if candidate_key not in resume_dict:
                next_candidate = candidate
                break
        
        return jsonify({             
            "success": True,             
            "submitted": candidates_with_resumes,             
            "target": target,             
            "candidate": next_candidate,
            "isLast": (candidates_with_resumes == target - 1),             
            "next": True         
        })      
    
    # === GET (initial load) ===
    right_fields = {
        "hiringCompany": result.data.get("rc_name", ""),
        "companyLocation": result.data.get("company_location", ""),
        "hrCompany": result.data.get("hc_name", ""),
        "noticePeriod": result.data.get("notice_period", ""),
        "remoteWork": result.data.get("remote_work", False),
        "contractHiring": result.data.get("contract_hiring", False),
    }
    
    return jsonify({         
        "success": True,         
        "submitted": candidates_with_resumes,         
        "target": target,         
        "candidate": current_candidate,
        "isLast": is_last,         
        "shortlisted": shortlisted,
        "shortlisted_indices": [c.get('index') for c in shortlisted if isinstance(c, dict)],
        "old_candidate_count": old_candidate_count,
        "old_candidates": old_candidates,
        "new_candidates": new_candidates,
        "right_fields": right_fields,         
        "next": True     
    })


# Process candidates route - handles both old and new candidates
@app.route("/api/process-candidates/<int:search_id>", methods=["POST"])
@jwt_required
def process_candidates(search_id):
    user = request.current_user
    user_id = user["user_id"]
    org_id = user.get('org_id')

    try:
        # Get search data
        result = supabase.table("search").select(
            "process_state, job_description, key_skills, shortlisted_index"
        ).eq("id", search_id).single().execute()
        
        if not result.data:
            return jsonify(success=False, error="Search not found"), 404
        
        process_state = result.data.get("process_state") or {}
        if isinstance(process_state, str):
            process_state = json.loads(process_state or "{}")
        
        shortlisted = result.data.get("shortlisted_index") or []
        if isinstance(shortlisted, str):
            shortlisted = json.loads(shortlisted or "[]")
        
        jd = result.data.get("job_description")
        skills = result.data.get("key_skills")
        resume_dict = process_state.get("resume_dict", {})
        old_candidate_count = process_state.get("old_candidate_count", 0)
        
        print(f"Processing {len(resume_dict)} resumes for search {search_id}")
        print(f"Old candidates: {old_candidate_count}, Total candidates: {len(shortlisted)}")
        
        # Separate old and new candidate resumes
        old_resumes = []
        new_resumes = []
        
        for idx, candidate in enumerate(shortlisted):
            candidate_key = f"candidate_{candidate['index']}"
            if candidate_key in resume_dict:
                resume_text = resume_dict[candidate_key]
                if idx < old_candidate_count:
                    # This is an old candidate (already processed before)
                    old_resumes.append(resume_text)
                else:
                    # This is a new candidate (added in this update)
                    new_resumes.append(resume_text)
        
        print(f"Old resumes: {len(old_resumes)}, New resumes: {len(new_resumes)}")
        
        # Combine all resumes for processing
        all_resumes = "\n\n--- RESUME SEPARATOR ---\n\n".join(old_resumes + new_resumes)
        
        # Validate required data
        if not all_resumes.strip():
            supabase.table("search").update({"status": "error", "processed": True}).eq("id", search_id).execute()
            return jsonify(success=False, error="No resume data found"), 400
        if not jd or not jd.strip():
            supabase.table("search").update({"status": "error", "processed": True}).eq("id", search_id).execute()
            return jsonify(success=False, error="No job description found"), 400
        
        print(f"Processing {len(resume_dict)} total resumes, combined length: {len(all_resumes)}")
        
        # Get candidate details from AI (process all candidates together)
        candidate_data = get_candidate_details(all_resumes, jd, skills)
        print(f"AI returned {len(candidate_data) if isinstance(candidate_data, list) else 0} candidates")
        
        if not candidate_data or not isinstance(candidate_data, list):
            print("No valid candidate data received from AI")
            supabase.table("search").update({"status": "error", "processed": True}).eq("id", search_id).execute()
            return jsonify(success=False, error="Failed to process candidates"), 500
        
        # Remove duplicates and process candidates
        unique_candidates = []
        seen = set()
        
        for candidate in candidate_data:
            if not isinstance(candidate, dict):
                continue
            
            email = candidate.get("email", "").lower().strip()
            phone_raw = candidate.get("phone", "")
            phone_digits = "".join(filter(str.isdigit, phone_raw))[-10:] if phone_raw else ""
            
            identifier = f"{email}_{phone_digits}"
            if identifier not in seen and email and phone_digits:
                seen.add(identifier)
                unique_candidates.append(candidate)
        
        print(f"Processing {len(unique_candidates)} unique candidates")
        
        # Delete existing candidates for this search (to avoid duplicates during re-processing)
        try:
            supabase.table("candidates").delete().eq("search_id", search_id).execute()
            print(f"Deleted existing candidates for search {search_id}")
        except Exception as delete_error:
            print(f"Error deleting existing candidates: {delete_error}")
        
        # Insert all candidates (both old and new) into database
        inserted_count = 0
        for candidate in unique_candidates:
            try:
                match_score = candidate.get("match_score", 0)
                if match_score <= 70:
                    continue
                
                name = candidate.get("name", "").strip()
                email = candidate.get("email", "").strip().lower()
                phone_raw = candidate.get("phone", "")
                phone_digits = "".join(filter(str.isdigit, phone_raw))[-10:] if phone_raw else ""
                
                if not name or not email:
                    continue
                
                phone = int(phone_digits) if len(phone_digits) == 10 else None
                
                # Prepare skills data
                experience_in_skills = candidate.get("experience_in_skills", {})
                skills_list = ", ".join(experience_in_skills.keys()) if experience_in_skills else ""
                
                # Insert candidate
                insert_data = {
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "skills": skills_list,
                    "summary": candidate.get("job_summary", ""),
                    "skills_experience": experience_in_skills,
                    "search_id": search_id,
                    "user_id": user_id,
                    "total_experience": candidate.get("total_experience", ""),
                    "relevant_work_experience": candidate.get("relevant_experience", ""),
                    "match_score": match_score,
                    "org_id": org_id
                }
                
                # Add history_id if it exists
                try:
                    history_check = supabase.table("history").select("id").eq("id", search_id).execute()
                    if history_check.data:
                        insert_data["history_id"] = search_id
                except:
                    pass
                
                result = supabase.table("candidates").insert(insert_data).execute()
                
                if result.data:
                    inserted_count += 1
                    print(f"Inserted candidate: {name} ({email})")
                
            except Exception as candidate_error:
                print(f"Error processing candidate: {candidate_error}")
                continue
        
        print(f"Successfully inserted {inserted_count} candidates")
        
        # Update search status
        supabase.table("search").update({
            "status": "results",
            "processed": True
        }).eq("id", search_id).execute()
        
        # Deduct credits only once (not per update)
        # Check if credits were already deducted by looking at a flag
        credits_deducted = process_state.get("credits_deducted", False)
        if not credits_deducted:
            deduct_credits(user_id, org_id, "process_candidate", reference_id=None)
            # Mark that credits have been deducted
            process_state["credits_deducted"] = True
            supabase.table("search").update({
                "process_state": json.dumps(process_state)
            }).eq("id", search_id).execute()
        
        return jsonify({
            "success": True,
            "candidates_processed": inserted_count,
            "search_id": search_id,
            "old_candidates": len(old_resumes),
            "new_candidates": len(new_resumes)
        })
        
    except Exception as processing_error:
        print(f"Error in candidate processing: {processing_error}")
        import traceback
        traceback.print_exc()
        
        # Mark as error but processed to prevent infinite loops
        supabase.table("search").update({
            "status": "error",
            "processed": True
        }).eq("id", search_id).execute()
        
        return jsonify({
            "success": False,
            "error": "Failed to process candidates. Please try again."
        }), 500


# Status checking route
@app.route("/api/check-processing/<int:search_id>", methods=["GET"])
@jwt_required
def check_processing(search_id):
    try:
        result = supabase.table("search").select("status, processed").eq("id", search_id).single().execute()
        if not result.data:
            return jsonify(success=False, error="Search not found"), 404
        
        status = result.data.get("status")
        processed = result.data.get("processed", False)
        
        return jsonify({
            "success": True,
            "processed": processed,
            "status": status,
            "search_id": search_id
        })
        
    except Exception as e:
        print(f"Error checking processing status: {e}")
        return jsonify(success=False, error="Failed to check processing status"), 500
    
@app.route("/api/get-questions", methods=["GET"])
@jwt_required
def generate_questions():
    """Generate interview questions with JWT authentication"""
    # Get user info from JWT token
    user = request.current_user
    user_id = user['user_id']
    
    # Get search_id from query parameters or request body
    search_id = request.args.get('search_id')
    if not search_id:
        return jsonify({"error": "search_id parameter is required"}), 400

    try:
        print(f"Generating questions for search_id: {search_id}, user_id: {user_id}")
        
        # Verify the search belongs to the current user (security check)
        result = supabase.table("search").select("job_description").eq("id", search_id).eq("user_id", user_id).execute().data
        
        if not result:
            return jsonify({"error": "Search not found or access denied"}), 404
            
        if not result[0].get("job_description"):
            return jsonify({"error": "No job description found"}), 404

        jd = result[0]["job_description"]
        print(f"Job description length: {len(jd)}")
        
        # Generate questions using AI parser
        raw_questions = get_questions(jd)

        if isinstance(raw_questions, str):
            questions = [q.strip("1234567890.- ") for q in raw_questions.strip().split("\n") if q.strip()]
        else:
            questions = raw_questions  # Already a list
            
        print(f"Generated {len(questions)} questions")
        
        return jsonify({
            "success": True,
            "questions": questions,
            "search_id": search_id
        })
        
    except Exception as e:
        print(f"Generate questions error: {str(e)}")
        return jsonify({"error": "Failed to generate questions"}), 500

@app.route("/api/custom-question", methods=["GET", "POST"])
@jwt_required
def custom_question_handler():
    """Handles both saving (POST) and fetching (GET) a custom question for a search."""
    user = request.current_user
    user_id = user['user_id']
    
    if request.method == "POST":
        # --- POST: Save Custom Question ---
        data = request.get_json()
        search_id = data.get('search_id')
        question = data.get('question')

        if not search_id or not question:
            return jsonify({"error": "search_id and question are required"}), 400

        try:
            # Update the 'search' table with the custom_question
            result = supabase.table("search").update({"custom_question": question}).eq("id", search_id).eq("user_id", user_id).execute()
            
            # Check if any rows were updated (ensures search exists and belongs to user)
            if not result.data:
                return jsonify({"error": "Search not found or access denied"}), 404

            return jsonify({"success": True, "message": "Custom question saved successfully"}), 200

        except Exception as e:
            print(f"Save custom question error: {str(e)}")
            return jsonify({"error": "Failed to save custom question"}), 500

    elif request.method == "GET":
        # --- GET: Fetch Custom Question ---
        search_id = request.args.get('search_id')
        
        if not search_id:
            return jsonify({"error": "search_id is required"}), 400

        try:
            # Select the 'custom_question' from the 'search' table
            result = supabase.table("search").select("custom_question").eq("id", search_id).eq("user_id", user_id).single().execute()
            
            question_data = result.data

            if not question_data:
                return jsonify({"error": "Search not found or access denied"}), 404

            custom_question = question_data.get('custom_question')

            return jsonify({"success": True, "custom_question": custom_question}), 200

        except Exception as e:
            print(f"Get custom question error: {str(e)}")
            return jsonify({"error": "Failed to fetch custom question"}), 500

@app.route('/api/results', methods=["GET"])
@jwt_required
def results():
    """Get search results with JWT authentication"""
    # Get user info from JWT token
    user = request.current_user
    user_id = user['user_id']
    
    # Get search_id from query parameters
    search_id = request.args.get('searchID')
    if not search_id:
        return jsonify({"error": "searchID parameter is required"}), 400

    try:
        search_id = int(search_id)
        print(f"Fetching results for search_id: {search_id}, user_id: {user_id}")
        
        # Verify the search belongs to the current user (security check)
        search_check = supabase.table("search").select("id").eq("id", search_id).eq("user_id", user_id).execute()
        if not search_check.data:
            return jsonify({"error": "Search not found or access denied"}), 404
        
        # Get candidates for this search
        response = supabase.table("candidates").select("*").eq("search_id", search_id).execute()
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
            
        print(f"Retrieved {len(shortlisted_candidates)} candidates")
        
        return jsonify({
            "success": True,
            "candidates": shortlisted_candidates,
            "total": len(shortlisted_candidates),
            "calls_scheduled": calls_scheduled,
            "rescheduled_calls": rescheduled_calls,
            "search_id": search_id
        })
        
    except ValueError:
        return jsonify({"error": "Invalid searchID format"}), 400
    except Exception as e:
        print(f"Results API error: {str(e)}")
        return jsonify({"error": "Failed to fetch results"}), 500

@app.route("/api/like-candidate", methods=["POST"])
@jwt_required
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

@app.route("/api/candidate", methods=["GET", "POST", "PUT", "DELETE"])
@jwt_required
def candidate_handler():
    """Handles CRUD operations for candidates."""
    user = request.current_user
    user_id = user['user_id']
    
    if request.method == "GET":
        # --- GET: Fetch a single candidate by ID ---
        candidate_id = request.args.get('candidate_id')
        
        if not candidate_id:
            return jsonify({"error": "candidate_id is required"}), 400

        try:
            result = supabase.table("candidates").select("*").eq("id", candidate_id).eq("user_id", user_id).single().execute()
            
            if not result.data:
                return jsonify({"error": "Candidate not found or access denied"}), 404

            return jsonify({"success": True, "candidate": result.data}), 200

        except Exception as e:
            print(f"Get candidate error: {str(e)}")
            return jsonify({"error": "Failed to fetch candidate"}), 500

    elif request.method == "POST":
        # --- POST: Add a new candidate ---
        data = request.get_json()
        
        # Required fields
        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        search_id = data.get('search_id')
        
        # Optional fields with defaults
        skills = data.get('skills', '')
        total_experience = data.get('total_experience', '0')
        relevant_work_experience = data.get('relevant_work_experience', '0')
        match_score = data.get('match_score', 0.0)
        summary = data.get('summary', '')
        call_status = data.get('call_status', 'not_called')
        
        if not all([name, email, phone, search_id]):
            return jsonify({"error": "name, email, phone, and search_id are required"}), 400

        try:
            # Verify that the search_id belongs to the user
            search_result = supabase.table("search").select("id").eq("id", search_id).eq("user_id", user_id).single().execute()
            
            if not search_result.data:
                return jsonify({"error": "Search not found or access denied"}), 404

            # Insert the new candidate
            candidate_data = {
                "name": name,
                "email": email,
                "phone": phone,
                "skills": skills,
                "search_id": search_id,
                "user_id": user_id,
                "total_experience": total_experience,
                "relevant_work_experience": relevant_work_experience,
                "match_score": match_score,
                "summary": summary,
                "call_status": call_status,
                "liked": False,
                "hiring_status": False,
                "join_status": False
            }
            
            result = supabase.table("candidates").insert(candidate_data).execute()
            
            if not result.data:
                return jsonify({"error": "Failed to create candidate"}), 500

            return jsonify({"success": True, "message": "Candidate added successfully", "candidate": result.data[0]}), 201

        except Exception as e:
            print(f"Add candidate error: {str(e)}")
            return jsonify({"error": f"Failed to add candidate: {str(e)}"}), 500

    elif request.method == "PUT":
        # --- PUT: Update an existing candidate ---
        data = request.get_json()
        candidate_id = data.get('candidate_id')
        
        if not candidate_id:
            return jsonify({"error": "candidate_id is required"}), 400

        try:
            # Verify the candidate exists and belongs to the user
            existing = supabase.table("candidates").select("id").eq("id", candidate_id).eq("user_id", user_id).single().execute()
            
            if not existing.data:
                return jsonify({"error": "Candidate not found or access denied"}), 404

            # Build update object with only provided fields
            update_data = {}
            updatable_fields = [
                'name', 'email', 'phone', 'skills', 'total_experience', 
                'relevant_work_experience', 'match_score', 'summary', 
                'call_status', 'liked', 'hiring_status', 'join_status'
            ]
            
            for field in updatable_fields:
                if field in data:
                    update_data[field] = data[field]
            
            if not update_data:
                return jsonify({"error": "No valid fields to update"}), 400

            # Update the candidate
            result = supabase.table("candidates").update(update_data).eq("id", candidate_id).eq("user_id", user_id).execute()
            
            if not result.data:
                return jsonify({"error": "Failed to update candidate"}), 500

            return jsonify({"success": True, "message": "Candidate updated successfully", "candidate": result.data[0]}), 200

        except Exception as e:
            print(f"Update candidate error: {str(e)}")
            return jsonify({"error": f"Failed to update candidate: {str(e)}"}), 500

    elif request.method == "DELETE":
        data = request.get_json()
        candidate_id = data.get('candidate_id')
        
        if not candidate_id:
            return jsonify({"error": "candidate_id is required"}), 400

        try:
            # First verify the candidate exists and belongs to user
            existing = supabase.table("candidates").select("id, name").eq("id", candidate_id).eq("user_id", user_id).execute()
            
            if not existing.data:
                return jsonify({"error": "Candidate not found or access denied"}), 404
            
            # Now delete
            result = supabase.table("candidates").delete().eq("id", candidate_id).eq("user_id", user_id).execute()
            
            return jsonify({
                "success": True, 
                "message": f"Candidate {existing.data[0].get('name', '')} deleted successfully"
            }), 200

        except Exception as e:
            print(f"Delete candidate error: {str(e)}")
            return jsonify({"error": "Failed to delete candidate"}), 500

@app.route("/api/unlike-candidate", methods=["POST"])
@jwt_required
def unlike_candidate():
    data = request.get_json()
    candidate_id = data.get("candidate_id")
    liked = False

    if not candidate_id:
        return jsonify({"error": "Missing candidate ID"}), 400

    # Update 'liked' column in Supabase
    supabase.table("candidates").update({"liked": liked}).eq("id", candidate_id).execute()
    return jsonify({"success": True}), 200

@app.route('/api/add-final-select', methods=['POST'])
@jwt_required
def mark_final_selects():
    data = request.json
    candidate_ids = data.get('candidate_ids', [])
    for cid in candidate_ids:
        supabase.table('candidates').update({
            "hiring_status": True
        }).eq('id', cid).execute()
    return jsonify({"message": "Updated successfully"}), 200

@app.route("/api/final-selects", methods=["GET", "POST"])
@jwt_required
def final_selects():
    user = request.current_user
    org_id = user.get('org_id')
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
    response = supabase.table("candidates").select("*").eq("hiring_status", True).eq("org_id", org_id).execute()
    data = response.data or []
    print(data)
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
@jwt_required
def remove_final_select():
    data = request.get_json()
    candidate_id = data.get("candidate_id")

    if not candidate_id:
        return jsonify({"error": "Candidate ID missing"}), 400

    # Update the hiring_status field to False
    supabase.table("candidates").update({"hiring_status": False}).eq("id", candidate_id).execute()

    return jsonify({"status": "success"})

# ─── Call Initiation Logic ───────────────────────────────────────────────────
def call_candidate(name, phone_number, skills, candidate_id, employer, hr, work_location, notice_period, custom_question, contracth, remotew, context):
    """
    Initiate a call via VAPI API
    Returns the call response with call_id for tracking
    """
    print("Call candidate works")
    print(f"Candidate ID {candidate_id}")
    
    headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}
    payload = {
        "assistantId": AGENT_ID,
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "assistantOverrides": {
            "variableValues": {
                "name": name,
                "skills": skills,
                "employer": employer,
                "hr": hr,
                "work_location": work_location,
                "days": notice_period,
                "candidate_id": str(candidate_id),
                "custom_question": custom_question,
                "contract_hiring": contracth,
                "remote_work": remotew,
                "context": context,
            }
        },
        "customer": {
            "number": f"+91{phone_number}",
            "name": name
        }
    }
    
    try:
        print("Making VAPI call...")
        resp = requests.post(VAPI_URL, headers=headers, json=payload)
        response_data = resp.json()
        print(response_data)
        
        # Return both status and call_id for tracking
        return {
            "name": name,
            "status": resp.status_code,
            "call_id": response_data.get("id"),  # VAPI returns call ID
            "phone": phone_number
        }
    except Exception as err:
        print(f"Error calling candidate: {err}")
        return {"name": name, "phone": phone_number, "error": str(err)}
    
def call_candidate_ringg(name, phone_number, skills, candidate_id, employer, hr, 
                        work_location, notice_period, custom_question, 
                        contracth, remotew, context):
    """
    Initiate a call via Ringg AI API
    """
    headers = {
        "X-API-KEY": RINGG_API_KEY,
        "Content-Type": "application/json",
    }

    # FIX: Changed "name" to "callee_name"
    payload = {
        "name": name, 
        "mobile_number": f"+91{phone_number}",
        "agent_id": RINGG_AGENT_ID,
        "from_number_id": RINGG_NUMBER_ID, 
        "custom_args_values": {
            "hr": hr,
            "employer": employer,
            "context": context or " ",
            "skills": skills,
            "days": notice_period,
            "work_location": work_location,
            "remote_work": remotew,
            "contract_hiring": contracth,
            "custom_question": custom_question,
            "candidate_id": str(candidate_id)
        },
        "call_config": {
            "idle_timeout_warning": 5,
            "idle_timeout_end": 10,
            "max_call_length": 240,
            "call_retry_config": {
                "retry_count": 3,
                "retry_busy": 30,
                "retry_not_picked": 30,
                "retry_failed": 30
            },
            "call_time": {
                "call_start_time": "12:00",
                "call_end_time": "23:59",
                "timezone": "Asia/Kolkata"
            }
        }
    }

    try:
        print(f"Initiating Ringg AI call for: {name} ({phone_number})")
        resp = requests.post(RINGG_URL, headers=headers, json=payload, timeout=30)
        print(resp)
        # LOGGING FOR DEBUGGING:
        if resp.status_code != 200:
            print(f"Ringg API Error Response: {resp.text}")
            
        resp.raise_for_status()
        response_data = resp.json()
        
        call_id = response_data.get("data", {}).get("Unique Call ID")
        return {
            "name": name,
            "status": "success",
            "call_id": call_id,
            "phone": phone_number
        }
    except Exception as err:
        print(f"Error calling candidate via Ringg: {err}")
        return {"name": name, "phone": phone_number, "error": str(err)}

def get_previous_call_context(candidate_id):
    """
    Returns extra context if the previous call for this candidate 
    was rescheduled and has a call summary.
    """
    try:
        prev_call = supabase.from_('calls') \
            .select('id, call_summary, status') \
            .eq('candidate_id', candidate_id) \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()
        
        if prev_call.data and prev_call.data[0]['status'] == 'Re-schedule':
            last_summary = prev_call.data[0].get('call_summary') or ""
            return (
                f"""This is a reschedule call and the below text is the call summary from the previous call. 
                Keep the context of the previous call and finish all remaining tasks which were not 
                completed earlier. Make sure to acknowledge that this is a rescheduled call, as requested by the user. 
                \n\nPrevious Call Summary:\n{last_summary}"""
            )
    except Exception as e:
        print("Error fetching previous call:", e)

    return "" 

def schedule_call(candidate_id, phone_number, candidate_name, reschedule_time_str, call_record_id, search_id):
    """
    Schedule a call at the specified time using APScheduler
    """
    try:
        # ✅ FIX: Validate inputs
        if not call_record_id:
            print(f"❌ Invalid call_record_id: {call_record_id}")
            return None
        
        if not reschedule_time_str:
            print(f"❌ Invalid reschedule_time_str: {reschedule_time_str}")
            return None
        
        # Parse the reschedule time
        reschedule_time = datetime.fromisoformat(reschedule_time_str)
        
        # Check if the time is in the past
        if reschedule_time <= datetime.now():
            print(f"⚠️ Reschedule time is in the past! Scheduling for 5 minutes from now instead.")
            reschedule_time = datetime.now() + timedelta(minutes=5)
        
        # Create a unique job ID
        job_id = f"call_{candidate_id}_{call_record_id}_{int(datetime.now().timestamp())}"
        
        print(f"⏰ Scheduling call:")
        print(f"   Candidate: {candidate_name} (ID: {candidate_id})")
        print(f"   Time: {reschedule_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Job ID: {job_id}")
        print(f"   Call Record ID: {call_record_id}")
        
        # Schedule the call
        scheduler.add_job(
            func=make_scheduled_vapi_call,
            trigger=DateTrigger(run_date=reschedule_time),
            args=[candidate_id, phone_number, candidate_name, search_id],
            id=job_id,
            name=f"Scheduled call for {candidate_name}",
            replace_existing=True
        )
        
        print(f"✅ Job added to scheduler")
        
        # Store the job_id in the database for tracking
        update_result = supabase.table("calls").update({
            "scheduled_job_id": job_id
        }).eq("id", call_record_id).execute()
        
        print(f"✅ Database updated with job_id")
        
        return job_id
        
    except ValueError as e:
        print(f"❌ Invalid datetime format: {e}")
        return None
    except Exception as e:
        print(f"❌ Error scheduling call: {e}")
        import traceback
        traceback.print_exc()
        return None

# ─── Schedule Ringg Call ─────────────────────────────────────────────────────
def schedule_ringg_call(candidate_id, phone_number, candidate_name, reschedule_time_str, call_record_id, search_id):
    """
    Schedule a Ringg AI call at the specified time using APScheduler
    """
    try:
        # Validate inputs
        if not call_record_id:
            print(f"❌ Invalid call_record_id: {call_record_id}")
            return None
        
        if not reschedule_time_str:
            print(f"❌ Invalid reschedule_time_str: {reschedule_time_str}")
            return None
        
        # Parse the reschedule time
        reschedule_time = datetime.fromisoformat(reschedule_time_str)
        
        # Check if the time is in the past
        if reschedule_time <= datetime.now():
            print(f"⚠️ Reschedule time is in the past! Scheduling for 5 minutes from now instead.")
            reschedule_time = datetime.now() + timedelta(minutes=5)
        
        # Create a unique job ID for Ringg calls
        job_id = f"ringg_call_{candidate_id}_{call_record_id}_{int(datetime.now().timestamp())}"
        
        print(f"⏰ Scheduling Ringg call:")
        print(f"   Candidate: {candidate_name} (ID: {candidate_id})")
        print(f"   Time: {reschedule_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Job ID: {job_id}")
        print(f"   Call Record ID: {call_record_id}")
        
        # Schedule the Ringg call
        scheduler.add_job(
            func=make_scheduled_ringg_call,
            trigger=DateTrigger(run_date=reschedule_time),
            args=[candidate_id, phone_number, candidate_name, search_id],
            id=job_id,
            name=f"Scheduled Ringg call for {candidate_name}",
            replace_existing=True
        )
        
        print(f"✅ Ringg job added to scheduler")
        
        # Store the job_id in the database for tracking
        update_result = supabase.table("calls").update({
            "scheduled_job_id": job_id
        }).eq("id", call_record_id).execute()
        
        print(f"✅ Database updated with Ringg job_id")
        
        return job_id
        
    except ValueError as e:
        print(f"❌ Invalid datetime format: {e}")
        return None
    except Exception as e:
        print(f"❌ Error scheduling Ringg call: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# ─── Make Scheduled VAPI Call ───────────────────────────────────────────────
def make_scheduled_vapi_call(candidate_id, phone_number, candidate_name, search_id):
    """
    Make an automated call using VAPI for scheduled/rescheduled calls
    This fetches all necessary data from the database
    """
    try:
        # Fetch search data
        da = supabase.from_('search') \
            .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id') \
            .eq('id', search_id) \
            .execute()
        
        if not da.data:
            print(f"❌ Search ID {search_id} not found")
            return None
        
        call_data = da.data[0]
        
        contracth = "Inform the candidate that this is a contract hiring." if call_data["contract_hiring"] else ""
        remotew = "This is a remote work." if call_data["remote_work"] else ""
        custom_question = call_data["custom_question"] or ""
        
        # Get previous call context
        context = get_previous_call_context(candidate_id)
        print(f"call context {context}")
        
        # Make the call using the existing call_candidate function
        result = call_candidate(
            name=candidate_name,
            phone_number=phone_number,
            skills=call_data["key_skills"],
            employer=call_data["rc_name"],
            hr=call_data["hc_name"],
            candidate_id=candidate_id,
            work_location=call_data["company_location"],
            notice_period=call_data["notice_period"],
            custom_question=custom_question,
            contracth=contracth,
            remotew=remotew,
            context=context
        )
        
        if result.get("status") == 201:
            print(f"✅ Scheduled call initiated successfully for {candidate_name}!")
            
            # Update candidate status
            supabase.table("candidates").update({
                "call_status": "Rescheduled Call In Progress"
            }).eq('id', candidate_id).execute()
            
            return result
        else:
            print(f"❌ Failed to initiate scheduled call: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Error making scheduled VAPI call: {e}")
        return None

# ─── Make Scheduled Ringg Call ───────────────────────────────────────────────
def make_scheduled_ringg_call(candidate_id, phone_number, candidate_name, search_id):
    """
    Make an automated call using Ringg AI for scheduled/rescheduled calls
    This fetches all necessary data from the database
    """
    try:
        print(f"\n{'='*80}")
        print(f"🔔 EXECUTING SCHEDULED RINGG CALL")
        print(f"{'='*80}")
        print(f"Candidate: {candidate_name} (ID: {candidate_id})")
        print(f"Phone: {phone_number}")
        print(f"Search ID: {search_id}")
        
        # Fetch search data
        da = supabase.from_('search') \
            .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id') \
            .eq('id', search_id) \
            .execute()
        
        if not da.data:
            print(f"❌ Search ID {search_id} not found")
            return None
        
        call_data = da.data[0]
        
        contracth = "Inform the candidate that this is a contract hiring." if call_data["contract_hiring"] else ""
        remotew = (
            "This is a remote work."
            if call_data["remote_work"]
            else "7.⁠ ⁠Ask for the candidate's current location and preferred location. "
                 "8.⁠ ⁠If the candidate's current location is not the same as {{work_location}}, then ask if the candidate is ready to relocate, else skip this question."
        )
        custom_question = call_data["custom_question"] or ""
        
        # Get previous call context
        context = get_previous_call_context(candidate_id)
        print(f"📄 Call context: {context[:100] if context else 'None'}...")
        
        # Make the call using the Ringg AI function
        result = call_candidate_ringg(
            name=candidate_name,
            phone_number=phone_number,
            skills=call_data["key_skills"],
            employer=call_data["rc_name"],
            hr=call_data["hc_name"],
            candidate_id=candidate_id,
            work_location=call_data["company_location"],
            notice_period=call_data["notice_period"],
            custom_question=custom_question,
            contracth=contracth,
            remotew=remotew,
            context=context
        )
        
        if result.get("status") == "success":
            print(f"✅ Scheduled Ringg call initiated successfully for {candidate_name}!")
            print(f"   Call ID: {result.get('call_id')}")
            
            # Update candidate status
            supabase.table("candidates").update({
                "call_status": "Rescheduled Call In Progress"
            }).eq('id', candidate_id).execute()
            
            print(f"{'='*80}\n")
            return result
        else:
            print(f"❌ Failed to initiate scheduled Ringg call: {result}")
            print(f"{'='*80}\n")
            return None
            
    except Exception as e:
        print(f"❌ Error making scheduled Ringg call: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        return None
    
@app.route('/api/call-single', methods=['POST'])
@jwt_required
def call_single():
    data = request.get_json()
    print("works")
    print(data)
    if not data or 'name' not in data or 'phone' not in data:
        return jsonify({'error': 'Missing candidate data'}), 400

    search_id = data['search_id']
    name = data['name']
    phone = data['phone']
    skills = data.get('skills', [])
    company = data.get('company', '')
    candidate_id = data.get('candidate_id')

    # ✅ Fetch search details
    data = supabase.from_('search') \
        .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id') \
        .eq('id', search_id) \
        .execute()

    call_data = data.data[0]

    # ✅ Keep your existing logic
    contracth = "Inform the candidate that this is a contract hiring." if call_data["contract_hiring"] else ""
    remotew = (
        "This is a remote work."
        if call_data["remote_work"]
        else "7.⁠ ⁠Ask for the candidate's current location and preferred location. "
             "8.⁠ ⁠If the candidate's current location is not the same as {{work_location}}, then ask if the candidate is ready to relocate, else skip this question."
    )

    custom_question = call_data["custom_question"] if call_data["custom_question"] else ""
    context = get_previous_call_context(candidate_id)
    # ✅ Call the agent as usual (but now with `context`)

    result = call_candidate_ringg(
        name=name,
        phone_number=phone,
        skills=call_data["key_skills"],
        employer=call_data["rc_name"],
        hr=call_data["hc_name"],
        candidate_id=candidate_id,
        work_location=call_data["company_location"],
        notice_period=call_data["notice_period"],
        custom_question=custom_question,
        contracth=contracth,
        remotew=remotew,
        context=context
    )
    # result = call_candidate(
    #     name=name,
    #     phone_number=phone,
    #     skills=call_data["key_skills"],
    #     employer=call_data["rc_name"],
    #     hr=call_data["hc_name"],
    #     candidate_id=candidate_id,
    #     work_location=call_data["company_location"],
    #     notice_period=call_data["notice_period"],
    #     custom_question=custom_question,
    #     contracth=contracth,
    #     remotew=remotew,
    #     context=context
    # )

    return jsonify({'status': 'success', 'result': result}), 200

@app.route("/api/call", methods=["POST"])
@jwt_required
def initiate_call():
    data = request.get_json()
    candidates = data.get("candidates", [])
    search_id = data.get("search_id")

    results = []

    da = supabase.from_('search') \
        .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id') \
        .eq('id', search_id) \
        .execute()

    call_data = da.data[0]
    
    contracth = "Inform the candidate that this is a contract hiring." if call_data["contract_hiring"] else ""
    remotew = "This is a remote work." if call_data["remote_work"] else ""
    custom_question = call_data["custom_question"] or ""
    
    for c in candidates:
        name = c.get("name")
        phone = c.get("phone")
        candidate_id = c.get("candidate_id")

        # ✅ Reuse previous call context logic for each candidate
        context = get_previous_call_context(candidate_id)

        res = call_candidate_ringg(
        name=data['name'],
        phone_number=data['phone'],
        skills=call_data["key_skills"],
        employer=call_data["rc_name"],
        hr=call_data["hc_name"],
        candidate_id=data.get('candidate_id'),
        work_location=call_data["company_location"],
        notice_period=call_data["notice_period"],
        custom_question=custom_question,
        contracth=contracth,
        remotew=remotew,
        context=context
        )
        # res = call_candidate(
        #     name=name,
        #     phone_number=phone,
        #     skills=call_data["key_skills"],
        #     employer=call_data["rc_name"],
        #     hr=call_data["hc_name"],
        #     candidate_id=candidate_id,
        #     work_location=call_data["company_location"],
        #     notice_period=call_data["notice_period"],
        #     custom_question=custom_question,
        #     contracth=contracth,
        #     remotew=remotew,
        #     context=context
        # )
        results.append(res)

    return jsonify({"message": "Calls initiated", "results": results}), 200

def add_call_data(transcript, summary, structuredData, call_status, success_eval, 
                  phone, durationMinutes, name, candidate_id,org_id, reschedule_time=None, 
                  reschedule_status=None):
    """Add call data to the calls table and return the record ID"""
    
    try:
        result = supabase.table("calls").insert({
            "candidate_id": candidate_id,
            "transcript": transcript,
            "call_summary": summary,
            "structured_call_data": json.dumps(structuredData),
            "status": call_status,
            "evaluation": success_eval,
            "name": name,
            "phone": phone,
            "call_duration": float(durationMinutes),
            "reschedule_time": reschedule_time,
            "reschedule_status": reschedule_status,
            "org_id":org_id
        }).execute()
        
        # ✅ FIX: Properly extract and return the ID
        if result.data and len(result.data) > 0:
            call_record_id = result.data[0]["id"]
            print(f"✅ Added call data (ID: {call_record_id}, reschedule_time: {reschedule_time})")
        else:
            print("❌ No data returned from insert operation")
            call_record_id = None
        
        # Update candidate status
        supabase.table("candidates").update({
            "call_status": call_status
        }).eq('id', candidate_id).execute()
        
        return call_record_id  # ✅ Return the ID
        
    except Exception as e:
        print(f"❌ Error in add_call_data: {e}")
        return None  # Return None on error

# ─── Updated Webhook Function ───────────────────────────────────────────────
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    print(data)
    message = data.get("message", data)
    end_call_status = message.get("endedReason", "")
    call = message.get("call", {})
    assistantOverrides = call.get("assistantOverrides", {})
    variableValues = assistantOverrides.get("variableValues", {})
    candidate_id = variableValues.get("candidate_id")
    
    # Lookup candidate
    candidate_res = supabase.table("candidates").select("id, search_id, org_id, call_status").eq("id", candidate_id).execute()
    if not candidate_res.data:
        return jsonify({"error": "Candidate not found"}), 404
    
    search_id = candidate_res.data[0]["search_id"]
    org_id = candidate_res.data[0]["org_id"]
    previous_call_status = candidate_res.data[0].get("call_status")
    
    search_res = supabase.table("search").select("user_id").eq("id", search_id).execute()
    if not search_res.data:
        return jsonify({"error": "Search not found"}), 404
    user_id = search_res.data[0]["user_id"]
    
    customer = call.get("customer", {})
    name = customer.get("name", "")
    phone = customer.get("number", "")
    phone_int = int(phone[-10:]) if phone else None
    transcript = message.get("transcript", "")
    analysis = message.get("analysis", {})
    summary = analysis.get("summary", "")
    structuredData = analysis.get("structuredData", {})
    status = structuredData.get("re-schedule", "")
    success_eval = analysis.get("successEvaluation", "")
    durationMinutes = message.get("durationMinutes", "")
    call_status = None
    
    # Check for pending reschedules
    previous_reschedule = supabase.table("calls")\
        .select("id, reschedule_status, scheduled_job_id")\
        .eq("candidate_id", candidate_id)\
        .eq("reschedule_status", "pending")\
        .order("created_at", desc=True)\
        .limit(1)\
        .execute()
    
    if end_call_status == "customer-did-not-answer":
        call_status = "Not Answered"
        supabase.table("candidates").update({
            "call_status": call_status
        }).eq('id', candidate_id).execute()
        
    elif status == "yes":
        # Candidate wants to reschedule
        call_status = "Re-schedule"
        raw_reschedule_time = structuredData.get("reschedule_time", "")
        reschedule_time = parse_reschedule_time(raw_reschedule_time)
        
        print(f"📅 Reschedule requested: '{raw_reschedule_time}' -> {reschedule_time}")
        
        # If there's a previous pending reschedule, cancel it
        if previous_reschedule.data and len(previous_reschedule.data) > 0:
            old_job_id = previous_reschedule.data[0].get("scheduled_job_id")
            if old_job_id:
                try:
                    scheduler.remove_job(old_job_id)
                    print(f"🗑️ Cancelled previous scheduled call: {old_job_id}")
                except Exception as e:
                    print(f"⚠️ Could not cancel previous job: {e}")
            
            supabase.table("calls").update({
                "reschedule_status": "completed_but_rescheduled_again"
            }).eq("id", previous_reschedule.data[0]["id"]).execute()
            print(f"✅ Updated previous reschedule status")
        
        # Add call data first to get the call record ID
        call_record_id = add_call_data(
            transcript, summary, structuredData, call_status, 
            success_eval, phone_int, durationMinutes, name, 
            candidate_id=int(candidate_id),
            org_id=org_id,
            reschedule_time=reschedule_time,
            reschedule_status="pending"
        )
        
        # ✅ FIX: Check if call_record_id is valid before scheduling
        if not call_record_id:
            print("❌ Failed to get call_record_id, cannot schedule call")
            deduct_credits(user_id, org_id, "rescheduled_call", reference_id=None)
            return jsonify({"status": "error", "message": "Failed to save call data"}), 500
        
        print(f"✅ Call record created with ID: {call_record_id}")
        
        # Schedule the automated call
        if reschedule_time and phone:
            job_id = schedule_call(
                candidate_id=int(candidate_id),
                phone_number=str(phone_int),  # Use cleaned phone number
                candidate_name=name,
                reschedule_time_str=reschedule_time,
                call_record_id=call_record_id,  # ✅ Now guaranteed to be valid
                search_id=search_id
            )
            
            if job_id:
                print(f"✅ Automated call scheduled successfully! Job ID: {job_id}")
            else:
                print(f"⚠️ Failed to schedule automated call")
        else:
            print(f"⚠️ Missing reschedule_time or phone, cannot schedule call")
        
        deduct_credits(user_id, org_id, "rescheduled_call", reference_id=None)
        
    elif status == "no":
        # Call completed successfully
        call_status = "Called & Answered"
        
        # Cancel any pending scheduled calls
        if previous_reschedule.data and len(previous_reschedule.data) > 0:
            old_job_id = previous_reschedule.data[0].get("scheduled_job_id")
            if old_job_id:
                try:
                    scheduler.remove_job(old_job_id)
                    print(f"🗑️ Cancelled scheduled call (completed): {old_job_id}")
                except Exception as e:
                    print(f"⚠️ Could not cancel job: {e}")
            
            supabase.table("calls").update({
                "reschedule_status": "completed"
            }).eq("id", previous_reschedule.data[0]["id"]).execute()
            print(f"✅ Updated previous reschedule status to 'completed'")
        
        add_call_data(
            transcript, summary, structuredData, call_status, 
            success_eval, phone_int, durationMinutes, name, 
            candidate_id=int(candidate_id),
            org_id=org_id,
            reschedule_time=None,
            reschedule_status=None
        )
        deduct_credits(user_id, org_id, "ai_call", reference_id=None)
    
    print(f"✅ Webhook processed successfully")
    return jsonify({"status": "received"}), 200

@app.route('/ring/webhook', methods=['POST'])
def ringg_webhook():
    """
    Ringg AI webhook handler - processes call events and stores data
    Similar to VAPI webhook implementation
    """
    data = request.get_json()
    
    if not data:
        print("❌ Received webhook with no JSON body")
        return jsonify({"error": "No data received"}), 400

    print("=" * 80)
    print("📞 RINGG WEBHOOK RECEIVED")
    print("=" * 80)
    print(json.dumps(data, indent=2))
    
    # Extract common fields
    event_type = data.get("event_type")
    call_id = data.get("call_id")
    custom_args = data.get("custom_args_values", {})
    candidate_id = custom_args.get("candidate_id")
    
    print(f"\n🔔 Event Type: {event_type}")
    print(f"📞 Call ID: {call_id}")
    print(f"👤 Candidate ID: {candidate_id}")
    
    if not candidate_id:
        print("⚠️ No candidate_id found in webhook data")
        return jsonify({"status": "received", "warning": "No candidate_id"}), 200
    
    # Lookup candidate to get search_id, org_id, and user_id
    try:
        candidate_res = supabase.table("candidates")\
            .select("id, search_id, org_id, call_status")\
            .eq("id", candidate_id)\
            .execute()
        
        if not candidate_res.data:
            print(f"❌ Candidate {candidate_id} not found")
            return jsonify({"error": "Candidate not found"}), 404
        
        search_id = candidate_res.data[0]["search_id"]
        org_id = candidate_res.data[0]["org_id"]
        previous_call_status = candidate_res.data[0].get("call_status")
        
        # Get user_id from search
        search_res = supabase.table("search")\
            .select("user_id")\
            .eq("id", search_id)\
            .execute()
        
        if not search_res.data:
            print(f"❌ Search {search_id} not found")
            return jsonify({"error": "Search not found"}), 404
        
        user_id = search_res.data[0]["user_id"]
        
        print(f"✅ Found candidate - Search ID: {search_id}, Org ID: {org_id}, User ID: {user_id}")
        
    except Exception as e:
        print(f"❌ Error looking up candidate: {e}")
        return jsonify({"error": "Database lookup failed"}), 500
    
    # ─── Handle Different Event Types ───────────────────────────────────────
    
    # 1. CALL COMPLETED - Initial call data storage
    if event_type == 'call_completed':
        print("\n📞 Processing call_completed event...")
        
        # Extract call details
        duration_sec = data.get("call_duration", 0)
        duration_minutes = duration_sec / 60 if duration_sec else 0
        
        # Extract phone number
        to_number = data.get("to_number", "")
        phone_int = int(to_number[-10:]) if len(to_number) >= 10 else 0
        
        # Extract name  
        callee_name = custom_args.get("callee_name", "Unknown")
        
        # Extract transcript
        transcript_list = data.get("transcript", [])
        
        # Status based on call completion
        call_status = data.get("status", "completed")
        
        print(f"   Duration: {duration_minutes:.2f} minutes")
        print(f"   Phone: {phone_int}")
        print(f"   Name: {callee_name}")
        print(f"   Status: {call_status}")
        print(f"   Transcript items: {len(transcript_list)}")
        
        # Store initial call data (will be updated with analysis later)
        # We'll update this when we get the analysis events
        print("   ℹ️ Call completed, waiting for analysis events...")
        
    # 2. PLATFORM ANALYSIS COMPLETED - Ringg's AI analysis
    elif event_type == 'platform_analysis_completed':
        print("\n🤖 Processing platform_analysis_completed event...")
        
        analysis_data = data.get("analysis_data", {})
        
        # Extract analysis fields
        summary = analysis_data.get("summary", "")
        classification = analysis_data.get("classification", "")
        callback_requested = analysis_data.get("callback_requested", False)
        callback_time_str = analysis_data.get("callback_requested_time", "")
        key_points = analysis_data.get("key_points", [])
        action_items = analysis_data.get("action_items", [])
        
        print(f"   Summary: {summary[:100]}...")
        print(f"   Classification: {classification}")
        print(f"   Callback requested: {callback_requested}")
        print(f"   Callback time: {callback_time_str}")
        
        # Check for pending reschedules
        previous_reschedule = supabase.table("calls")\
            .select("id, reschedule_status, scheduled_job_id")\
            .eq("candidate_id", candidate_id)\
            .eq("reschedule_status", "pending")\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        # Determine if this is a reschedule
        is_reschedule = classification == "reschedule_requested" or callback_requested
        
        if is_reschedule and callback_time_str:
            print("\n📅 RESCHEDULE DETECTED!")
            call_status = "Re-schedule"
            
            # Parse the reschedule time
            reschedule_time = parse_reschedule_time(callback_time_str)
            print(f"   Parsed reschedule time: {reschedule_time}")
            
            # Cancel any previous pending reschedule
            if previous_reschedule.data and len(previous_reschedule.data) > 0:
                old_job_id = previous_reschedule.data[0].get("scheduled_job_id")
                if old_job_id:
                    try:
                        scheduler.remove_job(old_job_id)
                        print(f"   🗑️ Cancelled previous scheduled call: {old_job_id}")
                    except Exception as e:
                        print(f"   ⚠️ Could not cancel previous job: {e}")
                
                supabase.table("calls").update({
                    "reschedule_status": "completed_but_rescheduled_again"
                }).eq("id", previous_reschedule.data[0]["id"]).execute()
            
            # Extract call data
            duration_sec = data.get("call_duration", 0)
            duration_minutes = duration_sec / 60 if duration_sec else 0
            to_number = data.get("to_number", "")
            phone_int = int(to_number[-10:]) if len(to_number) >= 10 else 0
            callee_name = custom_args.get("callee_name", "Unknown")
            
            # Format transcript
            transcript_list = data.get("transcript", [])
            
            # Create structured data
            structured_data = {
                "classification": classification,
                "callback_requested": callback_requested,
                "callback_time": callback_time_str,
                "key_points": key_points,
                "action_items": action_items
            }
            
            # Add call data to database
            call_record_id = add_call_data(
                transcript=json.dumps(transcript_list),
                summary=summary,
                structuredData=structured_data,
                call_status=call_status,
                success_eval="Reschedule requested",
                phone=phone_int,
                durationMinutes=duration_minutes,
                name=callee_name,
                candidate_id=int(candidate_id),
                org_id=org_id,
                reschedule_time=reschedule_time,
                reschedule_status="pending"
            )
            
            if not call_record_id:
                print("   ❌ Failed to save call data")
                return jsonify({"status": "error"}), 500
            
            print(f"   ✅ Call record created: {call_record_id}")
            
            # Schedule the automated callback
            if reschedule_time:
                job_id = schedule_ringg_call(
                    candidate_id=int(candidate_id),
                    phone_number=str(phone_int),
                    candidate_name=callee_name,
                    reschedule_time_str=reschedule_time,
                    call_record_id=call_record_id,
                    search_id=search_id
                )
                
                if job_id:
                    print(f"   ✅ Scheduled Ringg callback: {job_id}")
                else:
                    print(f"   ⚠️ Failed to schedule callback")
            
            # Deduct credits
            deduct_credits(user_id, org_id, "rescheduled_call", reference_id=None)
            
        else:
            # Call completed successfully without reschedule
            print("\n✅ Call completed successfully (no reschedule)")
            call_status = "Called & Answered"
            
            # Cancel any pending reschedules
            if previous_reschedule.data and len(previous_reschedule.data) > 0:
                old_job_id = previous_reschedule.data[0].get("scheduled_job_id")
                if old_job_id:
                    try:
                        scheduler.remove_job(old_job_id)
                        print(f"   🗑️ Cancelled pending reschedule: {old_job_id}")
                    except Exception as e:
                        print(f"   ⚠️ Could not cancel job: {e}")
                
                supabase.table("calls").update({
                    "reschedule_status": "completed"
                }).eq("id", previous_reschedule.data[0]["id"]).execute()
            
            # Extract call data
            duration_sec = data.get("call_duration", 0)
            duration_minutes = duration_sec / 60 if duration_sec else 0
            to_number = data.get("to_number", "")
            phone_int = int(to_number[-10:]) if len(to_number) >= 10 else 0
            callee_name = custom_args.get("callee_name", "Unknown")
            
            # Format transcript
            transcript_list = data.get("transcript", [])
            
            # Create structured data
            structured_data = {
                "classification": classification,
                "key_points": key_points,
                "action_items": action_items
            }
            
            # Add call data to database
            add_call_data(
                transcript=json.dumps(transcript_list),
                summary=summary,
                structuredData=structured_data,
                call_status=call_status,
                success_eval="Call completed successfully",
                phone=phone_int,
                durationMinutes=duration_minutes,
                name=callee_name,
                candidate_id=int(candidate_id),
                org_id=org_id,
                reschedule_time=None,
                reschedule_status=None
            )
            
            # Deduct credits
            deduct_credits(user_id, org_id, "ai_call", reference_id=None)
    
    # 3. CLIENT ANALYSIS COMPLETED - Custom analysis based on your prompts
    elif event_type == 'client_analysis_completed':
        print("\n📊 Processing client_analysis_completed event...")
        
        analysis_data = data.get("analysis_data", {})
        
        # Extract the custom fields from your agent
        reschedule = analysis_data.get("reschedule", False)
        reschedule_time_str = analysis_data.get("reschedule_time", "")
        interested = analysis_data.get("intrested", False)  # Note: typo in API
        total_exp = analysis_data.get("total_exp", "")
        relevant_exp = analysis_data.get("relevant_exp", "")
        current_ctc = analysis_data.get("current_ctc", "")
        expected_ctc = analysis_data.get("expected_ctc", "")
        notice_period = analysis_data.get("notice_period", "")
        current_loc = analysis_data.get("current_loc", "")
        expected_loc = analysis_data.get("expected_loc", "")
        relocate = analysis_data.get("relocate", False)
        remote = analysis_data.get("remote", False)
        familiar = analysis_data.get("familiar", False)
        
        print(f"   Interested: {interested}")
        print(f"   Reschedule: {reschedule}")
        print(f"   Total Experience: {total_exp}")
        print(f"   Relevant Experience: {relevant_exp}")
        
        # Update candidate with extracted information
        update_data = {}
        
        if total_exp:
            update_data["total_experience"] = total_exp
        if relevant_exp:
            update_data["relevant_work_experience"] = relevant_exp
        
        if update_data:
            supabase.table("candidates").update(update_data)\
                .eq("id", candidate_id)\
                .execute()
            print(f"   ✅ Updated candidate with extracted data")
        
        # Store this analysis in the call record
        # Find the most recent call for this candidate
        recent_call = supabase.table("calls")\
            .select("id")\
            .eq("candidate_id", candidate_id)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        if recent_call.data:
            # Update with client analysis data
            call_id_to_update = recent_call.data[0]["id"]
            
            # Merge with existing structured data
            existing_call = supabase.table("calls")\
                .select("structured_call_data")\
                .eq("id", call_id_to_update)\
                .execute()
            
            if existing_call.data:
                existing_structured = json.loads(existing_call.data[0].get("structured_call_data", "{}"))
                existing_structured["client_analysis"] = analysis_data
                
                supabase.table("calls").update({
                    "structured_call_data": json.dumps(existing_structured)
                }).eq("id", call_id_to_update).execute()
                
                print(f"   ✅ Updated call record with client analysis")
    
    else:
        print(f"\n⚠️ Unknown event type: {event_type}")
    
    print("=" * 80)
    print("✅ Webhook processed successfully")
    print("=" * 80)
    
    return jsonify({"status": "received"}), 200

@app.route("/api/transcript/<int:candidate_id>")
@jwt_required
def api_transcript(candidate_id):
    try:
        import json

        # 1. Fetch candidate
        candidate_res = supabase.table("candidates").select("*").eq("id", candidate_id).execute()
        if not candidate_res.data:
            return jsonify({"error": "Candidate not found"}), 404
        candidate = candidate_res.data[0]

        # 2. Fetch ALL calls for the candidate
        call_res = supabase.table("calls").select("*").eq("candidate_id", candidate_id).order("id", desc=False).execute()
        all_calls_raw = call_res.data or []
        
        processed_calls = []

        for call in all_calls_raw:
            # Parse structured_call_data
            structured_data = call.get("structured_call_data")
            if isinstance(structured_data, str):
                try:
                    structured_data = json.loads(structured_data)
                except json.JSONDecodeError:
                    structured_data = {}

            # Parse transcript - Handle both formats
            transcript_raw = call.get("transcript", "")
            transcript = []
            
            # Try to parse as JSON array first (new format)
            if isinstance(transcript_raw, str):
                try:
                    parsed = json.loads(transcript_raw)
                    if isinstance(parsed, list):
                        # New format: array of objects with 'bot' and 'user' keys
                        for item in parsed:
                            if isinstance(item, dict):
                                if 'bot' in item and item['bot']:
                                    transcript.append({"speaker": "ai", "message": item['bot'], "timestamp": ""})
                                if 'user' in item and item['user']:
                                    transcript.append({"speaker": "candidate", "message": item['user'], "timestamp": ""})
                    else:
                        # If it's not a list, fall back to text parsing
                        raise ValueError("Not a list")
                except (json.JSONDecodeError, ValueError):
                    # Old format: multi-line text with AI:/User: prefixes
                    for line in transcript_raw.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("AI:"):
                            transcript.append({"speaker": "ai", "message": line[3:].strip(), "timestamp": ""})
                        elif line.startswith("User:") or line.startswith("You:"):
                            transcript.append({"speaker": "candidate", "message": line.split(":", 1)[1].strip(), "timestamp": ""})
            elif isinstance(transcript_raw, list):
                # Already parsed as list
                for item in transcript_raw:
                    if isinstance(item, dict):
                        if 'bot' in item and item['bot']:
                            transcript.append({"speaker": "ai", "message": item['bot'], "timestamp": ""})
                        if 'user' in item and item['user']:
                            transcript.append({"speaker": "candidate", "message": item['user'], "timestamp": ""})

            # Parse evaluation
            eval_raw = call.get("evaluation")
            evaluation = {}
            if eval_raw:
                try:
                    evaluation = json.loads(eval_raw)
                except:
                    evaluation = {"summary": eval_raw}

            # Append the structured call data to the list
            processed_calls.append({
                "id": call.get("id"),
                "transcript": transcript,
                "evaluation": evaluation,
                "structured": structured_data,
                "call": {
                    "duration": call.get("call_duration"),
                    "status": call.get("status"),
                    "summary": call.get("call_summary", ""),
                    "timestamp": call.get("created_at")
                }
            })

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
            "calls": processed_calls,
        })

    except Exception as e:
        print("Transcript API error:", e)
        return jsonify({"error": "Could not fetch transcript"}), 500

@app.route("/api/dashboard", methods=["GET"])
@jwt_required
def dashboard():
    try:
        user = request.current_user 
        user_id = user['user_id']
        org_id = user['org_id']
        print(f"Dashboard access granted for user_id: {user_id}")

        dashboard_data = get_dashboard_data(org_id)

        # Add extra simple lookups
        dashboard_data["creds_used"] = get_total_credits_used(user_id)
        dashboard_data["user_name"] = get_user_name(user_id)

        return make_response(jsonify(dashboard_data))
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ------ to fetch and render search table data in frontend -------------------------------

@app.route("/api/searches", methods=["GET"])
@jwt_required
def get_searches():
    """Get user searches with JWT authentication"""
    try:
        # Get user info from JWT token (set by @jwt_required decorator)
        user = request.current_user
        user_id = user['user_id']
        
        print(f"Fetching searches for user_id: {user_id}")

        response = supabase.table("search").select("*").eq("user_id", user_id).order("id", desc=True).execute()
        return jsonify(response.data)
        
    except Exception as e:
        print(f"Get searches error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/get-liked-candiates", methods=["GET"])
@jwt_required
def get_liked_candidates():
    user = request.current_user
    org_id = user.get('org_id')
    response = supabase.table("candidates") \
        .select("*") \
        .eq("liked", True).eq("org_id",org_id) \
        .execute()

    candidates = response.data or []
    return jsonify(candidates)

@app.route("/api/add-user", methods=["POST"])
@jwt_required
def add_user():
    """Add a new user to the organisation (Admin/Master only)"""
    try:
        # Get current user info
        current_user = request.current_user
        if not current_user:
            return jsonify({"success": False, "message": "Authentication required"}), 401
        
        # Check if current user has admin privileges
        current_user_data = supabase.table("users").select("role").eq("id", current_user['user_id']).execute()
        if not current_user_data.data:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        current_user_role = current_user_data.data[0]["role"]
        if current_user_role not in ['admin', 'master']:
            return jsonify({"success": False, "message": "Insufficient permissions"}), 403
        
        # Get form data
        data = request.get_json()
        name = data.get("name", "").strip()
        email = data.get("email", "").lower().strip()
        password = data.get("password", "")
        role = data.get("role", "").strip()
        
        # Validate input
        if not all([name, email, password, role]):
            return jsonify({"success": False, "message": "All fields are required"}), 400
        
        if role not in ['user', 'admin']:
            return jsonify({"success": False, "message": "Invalid role"}), 400
        
        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
        
        # Check if user already exists in our database
        existing_user = supabase.table("users").select("id").eq("email", email).execute()
        if existing_user.data:
            return jsonify({"success": False, "message": "User with this email already exists"}), 409
        
        # Create user in Supabase Auth using admin client
        try:
            # Use the admin client for creating users
            auth_response = supabase_admin.auth.admin.create_user({
                "email": email,
                "password": password,
                "email_confirm": True,  # Auto-confirm email
                "user_metadata": {
                    "name": name,
                    "role": role
                }
            })
            
            if not auth_response or not auth_response.user:
                return jsonify({"success": False, "message": "Failed to create user in authentication system"}), 500
                
            supabase_user_id = auth_response.user.id
            
        except Exception as auth_error:
            print(f"Supabase Auth error: {str(auth_error)}")
            # Handle specific Supabase errors
            error_message = str(auth_error)
            if "email_address_not_authorized" in error_message:
                return jsonify({"success": False, "message": "Email domain not authorized"}), 400
            elif "email_address_invalid" in error_message:
                return jsonify({"success": False, "message": "Invalid email address"}), 400
            elif "password" in error_message.lower():
                return jsonify({"success": False, "message": "Password does not meet requirements"}), 400
            else:
                return jsonify({"success": False, "message": "Failed to create user authentication"}), 500
        
        # Insert user into users table
        try:
            user_insert_data = {
                # "id": supabase_user_id,
                "email": email,
                "name": name,
                "role": role,
                "org_id": current_user['org_id'],  # Same organisation as current user
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            insert_result = supabase.table("users").insert(user_insert_data).execute()
            
            if not insert_result.data:
                # If database insert fails, clean up the auth user
                try:
                    supabase_admin.auth.admin.delete_user(supabase_user_id)
                except Exception as cleanup_error:
                    print(f"Failed to cleanup auth user: {cleanup_error}")
                return jsonify({"success": False, "message": "Failed to create user record"}), 500
            
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            # Clean up auth user if database insert fails
            try:
                supabase_admin.auth.admin.delete_user(supabase_user_id)
            except Exception as cleanup_error:
                print(f"Failed to cleanup auth user: {cleanup_error}")
            return jsonify({"success": False, "message": "Failed to create user record"}), 500
        
        return jsonify({
            "success": True,
            "message": f"User {name} created successfully",
            "user": {
                "id": supabase_user_id,
                "name": name,
                "email": email,
                "role": role,
                "org_id": current_user['org_id']
            }
        })
        
    except Exception as e:
        print(f"Add user error: {str(e)}")
        return jsonify({"success": False, "message": "Failed to create user"}), 500
    
@app.route("/api/user-profile", methods=["GET"])
@jwt_required
def get_user_profile():
    user = request.current_user
    user_id = user['user_id']
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    response = supabase.table("users").select("*").eq("id", user_id).single().execute()
    user = response.data
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "name": user.get("name"),
        "email": user.get("email"),
        "organisation": user.get("organisation"),
        "creds": user.get("creds"),
        "id": user.get("id"),
        "role":user.get("role")
    })

@app.route("/api/billing-data", methods=["GET"])
@jwt_required
def get_billing_data():
    """Get billing data with JWT authentication"""
    # Get user info from JWT token
    user = request.current_user
    user_id = user['user_id']

    try:
        print(f"Fetching billing data for user_id: {user_id}")
        
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

        billing_data = {
            "current_credits": current_credits,
            "credit_logs": credit_logs,
            "payment_history": payment_history,
            "start_credits": start_creds,
            "total_deductions": total_deductions
        }
        
        print(f"Billing data retrieved: Credits {current_credits}, Logs {len(credit_logs)}, Payments {len(payment_history)}")
        
        return jsonify(billing_data)

    except Exception as e:
        print(f"Billing API error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/api/settings", methods=["GET", "POST"])
@jwt_required
def user_settings():
    """Get and update user settings with JWT authentication"""
    # Get user info from JWT token
    user = request.current_user
    user_id = user['user_id']

    if request.method == "GET":
        try:
            print(f"Fetching settings for user_id: {user_id}")
            
            response = supabase.table("account_preferences") \
                .select("dark_theme, credit_warnings, weekly_reports, email_alerts") \
                .eq("user_id", user_id) \
                .limit(1) \
                .execute()

            data = response.data[0] if response.data else {}

            settings = {
                "creditWarnings": data.get("credit_warnings", True),
                "weeklyReports": data.get("weekly_reports", False),
                "emailAlerts": data.get("email_alerts", True),
                "darkTheme": data.get("dark_theme", False)
            }
            
            print(f"Retrieved settings: {settings}")
            return jsonify(settings)

        except Exception as e:
            print(f"Error fetching settings: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    elif request.method == "POST":
        try:
            data = request.get_json()
            print(f"Updating settings for user_id: {user_id} with data: {data}")

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
                print(f"Updating existing preferences with ID: {pref_id}")
                supabase.table("account_preferences").update(update_data).eq("id", pref_id).execute()
            else:
                print("Creating new preferences record")
                supabase.table("account_preferences").insert(update_data).execute()

            return jsonify({
                "success": True,
                "message": "Settings updated successfully"
            })

        except Exception as e:
            print(f"Error updating settings: {str(e)}")
            return jsonify({"error": "Failed to update settings"}), 500

@app.route("/api/create-task", methods=["POST"])
@jwt_required
def create_task():
    try:
        user = request.current_user
        user_id = user["user_id"]
        org_id = user.get("org_id")

        body = request.get_json()
        
        # Handle assignedTo field properly
        assigned_to_id = body.get("assignedTo")        

        if assigned_to_id and assigned_to_id.strip():
            try:
                assigned_to_id = int(assigned_to_id)
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": "Invalid assigned user ID format"}), 400
        else:
            return jsonify({"success": False, "error": "Please select a user to assign the task to"}), 400
        
        # Handle deadline - set to end of day to avoid constraint issues
        deadline_str = body.get("deadline")
        if not deadline_str:
            return jsonify({"success": False, "error": "Deadline is required"}), 400
        
        # If it's just a date (YYYY-MM-DD), convert to end of day
        if len(deadline_str) == 10:  # Format: YYYY-MM-DD
            deadline = deadline_str + "T23:59:59.999Z"  # End of day
        else:
            deadline = deadline_str
        
        assignor=supabase.table("users").select("name").eq("id",user_id).execute().data[0]["name"]
        assignee=supabase.table("users").select("name").eq("id",assigned_to_id).execute().data[0]["name"]

        new_task = {
            "title": body.get("title"),
            "priority": body.get("priority", "Medium"),
            "deadline": deadline,
            "company_name": body.get("companyName"),
            "job_location": body.get("jobLocation"),
            "manager_email": body.get("managerEmail"),
            "job_role": body.get("jobRole"),
            "openings": body.get("openings", 1),
            "ctc_range": body.get("ctcRange"),
            "time_to_hire": body.get("timeToHire"),
            "skills": body.get("skills", []),
            "jd_link": body.get("jdLink"),
            "notes": body.get("notes"),
            "assigned_to_user_id": assigned_to_id,
            "assigned_by_user_id": user_id,
            "assignee": assignee,
            "assignor": assignor,
            "org_id": org_id,
        }

        response = supabase.table("tasks").insert(new_task).execute()
        if hasattr(response, 'error') and response.error:
            return jsonify({"success": False, "error": str(response.error)}), 400

        return jsonify({"success": True, "task": response.data[0]}), 201

    except Exception as e:
        print(f"Create task error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/tasks/history", methods=["GET"])
@jwt_required
def get_history_data():
    """
    Get all tasks assigned to the current user (for inbox/history view)
    """
    try:
        user = request.current_user
        user_id = user.get("user_id")

        if not user_id:
            return jsonify({"success": False, "error": "User ID not found"}), 400

        # Fetch tasks assigned to current user with all necessary fields
        response = (
            supabase.table("tasks")
            .select("*")
            .eq("assigned_by_user_id", user_id)
            .is_("deleted_at", None)
            .order("created_at", desc=True)  # Most recent first
            .execute()
        )

        if hasattr(response, 'error') and response.error:
            return jsonify({"success": False, "error": str(response.error)}), 400

        tasks = response.data or []
        # Add default status if not present
        for task in tasks:
            if not task.get('status'):
                task['status'] = 'Pending'

        return jsonify({
            "success": True, 
            "tasks": tasks,
            "count": len(tasks)
        }), 200

    except Exception as e:
        print(f"Error fetching inbox tasks: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/get-search", methods=["GET"])
@jwt_required
def get_search_data():
    """
    Get all tasks assigned to the current user (for inbox/history view)
    """
    try:
        user = request.current_user
        user_id = user.get("user_id")
        
        if not user_id:
            return jsonify({"success": False, "error": "User ID not found"}), 400

        # Fetch tasks assigned to current user with all necessary fields
        response = (
            supabase.table("search")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        if hasattr(response, 'error') and response.error:
            return jsonify({"success": False, "error": str(response.error)}), 400

        searchs = response.data or []

        return jsonify({
            "success": True, 
            "tasks": searchs,
            "count": len(searchs)
        }), 200

    except Exception as e:
        print(f"Error fetching inbox tasks: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/tasks/inbox", methods=["GET"])
@jwt_required
def get_inbox_tasks():
    """
    Get all tasks assigned to the current user (for inbox/history view)
    """
    try:
        user = request.current_user
        user_id = user.get("user_id")
        
        if not user_id:
            return jsonify({"success": False, "error": "User ID not found"}), 400

        # Fetch tasks assigned to current user with all necessary fields
        response = (
            supabase.table("tasks")
            .select("*")
            .eq("assigned_to_user_id", user_id)
            .is_("deleted_at", None)
            .order("created_at", desc=True)  # Most recent first
            .execute()
        )
        if hasattr(response, 'error') and response.error:
            return jsonify({"success": False, "error": str(response.error)}), 400

        tasks = response.data or []
        # Add default status if not present
        for task in tasks:
            if not task.get('status'):
                task['status'] = 'Pending'

        return jsonify({
            "success": True, 
            "tasks": tasks,
            "count": len(tasks)
        }), 200

    except Exception as e:
        print(f"Error fetching inbox tasks: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
   
@app.route("/api/get-users", methods=["GET"])
@jwt_required
def get_org_users():
    try:
        user = request.current_user
        org_id = user.get("org_id")
        
        if not org_id:
            return jsonify({"success": False, "error": "Organization not found"}), 400
        
        # Fetch all users from the same organization
        response = supabase.table("users").select("id, name, email, role").eq("org_id", org_id).eq("role","user").execute()
        
        # Check if there's an error in the response
        if hasattr(response, 'error') and response.error:
            return jsonify({"success": False, "error": str(response.error)}), 400
        
        # Alternative error check if the above doesn't work
        if not response.data:
            return jsonify({"success": False, "error": "No users found or database error"}), 400
        
        # Filter out the current user (optional - they might want to assign to themselves)
        # users = [user_data for user_data in response.data if user_data["user_id"] != user["user_id"]]
        users = response.data
        
        return jsonify({"success": True, "users": users}), 200
        
    except Exception as e:
        print(f"Error fetching users: {str(e)}")  # For debugging
        return jsonify({"success": False, "error": str(e)}), 500
    
def deduct_credits(user_id,org_id, action_type, reference_id=None):
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
    supabase.table("organisation").update({"creds": new_balance}).eq("id", org_id).execute()

    # Log it
    supabase.table("credit_logs").insert({
        "user_id": user_id,
        "action": action_type,
        "deductions": cost,
        # "reference_id": reference_id
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
