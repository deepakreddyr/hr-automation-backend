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
from dashboard_data import (
    get_dashboard_data,
    get_creds_used,
    get_user_name
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

# ─── JWT Configuration ─────────────────────────────────────────────────────

# JWT Secret Key - Use a strong secret key
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)  # Access token expires in 24 hours
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)  # Refresh token expires in 30 days

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

# Secure secret key (still needed for Flask operations)
app.secret_key = os.getenv("SECRET_KEY", "xJ7vK9mQ2nR8pL6wE4tY1uI0oP3aS5dF7gH9jK2lM6nB8vC1xZ4qW7eR3tY6uI9o")

print(f"JWT Authentication configured with {JWT_ACCESS_TOKEN_EXPIRES} access token expiry")

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

# ─── JWT Helper Functions ─────────────────────────────────────────────────────

def generate_access_token(user_id, email):
    """Generate JWT access token"""
    payload = {
        'user_id': str(user_id),  # Ensure user_id is a string
        'email': str(email),      # Ensure email is a string
        'exp': datetime.now(timezone.utc) + JWT_ACCESS_TOKEN_EXPIRES,
        'iat': datetime.now(timezone.utc),
        'type': 'access'
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def generate_refresh_token(user_id, email):
    """Generate JWT refresh token"""
    payload = {
        'user_id': str(user_id),  # Ensure user_id is a string
        'email': str(email),      # Ensure email is a string
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
        'email': payload.get('email')
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
        user_data = supabase.table("users").select("id, email").eq("email", email).execute()
        if not user_data.data:
            return jsonify({"success": False, "message": "User not found in database"}), 404

        user_id = user_data.data[0]["id"]
        user_email = user_data.data[0]["email"]

        # Ensure user_id and email are strings and add debug logging
        user_id_str = str(user_id)
        user_email_str = str(user_email)
        
        print(f"Creating JWT tokens for user_id: {user_id_str} ({type(user_id)}), email: {user_email_str}")

        # Generate JWT tokens
        access_token = generate_access_token(user_id_str, user_email_str)
        refresh_token = generate_refresh_token(user_id_str, user_email_str)
        
        print(f"JWT tokens generated for user: {user_email}")
        
        return jsonify({
            "success": True, 
            "message": "Login successful",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": user_id_str,
                "email": user_email_str
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
        new_access_token = generate_access_token(payload['user_id'], payload['email'])
        
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

@app.route("/api/create-search", methods=["POST"])
@jwt_required
def create_search():
    """Create a new search with JWT authentication"""
    try:
        # Get user info from JWT token
        user = request.current_user
        user_id = user['user_id']
        
        print(f"Creating search for user_id: {user_id}")
        
        # Create search record
        

        # Create history record
        history_resp = supabase.table("history").insert({
            "user_id": user_id,
            "creds": 0
        }).execute()
        
        if not history_resp.data:
            print("❌ Failed to insert history")
            return jsonify({"error": "Error inserting history"}), 500

        history_id = history_resp.data[0]["id"]
        print(f"Created history record with ID: {history_id}")
        response = supabase.table("search").insert({
            "user_id": user_id,
            "history_id": history_id,
            "processed": False,
            "remote_work": False,
            "contract_hiring": False,
            "key_skills": "",
            "job_role": "",
            "raw_data": "",
            "shortlisted_index": "[]",
            "noc": 0,
            "job_description": "",
            "status": "shortlist"
        }).execute()
        if response.data:
            search_id = response.data[0]["id"]
            print(f"Created search record with ID: {search_id}")
            
            return jsonify({
                "success": True,
                "search_id": search_id,
                "history_id": history_id
            })
        else:
            return jsonify({"error": "Could not create search"}), 500
            
    except Exception as e:
        print(f"Create search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/shortlist/<int:search_id>", methods=["GET", "POST"])
@jwt_required
def shortlist(search_id):
    user = request.current_user
    user_id = user['user_id']
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
        print(f"SCRAPED DATA {final_candidates}")
        shortlisted_indices = shortlist_candidates(final_candidates, skills)
        print(f"SHORTLISTED DATA {shortlisted_indices}")
        
        if not shortlisted_indices:  # nothing matched
            return jsonify({
                "success": False,
                "message": "No candidates matched the required skills."
            }), 200
        
        # ✅ Update the search entry
        supabase.table("search").update({
            "raw_data": candidate_data,
            "key_skills": skills,
            "job_role": job_role,
            "user_id": user_id,
            "shortlisted_index": json.dumps(shortlisted_indices),
            "processed": True,
            # "remote_work": False,
            "contract_hiring": False,
            "noc": noc,
            "job_description": jd_text,
            "search_name": search_name,
            "status":"process"
        }).eq("id", search_id).execute()
        deduct_credits(user_id, "search", reference_id=None)
        return jsonify({"success":True})

@app.route("/api/process/<int:search_id>", methods=["GET", "POST"]) 
@jwt_required 
def process(search_id):     
    user = request.current_user     
    user_id = user["user_id"]      

    # === Load search ===     
    result = supabase.table("search").select("shortlisted_index, process_state, job_description, key_skills").eq("id", search_id).single().execute()     
    if not result.data:         
        return jsonify(success=False, error="Search not found"), 404      

    shortlisted = result.data.get("shortlisted_index") or []     
    if isinstance(shortlisted, str):         
        shortlisted = json.loads(shortlisted or "[]")      

    process_state = result.data.get("process_state") or {}     
    jd = result.data.get("job_description")     
    skills = result.data.get("key_skills")      

    # Ensure process_state is always a dict     
    if isinstance(process_state, str):         
        try:             
            process_state = json.loads(process_state or "{}")         
        except json.JSONDecodeError:             
            process_state = {}      

    submitted = len(process_state.get("resume_dict", {}))     
    target = len(shortlisted)     
    candidate_index = shortlisted[submitted] if submitted < target else None     
    is_last = (submitted == target - 1)      

    # === POST (submit resume) ===     
    if request.method == "POST":         
        resume_text = request.form.get("resumeText", "").strip()         
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
        # Only validate these once (first candidate)         
        if "right_fields" not in process_state:             
            if not hiring_company: errors.append("Hiring Company is required.")             
            if not company_location: errors.append("Company Location is required.")             
            if not hr_company: errors.append("HR Company is required.")             
            if not notice_period: errors.append("Notice Period is required.")          

        if errors:             
            return jsonify(success=False, errors=errors), 400          

        # === Update resume_dict ===         
        resume_dict = process_state.get("resume_dict", {})         
        resume_dict[f"candidate_{candidate_index}"] = resume_text         
        process_state["resume_dict"] = resume_dict          

        # === Set right_fields once ===         
        if "right_fields" not in process_state:             
            process_state["right_fields"] = {                 
                "hiringCompany": hiring_company,                 
                "companyLocation": company_location,                 
                "hrCompany": hr_company,                 
                "noticePeriod": notice_period,                 
                "remoteWork": remote_work,                 
                "contractH": contract_h,             
            }             
            # also update search table             
            supabase.table("search").update({                 
                "rc_name": hiring_company,                 
                "company_location": company_location,                 
                "hc_name": hr_company,                 
                "notice_period": notice_period,                 
                "remote_work": remote_work,                 
                "contract_hiring": contract_h,                 
                "user_id": user_id             
            }).eq("id", search_id).execute()          

        # === Save custom question if final ===         
        if submitted + 1 == target and custom_question:             
            process_state["custom_question"] = custom_question             
            supabase.table("search").update({                 
                "custom_question": custom_question             
            }).eq("id", search_id).execute()          

        # === Persist process_state ===         
        supabase.table("search").update({             
            "process_state": json.dumps(process_state)         
        }).eq("id", search_id).execute()          

        submitted = len(resume_dict)                  
        
        # === If this is the final submission, start processing ===         
        if submitted == target:
            # Set processing status immediately
            supabase.table("search").update({
                "status": "processing",
                "processed": False
            }).eq("id", search_id).execute()
            
            # Return loading state to frontend
            return jsonify({
                "success": True,
                "submitted": submitted,
                "target": target,
                "candidateIndex": candidate_index,
                "isLast": True,
                "right_fields": process_state["right_fields"],
                "next": False,
                "processing": True,  # Signal to show loading
                "search_id": search_id
            })

        # Next candidate (not final)         
        next_index = shortlisted[submitted]         
        return jsonify({             
            "success": True,             
            "submitted": submitted,             
            "target": target,             
            "candidateIndex": next_index,             
            "isLast": (submitted == target - 1),             
            "right_fields": process_state["right_fields"],             
            "next": True         
        })      

    # === GET (initial load) ===     
    return jsonify({         
        "success": True,         
        "submitted": submitted,         
        "target": target,         
        "candidateIndex": candidate_index,         
        "isLast": is_last,         
        "shortlisted_indices": shortlisted,         
        "right_fields": process_state.get("right_fields", {}),         
        "next": True     
    })


# New route to handle the actual processing (runs in background)
@app.route("/api/process-candidates/<int:search_id>", methods=["POST"])
@jwt_required
def process_candidates(search_id):
    user = request.current_user
    user_id = user["user_id"]
    
    try:
        # Get search data
        result = supabase.table("search").select("process_state, job_description, key_skills").eq("id", search_id).single().execute()
        if not result.data:
            return jsonify(success=False, error="Search not found"), 404
        
        process_state = result.data.get("process_state") or {}
        if isinstance(process_state, str):
            process_state = json.loads(process_state or "{}")
        
        jd = result.data.get("job_description")
        skills = result.data.get("key_skills")
        resume_dict = process_state.get("resume_dict", {})
        
        print(f"Processing {len(resume_dict)} resumes for search {search_id}")
        
        # Get combined resumes
        combined_resumes = "\n".join(resume_dict.values())
        
        # Validate required data
        if not combined_resumes.strip():
            supabase.table("search").update({"status": "error", "processed": True}).eq("id", search_id).execute()
            return jsonify(success=False, error="No resume data found"), 400
        if not jd or not jd.strip():
            supabase.table("search").update({"status": "error", "processed": True}).eq("id", search_id).execute()
            return jsonify(success=False, error="No job description found"), 400
        
        print(f"Processing {len(resume_dict)} resumes, combined length: {len(combined_resumes)}")
        
        # Get candidate details from AI
        candidate_data = get_candidate_details(combined_resumes, jd, skills)
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
        
        # Insert candidates into database
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
                
                # Check if candidate already exists
                existing = supabase.table("candidates").select("id").eq("email", email).eq("search_id", search_id).execute()
                if existing.data:
                    continue
                
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
                    "match_score": match_score
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
        
        # Deduct credits
        deduct_credits(user_id, "process_candidate", reference_id=None)
        
        return jsonify({
            "success": True,
            "candidates_processed": inserted_count,
            "search_id": search_id
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
    candidate_id= data.get('candidate_id')

    da=supabase.from_('search')\
    .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id')\
    .eq('id',search_id)\
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

@app.route("/api/call", methods=["POST"])
@jwt_required
def initiate_call():
    data = request.get_json()
    candidates = data.get("candidates", [])
    search_id=data.get("search_id",[])
    results = []
    da=supabase.from_('search')\
    .select('rc_name,hc_name,remote_work,contract_hiring,company_location,notice_period,key_skills,custom_question,id')\
    .eq('id',search_id)\
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


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    print(data)

    message = data.get("message", data)
    end_call_status = message.get("status", "")
    call = message.get("call", {})
    assistantOverrides = call.get("assistantOverrides", {})
    variableValues = assistantOverrides.get("variableValues", {})
    candidate_id = variableValues.get("candidate_id")

    # Lookup candidate to get user_id
    candidate_res = supabase.table("candidates").select("id, search_id").eq("id", candidate_id).execute()
    if not candidate_res.data:
        return jsonify({"error": "Candidate not found"}), 404

    search_id = candidate_res.data[0]["search_id"]

    search_res = supabase.table("search").select("user_id").eq("id", search_id).execute()
    if not search_res.data:
        return jsonify({"error": "Search not found"}), 404

    user_id = search_res.data[0]["user_id"]   # <-- instead of session["user_id"]

    customer = call.get("customer", {})
    name = customer.get("name", "")
    phone = int(customer.get("number", "")[-10:]) if customer.get("number") else None
    transcript = message.get("transcript", "")
    analysis = message.get("analysis", {})
    summary = analysis.get("summary", "")
    structuredData = analysis.get("structuredData", {})
    status = structuredData.get("re-schedule", "")
    success_eval = analysis.get("successEvaluation", "")
    durationMinutes = message.get("durationMinutes", "")

    call_status = None

    if end_call_status == "ended":
        call_status = "Not Answered"
        supabase.table("candidates").update({
            "call_status": call_status
        }).eq('id', candidate_id).execute()

    if status == "yes":
        call_status = "Re-schedule"
        add_call_data(transcript, summary, structuredData, call_status, success_eval, phone, durationMinutes, name, candidate_id=int(candidate_id))
        deduct_credits(user_id, "rescheduled_call", reference_id=None)

    elif status == "no":
        call_status = "Called & Answered"
        add_call_data(transcript, summary, structuredData, call_status, success_eval, phone, durationMinutes, name, candidate_id=int(candidate_id))
        deduct_credits(user_id, "ai_call", reference_id=None)

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
@jwt_required
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
@jwt_required
def dashboard():
    try:
        user = request.current_user
        user_id = user['user_id']

        print(f"Dashboard access granted for user_id: {user_id}")

        dashboard_data = get_dashboard_data(user_id)

        # Add extra simple lookups
        dashboard_data["creds_used"] = get_creds_used(user_id)
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
    
    response = supabase.table("candidates") \
        .select("*") \
        .eq("liked", True) \
        .execute()

    candidates = response.data or []
    return jsonify(candidates)

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
        "organization": user.get("organization"),
        "creds": user.get("creds"),
        "id": user.get("id")
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
