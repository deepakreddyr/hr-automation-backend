# import os
# import json
# from dotenv import load_dotenv
# from openai import OpenAI
# import numpy as np
# from supabase import create_client
# from openai import OpenAI

# # =======================
# # Setup
# # =======================
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# # Load environment variables
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# if not api_key:
#     raise RuntimeError("❌ OPENAI_API_KEY is not set. Please configure it in your environment.")

# client = OpenAI(api_key=api_key)

# # =======================
# # Helper: generate embedding
# # =======================
# def get_embedding(text: str, model="text-embedding-3-large"):
#     text = text.replace("\n", " ")
#     emb = client.embeddings.create(input=[text], model=model).data[0].embedding
#     return np.array(emb)

# # =======================
# # Match JD against resumes
# # =======================
# def match_candidates(job_description: str, threshold: float = 0.85, limit: int = 20):
#     # Step 1: embed the job description
#     jd_embedding = get_embedding(job_description)

#     # Step 2: query resumes with cosine similarity
#     # Supabase/Postgres syntax for pgvector
#     response = supabase.rpc(
#         "match_resumes",  # a Postgres function you’ll create
#         {"query_embedding": jd_embedding.tolist(), "similarity_threshold": threshold, "match_count": limit},
#     ).execute()

#     return response.data


































# import numpy as np
# import time
# import logging
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass, asdict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import re
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from supabase import create_client, Client
# import json

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class JobDescription:
#     """Job Description data structure for filtering"""
#     title: str
#     description: str
#     required_skills: List[str]
#     preferred_skills: List[str] = None
#     min_experience: int = 0
#     max_experience: int = None
#     location: str = None
#     employment_type: str = None

# @dataclass
# class CandidateMatch:
#     """Result structure for matched candidates"""
#     candidate_id: str
#     overall_score: float
#     semantic_similarity: float
#     skill_match_score: float
#     experience_score: float
#     matched_skills: List[str]
#     missing_critical_skills: List[str]
#     experience_years: int
#     confidence_rating: str  # EXCELLENT, GOOD, FAIR, POOR
#     match_explanation: List[str]
#     rank: int

# class FlaskJDBasedResumeFilter:
#     """
#     Flask service for filtering candidates based on Job Description
#     Optimized for production use with large candidate pools
#     """
    
#     def __init__(self, supabase_url: str, supabase_key: str, 
#                  model_name: str = "all-MiniLM-L6-v2"):
#         """
#         Initialize the JD-based filtering service
        
#         Args:
#             supabase_url: Supabase project URL
#             supabase_key: Supabase anon/service key
#             model_name: Sentence transformer model name
#         """
#         self.supabase: Client = create_client(supabase_url, supabase_key)
#         self.model = SentenceTransformer(model_name)
#         self.max_workers = 8
        
#         # Performance optimization caches
#         self._jd_embedding_cache = {}
#         self._skill_extraction_cache = {}
        
#         # Pre-built skill patterns for fast extraction
#         self._skill_patterns = self._build_comprehensive_skill_patterns()
        
#         logger.info(f"JD-based Resume Filter initialized with model: {model_name}")
    
#     def _build_comprehensive_skill_patterns(self) -> Dict[str, re.Pattern]:
#         """Build comprehensive skill patterns for extraction"""
#         skill_categories = {
#             # Programming Languages
#             'python': r'\bpython\b',
#             'java': r'\bjava\b(?!\s*script)',
#             'javascript': r'\b(?:javascript|js)\b',
#             'typescript': r'\btypescript\b',
#             'react': r'\breact(?:js)?\b',
#             'angular': r'\bangular\b',
#             'vue': r'\bvue(?:js)?\b',
#             'node': r'\bnode(?:\.js)?\b',
#             'django': r'\bdjango\b',
#             'flask': r'\bflask\b',
#             'spring': r'\bspring\b',
#             'express': r'\bexpress(?:js)?\b',
            
#             # Databases
#             'mysql': r'\bmysql\b',
#             'postgresql': r'\bpostgresql\b',
#             'mongodb': r'\bmongodb\b',
#             'redis': r'\bredis\b',
#             'oracle': r'\boracle\b',
#             'sqlite': r'\bsqlite\b',
#             'elasticsearch': r'\belasticsearch\b',
            
#             # Cloud Platforms
#             'aws': r'\b(?:aws|amazon web services)\b',
#             'azure': r'\b(?:azure|microsoft azure)\b',
#             'gcp': r'\b(?:gcp|google cloud)\b',
#             'docker': r'\bdocker\b',
#             'kubernetes': r'\bkubernetes\b',
            
#             # Data Science & ML
#             'machine learning': r'\b(?:machine learning|ml)\b',
#             'data science': r'\bdata science\b',
#             'tensorflow': r'\btensorflow\b',
#             'pytorch': r'\bpytorch\b',
#             'pandas': r'\bpandas\b',
#             'numpy': r'\bnumpy\b',
#             'scikit-learn': r'\bscikit-learn\b',
            
#             # DevOps & Tools
#             'git': r'\bgit\b',
#             'jenkins': r'\bjenkins\b',
#             'terraform': r'\bterraform\b',
#             'ansible': r'\bansible\b',
            
#             # Web Technologies
#             'html': r'\bhtml\b',
#             'css': r'\bcss\b',
#             'rest api': r'\b(?:rest|restful)\s*api\b',
#             'graphql': r'\bgraphql\b',
#             'microservices': r'\bmicroservices\b',
            
#             # Methodologies
#             'agile': r'\bagile\b',
#             'scrum': r'\bscrum\b',
#             'devops': r'\bdevops\b'
#         }
        
#         return {skill: re.compile(pattern, re.IGNORECASE) 
#                 for skill, pattern in skill_categories.items()}
    
#     def fetch_candidates_for_filtering(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
#         """
#         Fetch candidates with resume embeddings and text for JD-based filtering
        
#         Args:
#             limit: Number of candidates to fetch
#             offset: Offset for pagination
            
#         Returns:
#             List of candidate records
#         """
#         try:
#             start_time = time.time()
            
#             # Fetch only necessary fields for performance
#             response = (self.supabase
#                        .table('candidates')
#                        .select('id, resume_data->full_text, resume_embedding, resume_data->experience_years, resume_data->location')
#                        .is_('resume_embedding', 'not', 'null')
#                        .is_('resume_data', 'not', 'null')
#                        .range(offset, offset + limit - 1)
#                        .execute())
            
#             fetch_time = time.time() - start_time
#             logger.info(f"Fetched {len(response.data)} candidates in {fetch_time:.2f}s")
            
#             return response.data
            
#         except Exception as e:
#             logger.error(f"Error fetching candidates: {str(e)}")
#             return []
    
#     def generate_jd_embedding(self, job_desc: JobDescription) -> np.ndarray:
#         """
#         Generate embedding for job description with caching
        
#         Args:
#             job_desc: JobDescription object
            
#         Returns:
#             Numpy array representing JD embedding
#         """
#         # Create cache key based on JD content
#         cache_key = hash(f"{job_desc.title}_{job_desc.description}_{'_'.join(job_desc.required_skills or [])}")
        
#         if cache_key in self._jd_embedding_cache:
#             return self._jd_embedding_cache[cache_key]
        
#         # Construct comprehensive JD text for embedding
#         jd_text_parts = [job_desc.title, job_desc.description]
        
#         if job_desc.required_skills:
#             jd_text_parts.append(f"Required skills: {', '.join(job_desc.required_skills)}")
        
#         if job_desc.preferred_skills:
#             jd_text_parts.append(f"Preferred skills: {', '.join(job_desc.preferred_skills)}")
        
#         if job_desc.location:
#             jd_text_parts.append(f"Location: {job_desc.location}")
        
#         combined_jd_text = ". ".join(jd_text_parts)
        
#         # Generate embedding
#         embedding = self.model.encode(combined_jd_text, convert_to_tensor=False)
#         self._jd_embedding_cache[cache_key] = embedding
        
#         return embedding
    
#     def extract_skills_from_resume(self, resume_text: str) -> List[str]:
#         """
#         Fast skill extraction from resume text using cached patterns
        
#         Args:
#             resume_text: Resume text content
            
#         Returns:
#             List of extracted skills
#         """
#         if not resume_text:
#             return []
        
#         # Check cache first
#         text_hash = hash(resume_text)
#         if text_hash in self._skill_extraction_cache:
#             return self._skill_extraction_cache[text_hash]
        
#         extracted_skills = []
#         resume_lower = resume_text.lower()
        
#         # Use pre-compiled patterns for fast matching
#         for skill, pattern in self._skill_patterns.items():
#             if pattern.search(resume_lower):
#                 extracted_skills.append(skill)
        
#         # Cache the result
#         self._skill_extraction_cache[text_hash] = extracted_skills
        
#         return extracted_skills
    
#     def extract_experience_years(self, resume_text: str, fallback_years: int = None) -> int:
#         """
#         Extract years of experience from resume text
        
#         Args:
#             resume_text: Resume text content
#             fallback_years: Fallback value if extraction fails
            
#         Returns:
#             Years of experience
#         """
#         if fallback_years is not None:
#             return fallback_years
        
#         if not resume_text:
#             return 0
        
#         # Experience extraction patterns
#         patterns = [
#             r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
#             r'experience\s*:?\s*(\d+)\s*\+?\s*years?',
#             r'(\d+)\s*\+?\s*yrs\s+(?:of\s+)?experience',
#             r'(\d+)\s*years?\s+in\s+(?:software|development|programming)',
#         ]
        
#         max_experience = 0
#         text_lower = resume_text.lower()
        
#         for pattern in patterns:
#             matches = re.findall(pattern, text_lower)
#             for match in matches:
#                 try:
#                     years = int(match)
#                     max_experience = max(max_experience, years)
#                 except ValueError:
#                     continue
        
#         return max_experience
    
#     def calculate_semantic_similarity(self, jd_embedding: np.ndarray, 
#                                     resume_embeddings: List[np.ndarray]) -> List[float]:
#         """
#         Calculate semantic similarity between JD and resume embeddings
        
#         Args:
#             jd_embedding: Job description embedding
#             resume_embeddings: List of resume embeddings
            
#         Returns:
#             List of similarity scores
#         """
#         if not resume_embeddings:
#             return []
        
#         # Stack embeddings for batch computation
#         resume_matrix = np.vstack(resume_embeddings)
#         jd_embedding_2d = jd_embedding.reshape(1, -1)
        
#         # Compute cosine similarities
#         similarities = cosine_similarity(jd_embedding_2d, resume_matrix)[0]
        
#         return similarities.tolist()
    
#     def calculate_skill_match_score(self, resume_skills: List[str], 
#                                   required_skills: List[str], 
#                                   preferred_skills: List[str] = None) -> Tuple[float, List[str], List[str]]:
#         """
#         Calculate skill matching score against JD requirements
        
#         Args:
#             resume_skills: Skills extracted from resume
#             required_skills: Required skills from JD
#             preferred_skills: Preferred skills from JD
            
#         Returns:
#             Tuple of (skill_score, matched_skills, missing_skills)
#         """
#         if not required_skills:
#             return 1.0, [], []
        
#         resume_skills_set = set(skill.lower() for skill in resume_skills)
#         required_set = set(skill.lower() for skill in required_skills)
#         preferred_set = set(skill.lower() for skill in (preferred_skills or []))
        
#         # Calculate matches
#         matched_required = list(required_set & resume_skills_set)
#         matched_preferred = list(preferred_set & resume_skills_set)
#         missing_required = list(required_set - resume_skills_set)
        
#         # Calculate scores
#         required_match_ratio = len(matched_required) / len(required_set)
#         preferred_match_ratio = len(matched_preferred) / len(preferred_set) if preferred_set else 0
        
#         # Weighted skill score (required skills are more important)
#         skill_score = (required_match_ratio * 0.8) + (preferred_match_ratio * 0.2)
        
#         all_matched_skills = matched_required + matched_preferred
        
#         return skill_score, all_matched_skills, missing_required
    
#     def calculate_experience_match_score(self, candidate_experience: int, 
#                                        required_experience: int, 
#                                        max_experience: int = None) -> float:
#         """
#         Calculate experience matching score against JD requirements
        
#         Args:
#             candidate_experience: Candidate's years of experience
#             required_experience: Required years from JD
#             max_experience: Maximum preferred years from JD
            
#         Returns:
#             Experience match score
#         """
#         if required_experience == 0:
#             return 1.0
        
#         if candidate_experience >= required_experience:
#             # Full score for meeting requirements
#             base_score = 1.0
            
#             # Check if within preferred range
#             if max_experience and candidate_experience <= max_experience:
#                 return 1.0
#             elif max_experience and candidate_experience > max_experience:
#                 # Small penalty for being overqualified
#                 return max(0.8, 1.0 - (candidate_experience - max_experience) * 0.05)
#             else:
#                 # Bonus for exceeding minimum (capped)
#                 bonus = min((candidate_experience - required_experience) * 0.1, 0.2)
#                 return min(base_score + bonus, 1.2)
#         else:
#             # Partial score for less experience
#             return max(candidate_experience / required_experience, 0.1)
    
#     def determine_confidence_rating(self, overall_score: float, 
#                                   skill_score: float, 
#                                   semantic_score: float) -> str:
#         """
#         Determine confidence rating based on scores
        
#         Args:
#             overall_score: Overall matching score
#             skill_score: Skill matching score
#             semantic_score: Semantic similarity score
            
#         Returns:
#             Confidence rating string
#         """
#         if overall_score >= 0.8 and skill_score >= 0.7 and semantic_score >= 0.6:
#             return "EXCELLENT"
#         elif overall_score >= 0.65 and skill_score >= 0.5:
#             return "GOOD"
#         elif overall_score >= 0.4:
#             return "FAIR"
#         else:
#             return "POOR"
    
#     def generate_match_explanation(self, semantic_score: float, 
#                                  skill_score: float, 
#                                  experience_score: float,
#                                  matched_skills: List[str], 
#                                  missing_skills: List[str]) -> List[str]:
#         """
#         Generate human-readable match explanation
        
#         Args:
#             semantic_score: Semantic similarity score
#             skill_score: Skill match score
#             experience_score: Experience match score
#             matched_skills: List of matched skills
#             missing_skills: List of missing skills
            
#         Returns:
#             List of explanation strings
#         """
#         explanations = []
        
#         # Semantic similarity explanation
#         if semantic_score > 0.7:
#             explanations.append("Strong semantic match with job description")
#         elif semantic_score > 0.5:
#             explanations.append("Good semantic alignment with job requirements")
#         elif semantic_score > 0.3:
#             explanations.append("Moderate semantic similarity")
#         else:
#             explanations.append("Limited semantic alignment")
        
#         # Skill match explanation
#         if skill_score > 0.8:
#             explanations.append(f"Excellent skill match: {', '.join(matched_skills[:5])}")
#         elif skill_score > 0.6:
#             explanations.append(f"Good skill alignment: {', '.join(matched_skills[:3])}")
#         elif skill_score > 0.3:
#             explanations.append(f"Some relevant skills: {', '.join(matched_skills[:3])}")
        
#         # Missing skills
#         if missing_skills:
#             if len(missing_skills) <= 2:
#                 explanations.append(f"Missing: {', '.join(missing_skills)}")
#             else:
#                 explanations.append(f"Missing key skills: {', '.join(missing_skills[:2])} and {len(missing_skills)-2} more")
        
#         # Experience explanation
#         if experience_score > 1.0:
#             explanations.append("Exceeds experience requirements")
#         elif experience_score >= 0.8:
#             explanations.append("Meets experience requirements")
#         elif experience_score >= 0.5:
#             explanations.append("Partially meets experience requirements")
#         else:
#             explanations.append("Below experience requirements")
        
#         return explanations
    
#     def filter_candidates_by_jd(self, job_desc: JobDescription, 
#                               min_score: float = 0.3, 
#                               max_results: int = 100,
#                               candidate_limit: int = 1000) -> List[CandidateMatch]:
#         """
#         Main method to filter candidates based on Job Description
        
#         Args:
#             job_desc: JobDescription object with requirements
#             min_score: Minimum overall score threshold
#             max_results: Maximum number of results to return
#             candidate_limit: Maximum candidates to evaluate
            
#         Returns:
#             List of CandidateMatch objects sorted by score
#         """
#         start_time = time.time()
#         logger.info(f"Starting JD-based filtering for: {job_desc.title}")
        
#         # Step 1: Generate JD embedding
#         jd_embedding = self.generate_jd_embedding(job_desc)
        
#         # Step 2: Fetch candidates
#         candidates = self.fetch_candidates_for_filtering(limit=candidate_limit)
#         if not candidates:
#             logger.warning("No candidates found for filtering")
#             return []
        
#         # Step 3: Process candidates in parallel
#         valid_candidates = []
#         resume_embeddings = []
        
#         for candidate in candidates:
#             try:
#                 if candidate.get('resume_embedding'):
#                     resume_embedding = np.array(candidate['resume_embedding'])
#                     valid_candidates.append(candidate)
#                     resume_embeddings.append(resume_embedding)
#             except Exception as e:
#                 logger.warning(f"Error processing candidate {candidate.get('id')}: {e}")
#                 continue
        
#         if not valid_candidates:
#             logger.warning("No valid candidates with embeddings found")
#             return []
        
#         # Step 4: Calculate semantic similarities in batch
#         semantic_scores = self.calculate_semantic_similarity(jd_embedding, resume_embeddings)
        
#         # Step 5: Process each candidate for detailed scoring
#         results = []
        
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             future_to_candidate = {}
            
#             for i, candidate in enumerate(valid_candidates):
#                 future = executor.submit(
#                     self._process_single_candidate,
#                     candidate,
#                     job_desc,
#                     semantic_scores[i]
#                 )
#                 future_to_candidate[future] = (candidate, i)
            
#             # Collect results
#             for future in as_completed(future_to_candidate):
#                 try:
#                     match_result = future.result()
#                     if match_result and match_result.overall_score >= min_score:
#                         results.append(match_result)
#                 except Exception as e:
#                     candidate, idx = future_to_candidate[future]
#                     logger.error(f"Error processing candidate {candidate.get('id')}: {e}")
        
#         # Step 6: Sort and rank results
#         results.sort(key=lambda x: x.overall_score, reverse=True)
        
#         # Add rankings
#         for i, result in enumerate(results[:max_results], 1):
#             result.rank = i
        
#         processing_time = time.time() - start_time
#         logger.info(f"Filtered {len(results)} candidates from {len(candidates)} in {processing_time:.2f}s")
        
#         return results[:max_results]
    
#     def _process_single_candidate(self, candidate: Dict, 
#                                 job_desc: JobDescription, 
#                                 semantic_score: float) -> Optional[CandidateMatch]:
#         """
#         Process a single candidate for detailed matching
        
#         Args:
#             candidate: Candidate data from database
#             job_desc: Job description requirements
#             semantic_score: Pre-calculated semantic similarity score
            
#         Returns:
#             CandidateMatch object or None
#         """
#         try:
#             candidate_id = candidate['id']
#             resume_text = candidate.get('full_text', '')
#             experience_years = candidate.get('experience_years', 0)
            
#             # Extract skills from resume
#             resume_skills = self.extract_skills_from_resume(resume_text)
            
#             # Extract experience if not provided
#             if not experience_years:
#                 experience_years = self.extract_experience_years(resume_text)
            
#             # Calculate skill matching
#             skill_score, matched_skills, missing_skills = self.calculate_skill_match_score(
#                 resume_skills, 
#                 job_desc.required_skills or [], 
#                 job_desc.preferred_skills or []
#             )
            
#             # Calculate experience matching
#             experience_score = self.calculate_experience_match_score(
#                 experience_years, 
#                 job_desc.min_experience,
#                 job_desc.max_experience
#             )
            
#             # Calculate overall weighted score
#             overall_score = (
#                 semantic_score * 0.4 +      # Semantic similarity with JD
#                 skill_score * 0.4 +         # Skill matching
#                 experience_score * 0.2      # Experience matching
#             )
            
#             # Determine confidence rating
#             confidence_rating = self.determine_confidence_rating(
#                 overall_score, skill_score, semantic_score
#             )
            
#             # Generate match explanation
#             match_explanation = self.generate_match_explanation(
#                 semantic_score, skill_score, experience_score,
#                 matched_skills, missing_skills
#             )
            
#             return CandidateMatch(
#                 candidate_id=candidate_id,
#                 overall_score=round(overall_score, 3),
#                 semantic_similarity=round(semantic_score, 3),
#                 skill_match_score=round(skill_score, 3),
#                 experience_score=round(experience_score, 3),
#                 matched_skills=matched_skills,
#                 missing_critical_skills=missing_skills,
#                 experience_years=experience_years,
#                 confidence_rating=confidence_rating,
#                 match_explanation=match_explanation,
#                 rank=0  # Will be set after sorting
#             )
            
#         except Exception as e:
#             logger.error(f"Error processing candidate {candidate.get('id')}: {e}")
#             return None
    
#     def get_filtering_summary(self, results: List[CandidateMatch]) -> Dict[str, Any]:
#         """
#         Generate summary statistics for filtering results
        
#         Args:
#             results: List of CandidateMatch results
            
#         Returns:
#             Dictionary with summary statistics
#         """
#         if not results:
#             return {"total_matches": 0}
        
#         confidence_counts = {
#             "EXCELLENT": len([r for r in results if r.confidence_rating == "EXCELLENT"]),
#             "GOOD": len([r for r in results if r.confidence_rating == "GOOD"]),
#             "FAIR": len([r for r in results if r.confidence_rating == "FAIR"]),
#             "POOR": len([r for r in results if r.confidence_rating == "POOR"])
#         }
        
#         scores = [r.overall_score for r in results]
        
#         return {
#             "total_matches": len(results),
#             "confidence_distribution": confidence_counts,
#             "score_statistics": {
#                 "average_score": round(sum(scores) / len(scores), 3),
#                 "highest_score": max(scores),
#                 "lowest_score": min(scores)
#             },
#             "top_candidate_id": results[0].candidate_id if results else None
#         }

# # Flask usage example
# """
# # Initialize the service
# resume_filter = FlaskJDBasedResumeFilter(
#     supabase_url="your-supabase-url",
#     supabase_key="your-supabase-key"
# )

# # Define job description
# job_desc = JobDescription(
#     title="Senior Python Developer",
#     description="Looking for an experienced Python developer to build scalable web applications",
#     required_skills=["python", "django", "postgresql", "rest api"],
#     preferred_skills=["react", "aws", "docker"],
#     min_experience=3,
#     max_experience=8,
#     location="Remote"
# )

# # Filter candidates
# matches = resume_filter.filter_candidates_by_jd(
#     job_desc=job_desc,
#     min_score=0.4,
#     max_results=50,
#     candidate_limit=1000
# )

# # Get summary
# summary = resume_filter.get_filtering_summary(matches)
# """