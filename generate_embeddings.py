# <--- full corrected MemoryOptimizedResumeExtractor class --->
import pandas as pd
import requests
import io
from urllib.parse import urlparse, parse_qs
import PyPDF2
from docx import Document
import tempfile
import os
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
import json
from typing import List, Dict, Optional, Generator
import logging
from openai import OpenAI
from supabase import create_client
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizedResumeExtractor:
    def __init__(self, max_workers: int = 3, batch_size: int = 50, max_memory_mb: int = 512,
                 openai_api_key: str = None, supabase_url: str = None, supabase_key: str = None):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.openai_client = None
        self.supabase_client = None

        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

        if (supabase_url or os.getenv("SUPABASE_URL")) and (supabase_key or os.getenv("SUPABASE_KEY")):
            try:
                self.supabase_client = create_client(
                    supabase_url or os.getenv("SUPABASE_URL"),
                    supabase_key or os.getenv("SUPABASE_KEY")
                )
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")

        self.session_pool = Queue(maxsize=max_workers * 2)
        for _ in range(max_workers * 2):
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            session.mount('http://', requests.adapters.HTTPAdapter(
                pool_connections=1, pool_maxsize=1, max_retries=2
            ))
            session.mount('https://', requests.adapters.HTTPAdapter(
                pool_connections=1, pool_maxsize=1, max_retries=2
            ))
            self.session_pool.put(session)

        self.stats = {
            'processed': 0,
            'errors': 0,
            'db_inserted': 0,
            'db_errors': 0,
            'embedding_errors': 0,
            'total_size_mb': 0,
            'start_time': None
        }
        self.stats_lock = threading.Lock()

    def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return None

        max_tokens = 8000
        if len(text) > max_tokens * 4:
            text = text[:max_tokens * 4] + "...(truncated for embedding)"
            logger.warning(f"Text truncated to {len(text)} characters for embedding")

        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                emb = response.data[0].embedding
                # Convert numpy-like to list if needed
                if hasattr(emb, 'tolist'):
                    emb = emb.tolist()
                return emb
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts")
                    with self.stats_lock:
                        self.stats['embedding_errors'] += 1
                    return None

    def insert_resume_to_db(self, user_id: str, name: str, email: str, phone: str,
                            resume_text: str, embedding: List[float],
                            additional_data: Dict = None, search_id: Optional[int] = None) -> bool:
        """
        Changed signature: user_id first, search_id is optional keyword.
        """
        if not self.supabase_client:
            logger.error("Supabase client not initialized")
            return False

        try:
            insert_data = {
                "user_id": user_id,
                "name": name,
                "email": email,
                "phone": phone,
                "resume_text": resume_text,
                "embedding": embedding,
                "search_id": search_id
            }
            if additional_data:
                insert_data.update(additional_data)

            result = self.supabase_client.table("resumes").insert(insert_data).execute()
            # Check status_code or data presence
            if getattr(result, 'status_code', None) and result.status_code >= 400:
                logger.error(f"Supabase insertion responded with status {result.status_code}: {getattr(result, 'data', None)}")
                with self.stats_lock:
                    self.stats['db_errors'] += 1
                return False

            if result.data:
                with self.stats_lock:
                    self.stats['db_inserted'] += 1
                logger.info(f"Successfully inserted resume for {name} (user_id: {user_id})")
                return True
            else:
                logger.error(f"Failed to insert resume for {name} (user_id: {user_id}): No data returned")
                with self.stats_lock:
                    self.stats['db_errors'] += 1
                return False
        except Exception as e:
            logger.error(f"Database insertion error for {name} (user_id {user_id}): {e}")
            with self.stats_lock:
                self.stats['db_errors'] += 1
            return False

    def get_session(self) -> requests.Session:
        return self.session_pool.get()

    def return_session(self, session: requests.Session):
        self.session_pool.put(session)

    def convert_drive_link_to_direct(self, drive_url: str) -> str:
        if not isinstance(drive_url, str):
            return drive_url
        if 'drive.google.com' not in drive_url:
            return drive_url
        try:
            if '/file/d/' in drive_url:
                file_id = drive_url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in drive_url:
                parsed_url = urlparse(drive_url)
                file_id = parse_qs(parsed_url.query)['id'][0]
            else:
                return drive_url
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except Exception as e:
            logger.error(f"Error converting drive link: {e}")
            return drive_url

    def clean_personal_data(self, value) -> str:
        if pd.isna(value) or value is None:
            return ""
        cleaned = str(value).strip()
        if cleaned.lower() in ['nan', 'none', 'null', '']:
            return ""
        return cleaned

    def read_spreadsheet_in_chunks(self, drive_url: str, chunk_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
        session = self.get_session()
        try:
            direct_url = self.convert_drive_link_to_direct(drive_url)
            response = session.get(direct_url, stream=True, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'csv' in content_type or drive_url.lower().endswith('.csv') or direct_url.lower().endswith('.csv'):
                # Use BytesIO -> pandas can read it
                csv_bytes = io.BytesIO(response.content)
                for chunk in pd.read_csv(csv_bytes, chunksize=chunk_size):
                    yield chunk
            else:
                df = pd.read_excel(io.BytesIO(response.content))
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size].copy()
                    yield chunk
                del df
                gc.collect()
        except Exception as e:
            logger.error(f"Error reading spreadsheet: {e}")
            raise
        finally:
            self.return_session(session)

    def extract_text_with_size_limit(self, content: bytes, file_type: str) -> str:
        if len(content) > self.max_memory_bytes:
            logger.warning(f"File size {len(content)/1024/1024:.2f}MB exceeds limit, skipping")
            return f"File too large ({len(content)/1024/1024:.2f}MB) - skipped for memory safety"
        try:
            if file_type == 'pdf':
                return self._extract_pdf_optimized(content)
            elif file_type == 'docx':
                return self._extract_docx_optimized(content)
            elif file_type == 'txt':
                return content.decode('utf-8', errors='ignore').strip()
            else:
                return "Unsupported file format"
        except Exception as e:
            logger.error(f"Error extracting {file_type}: {e}")
            return f"Error extracting text: {e}"
        finally:
            gc.collect()

    def _extract_pdf_optimized(self, pdf_content: bytes) -> str:
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            max_pages = min(len(pdf_reader.pages), 10)
            text_parts = []
            for i in range(max_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    if sum(len(part) for part in text_parts) > 50000:
                        text_parts.append("\n...(truncated for memory efficiency)")
                        break
                except Exception as e:
                    logger.warning(f"Error extracting page {i}: {e}")
                    continue
            return "\n".join(text_parts)
        except Exception as e:
            return f"PDF extraction error: {e}"
        finally:
            del pdf_content
            gc.collect()

    def _extract_docx_optimized(self, docx_content: bytes) -> str:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                temp_file.write(docx_content)
                temp_file_path = temp_file.name
            doc = Document(temp_file_path)
            text_parts = []
            total_length = 0
            max_length = 50000
            for paragraph in doc.paragraphs:
                if total_length >= max_length:
                    text_parts.append("\n...(truncated for memory efficiency)")
                    break
                text = paragraph.text
                text_parts.append(text)
                total_length += len(text)
            return "\n".join(text_parts)
        except Exception as e:
            return f"DOCX extraction error: {e}"
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            del docx_content
            gc.collect()

    # ---- FIXED signature: row_data first, then user_id, then search_id
    def process_single_resume(self, row_data: Dict, user_id: str, search_id: Optional[int]) -> Dict:
        """Process a single resume with embedding generation and database insertion"""
        session = self.get_session()
        # ensure row_data is a dict
        if not isinstance(row_data, dict):
            row_data = {'row_number': 0, 'name': '', 'email': '', 'phone': '', 'resume_url': row_data}
        try:
            row_number = row_data.get('row_number', 0)
            name = row_data.get('name', '')
            email = row_data.get('email', '')
            phone = row_data.get('phone', '')
            resume_url = row_data.get('resume_url', '')

            if not resume_url or pd.isna(resume_url):
                return {
                    'row_number': row_number,
                    'user_id': user_id,
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'resume_url': resume_url,
                    'resume_text': 'No resume URL provided',
                    'text_length': 0,
                    'status': 'skipped',
                    'processing_time': 0,
                    'embedding_generated': False,
                    'db_inserted': False
                }

            start_time = time.time()
            direct_url = self.convert_drive_link_to_direct(str(resume_url))
            response = session.get(direct_url, stream=True, timeout=30)
            response.raise_for_status()

            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_memory_bytes:
                return {
                    'row_number': row_number,
                    'user_id': user_id,
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'resume_url': resume_url,
                    'resume_text': f'File too large ({int(content_length)/1024/1024:.2f}MB)',
                    'text_length': 0,
                    'status': 'too_large',
                    'processing_time': time.time() - start_time,
                    'embedding_generated': False,
                    'db_inserted': False
                }

            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                content += chunk
                if len(content) > self.max_memory_bytes:
                    return {
                        'row_number': row_number,
                        'user_id': user_id,
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'resume_url': resume_url,
                        'resume_text': 'File too large - download stopped',
                        'text_length': 0,
                        'status': 'too_large',
                        'processing_time': time.time() - start_time,
                        'embedding_generated': False,
                        'db_inserted': False
                    }

            content_type = response.headers.get('content-type', '').lower()
            url_lower = str(resume_url).lower()
            if 'pdf' in content_type or url_lower.endswith('.pdf'):
                file_type = 'pdf'
            elif 'word' in content_type or url_lower.endswith(('.docx', '.doc')):
                file_type = 'docx'
            elif 'text' in content_type or url_lower.endswith('.txt'):
                file_type = 'txt'
            else:
                file_type = 'pdf'

            resume_text = self.extract_text_with_size_limit(content, file_type)

            embedding = None
            embedding_generated = False
            if resume_text and len(resume_text.strip()) > 50 and not resume_text.startswith('Error'):
                embedding = self.generate_embedding(resume_text)
                embedding_generated = embedding is not None

            db_inserted = False
            if embedding and user_id:
                additional_data = {
                    'source_url': resume_url,
                    'file_type': file_type,
                    'file_size_mb': len(content) / 1024 / 1024,
                    'row_number': row_number
                }
                db_inserted = self.insert_resume_to_db(
                    user_id=user_id,
                    name=name,
                    email=email,
                    phone=phone,
                    resume_text=resume_text,
                    embedding=embedding,
                    additional_data=None,
                    search_id=search_id
                )

            with self.stats_lock:
                self.stats['processed'] += 1
                self.stats['total_size_mb'] += len(content) / 1024 / 1024

            processing_time = time.time() - start_time

            return {
                'row_number': row_number,
                'user_id': user_id,
                'name': name,
                'email': email,
                'phone': phone,
                'resume_url': resume_url,
                'resume_text': resume_text,
                'text_length': len(resume_text) if resume_text else 0,
                'status': 'success',
                'processing_time': processing_time,
                'file_size_mb': len(content) / 1024 / 1024,
                'embedding_generated': embedding_generated,
                'db_inserted': db_inserted
            }
        except Exception as e:
            with self.stats_lock:
                self.stats['errors'] += 1
            logger.error(f"Error processing resume (row {row_data.get('row_number', 'N/A')}): {e}")
            return {
                'row_number': row_data.get('row_number', 0),
                'user_id': user_id,
                'name': row_data.get('name', ''),
                'email': row_data.get('email', ''),
                'phone': row_data.get('phone', ''),
                'resume_url': row_data.get('resume_url', ''),
                'resume_text': f'Error: {str(e)}',
                'text_length': 0,
                'status': 'error',
                'processing_time': 0,
                'embedding_generated': False,
                'db_inserted': False
            }
        finally:
            self.return_session(session)
            gc.collect()

    # ---- FIXED: batch_data first, then user_id, then search_id
    def process_batch(self, batch_data: List[Dict], user_id: str, search_id: Optional[int]) -> List[Dict]:
        """Process a batch of resumes concurrently"""
        results = []
        if not isinstance(batch_data, list):
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_row = {}
            for row_data in batch_data:
                future = executor.submit(self.process_single_resume, row_data, user_id, search_id)
                future_to_row[future] = row_data

            for future in as_completed(future_to_row):
                row_data = future_to_row[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Future execution error for row {row_data.get('row_number', 'N/A')}: {e}")
                    results.append({
                        'row_number': row_data.get('row_number', 0),
                        'user_id': user_id,
                        'name': row_data.get('name', ''),
                        'email': row_data.get('email', ''),
                        'phone': row_data.get('phone', ''),
                        'resume_url': row_data.get('resume_url', ''),
                        'resume_text': f'Processing error: {str(e)}',
                        'text_length': 0,
                        'status': 'error',
                        'processing_time': 0,
                        'embedding_generated': False,
                        'db_inserted': False
                    })
        return results

    def process_drive_spreadsheet_optimized(self, drive_url: str, user_id: str, search_id: Optional[int],
                                           output_file: str = None, progress_callback=None) -> Generator[List[Dict], None, None]:
        self.stats['start_time'] = time.time()
        logger.info("Starting memory-optimized processing...")

        total_processed = 0
        batch_number = 0

        try:
            match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", drive_url)
            if match:
                file_id = match.group(1)
                gid_match = re.search(r"gid=(\d+)", drive_url)
                gid = gid_match.group(1) if gid_match else "0"
                drive_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}"
                logger.info(f"Converted Google Sheet link to CSV export: {drive_url}")

            for chunk_df in self.read_spreadsheet_in_chunks(drive_url, chunk_size=self.batch_size):
                logger.info(f"Processing chunk with {len(chunk_df)} rows")

                columns = list(chunk_df.columns)
                if len(columns) < 4:
                    logger.error(f"Expected at least 4 columns (name, email, phone, resume_url), but found {len(columns)}")
                    logger.error(f"Available columns: {columns}")
                    raise ValueError("Insufficient columns in spreadsheet")

                name_col = columns[0]
                email_col = columns[1]
                phone_col = columns[2]
                resume_col = columns[3]

                logger.info(f"Using columns - Name: '{name_col}', Email: '{email_col}', Phone: '{phone_col}', Resume: '{resume_col}'")

                batch_data = []
                for idx, (_, row) in enumerate(chunk_df.iterrows()):
                    row_number = total_processed + idx + 1
                    # use dict access safely
                    try:
                        resume_val = row[resume_col]
                    except Exception:
                        resume_val = row.get(resume_col) if hasattr(row, 'get') else ""
                    batch_data.append({
                        'row_number': row_number,
                        'name': self.clean_personal_data(row.get(name_col) if hasattr(row, 'get') else row[name_col]),
                        'email': self.clean_personal_data(row.get(email_col) if hasattr(row, 'get') else row[email_col]),
                        'phone': self.clean_personal_data(row.get(phone_col) if hasattr(row, 'get') else row[phone_col]),
                        'resume_url': resume_val
                    })

                # Process batch correctly: batch_data, user_id, search_id
                batch_results = self.process_batch(batch_data, user_id, search_id)

                batch_number += 1
                total_processed += len(batch_results)

                if output_file:
                    self._save_batch_results(batch_results, output_file, batch_number == 1)

                if progress_callback:
                    progress_callback(total_processed, batch_number, self.stats)

                elapsed = time.time() - self.stats['start_time']
                logger.info(f"Batch {batch_number} complete. Total processed: {total_processed}, Errors: {self.stats['errors']}, DB Inserted: {self.stats['db_inserted']}, DB Errors: {self.stats['db_errors']}, Embedding Errors: {self.stats['embedding_errors']}, Time: {elapsed:.1f}s")

                yield batch_results

                del chunk_df, batch_data, batch_results
                gc.collect()

        except Exception as e:
            logger.error(f"Error in optimized processing: {e}")
            raise

    def _save_batch_results(self, results: List[Dict], output_file: str, is_first_batch: bool):
        mode = 'w' if is_first_batch else 'a'
        try:
            if output_file.endswith('.json'):
                if is_first_batch:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                else:
                    try:
                        with open(output_file, 'r') as f:
                            existing = json.load(f)
                        existing.extend(results)
                        with open(output_file, 'w') as f:
                            json.dump(existing, f, indent=2)
                    except:
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
            else:
                df = pd.DataFrame(results)
                df.to_csv(output_file, mode=mode, header=is_first_batch, index=False)
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
