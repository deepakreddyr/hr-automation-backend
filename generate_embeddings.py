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
from typing import List, Dict, Optional, Generator, Tuple
import logging
from openai import OpenAI
from supabase import create_client
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryOptimizedResumeExtractor:
    def __init__(self, max_workers: int = 3, batch_size: int = 50, max_memory_mb: int = 512,
                 openai_api_key: str = None, supabase_url: str = None, supabase_key: str = None,
                 embedding_batch_size: int = 100):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        # -------------------------------------------------------------------------
        # NEW: how many resume texts to embed in a single OpenAI API call.
        # OpenAI allows up to 2048 inputs per request, but 100 is a safe default
        # that stays well under the 300,000 token-per-request limit.
        # -------------------------------------------------------------------------
        self.embedding_batch_size = embedding_batch_size
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

    # =========================================================================
    # NEW: generate_embeddings_batch
    # Replaces the old single-call generate_embedding for bulk usage.
    # =========================================================================
    def generate_embeddings_batch(self, texts: List[str], max_retries: int = 3) -> List[Optional[List[float]]]:
        """
        Send up to self.embedding_batch_size texts in a SINGLE OpenAI API call.

        OpenAI's embeddings endpoint accepts an array of strings as `input`:
            https://platform.openai.com/docs/api-reference/embeddings/create

        Constraints:
        - Up to 2048 inputs per request
        - Max 300,000 tokens summed across all inputs
        - Each individual text still capped at 8,192 tokens

        Returns a list of embeddings in the SAME ORDER as the input texts.
        If a sub-batch fails entirely after retries, None is inserted for each
        text in that sub-batch so callers can handle missing embeddings gracefully.
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return [None] * len(texts)

        # Pre-truncate each text to stay under the per-input token limit
        MAX_CHARS = 8000 * 4  # ~8000 tokens × ~4 chars/token
        truncated_texts = []
        for text in texts:
            if len(text) > MAX_CHARS:
                logger.warning(f"Text truncated from {len(text)} to {MAX_CHARS} chars for embedding")
                truncated_texts.append(text[:MAX_CHARS] + "...(truncated)")
            else:
                truncated_texts.append(text)

        all_embeddings: List[Optional[List[float]]] = [None] * len(truncated_texts)

        # Split into sub-batches of self.embedding_batch_size
        for batch_start in range(0, len(truncated_texts), self.embedding_batch_size):
            batch_end = min(batch_start + self.embedding_batch_size, len(truncated_texts))
            sub_batch = truncated_texts[batch_start:batch_end]

            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=sub_batch          # <-- array of strings, not a single string
                    )
                    # response.data is sorted by .index, but we use .index to be safe
                    for item in response.data:
                        emb = item.embedding
                        if hasattr(emb, 'tolist'):
                            emb = emb.tolist()
                        all_embeddings[batch_start + item.index] = emb

                    logger.info(
                        f"Batch embedded texts [{batch_start}:{batch_end}] "
                        f"({len(sub_batch)} texts) — "
                        f"tokens used: {response.usage.total_tokens}"
                    )
                    break  # success, move to next sub-batch

                except Exception as e:
                    logger.warning(
                        f"Embedding batch [{batch_start}:{batch_end}] "
                        f"attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(
                            f"Embedding batch [{batch_start}:{batch_end}] "
                            f"failed after {max_retries} attempts — filling with None"
                        )
                        with self.stats_lock:
                            self.stats['embedding_errors'] += len(sub_batch)

        return all_embeddings

    # =========================================================================
    # KEPT for backward compat / single-resume edge cases
    # =========================================================================
    def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        results = self.generate_embeddings_batch([text], max_retries=max_retries)
        return results[0] if results else None

    def insert_resume_to_db(self, user_id: str, name: str, email: str, phone: str,
                            resume_text: str, embedding: List[float],
                            additional_data: Dict = None, search_id: Optional[int] = None) -> bool:
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

            if getattr(result, 'status_code', None) and result.status_code >= 400:
                logger.error(f"Supabase insertion responded with status {result.status_code}")
                with self.stats_lock:
                    self.stats['db_errors'] += 1
                return False

            if result.data:
                with self.stats_lock:
                    self.stats['db_inserted'] += 1
                logger.info(f"Successfully inserted resume for {name} (user_id: {user_id})")
                return True
            else:
                logger.error(f"Failed to insert resume for {name}: No data returned")
                with self.stats_lock:
                    self.stats['db_errors'] += 1
                return False
        except Exception as e:
            logger.error(f"Database insertion error for {name} (user_id {user_id}): {e}")
            with self.stats_lock:
                self.stats['db_errors'] += 1
            return False

    # =========================================================================
    # NEW: bulk_insert_resumes_to_db
    # Inserts multiple resume records in a single Supabase call instead of
    # one call per record, which reduces round-trips significantly.
    # =========================================================================
    def bulk_insert_resumes_to_db(self, records: List[Dict]) -> int:
        """
        Insert a list of resume dicts in one Supabase call.
        Each dict must contain: user_id, name, email, phone, resume_text, embedding, search_id.
        Returns the number of successfully inserted records.
        """
        if not self.supabase_client:
            logger.error("Supabase client not initialized")
            return 0
        if not records:
            return 0

        try:
            result = self.supabase_client.table("resumes").insert(records).execute()

            if getattr(result, 'status_code', None) and result.status_code >= 400:
                logger.error(f"Bulk insert failed with status {result.status_code}")
                with self.stats_lock:
                    self.stats['db_errors'] += len(records)
                return 0

            inserted = len(result.data) if result.data else 0
            with self.stats_lock:
                self.stats['db_inserted'] += inserted
            logger.info(f"Bulk inserted {inserted}/{len(records)} resumes")
            return inserted

        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            with self.stats_lock:
                self.stats['db_errors'] += len(records)
            return 0

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
                csv_bytes = io.BytesIO(response.content)
                for chunk in pd.read_csv(csv_bytes, chunksize=chunk_size):
                    yield chunk
            else:
                df = pd.read_excel(io.BytesIO(response.content))
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i:i + chunk_size].copy()
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
                    if sum(len(p) for p in text_parts) > 50000:
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
            for paragraph in doc.paragraphs:
                if total_length >= 50000:
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

    # =========================================================================
    # process_single_resume — NO LONGER generates embeddings or inserts to DB.
    # It only downloads and extracts text. Embedding + DB insert is now done
    # in bulk after all resumes in a batch are downloaded (see process_batch).
    # =========================================================================
    def process_single_resume(self, row_data: Dict, user_id: str, search_id: Optional[int]) -> Dict:
        """
        Download and extract text from a single resume.
        Embedding generation and DB insertion are intentionally removed from here
        and handled in bulk inside process_batch() to minimize API round-trips.
        """
        session = self.get_session()
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
                    'row_number': row_number, 'user_id': user_id, 'name': name,
                    'email': email, 'phone': phone, 'resume_url': resume_url,
                    'resume_text': 'No resume URL provided', 'text_length': 0,
                    'status': 'skipped', 'processing_time': 0,
                    'embedding_generated': False, 'db_inserted': False
                }

            resume_url_str = str(resume_url).strip()

            # ── TEST / INLINE-TEXT MODE ───────────────────────────────────────
            # If the column value is NOT a URL (doesn't start with http/https
            # and isn't a recognisable Drive/Docs link), treat it as the resume
            # text itself.  This lets you test with an Excel that has a
            # "resume_text" column instead of a drive link.
            is_url = (
                resume_url_str.lower().startswith(('http://', 'https://'))
                or 'drive.google.com' in resume_url_str.lower()
                or 'docs.google.com' in resume_url_str.lower()
            )
            if not is_url:
                logger.info(f"Row {row_number}: resume column contains plain text — using directly (test/inline mode)")
                resume_text = resume_url_str
                return {
                    'row_number': row_number, 'user_id': user_id, 'name': name,
                    'email': email, 'phone': phone, 'resume_url': '',
                    'resume_text': resume_text,
                    'text_length': len(resume_text),
                    'status': 'success',
                    'processing_time': 0,
                    'file_size_mb': 0,
                    'file_type': 'text',
                    'embedding_generated': False,
                    'db_inserted': False
                }
            # ─────────────────────────────────────────────────────────────────

            start_time = time.time()
            direct_url = self.convert_drive_link_to_direct(resume_url_str)
            response = session.get(direct_url, stream=True, timeout=30)
            response.raise_for_status()

            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_memory_bytes:
                return {
                    'row_number': row_number, 'user_id': user_id, 'name': name,
                    'email': email, 'phone': phone, 'resume_url': resume_url,
                    'resume_text': f'File too large ({int(content_length)/1024/1024:.2f}MB)',
                    'text_length': 0, 'status': 'too_large',
                    'processing_time': time.time() - start_time,
                    'embedding_generated': False, 'db_inserted': False
                }

            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                content += chunk
                if len(content) > self.max_memory_bytes:
                    return {
                        'row_number': row_number, 'user_id': user_id, 'name': name,
                        'email': email, 'phone': phone, 'resume_url': resume_url,
                        'resume_text': 'File too large - download stopped',
                        'text_length': 0, 'status': 'too_large',
                        'processing_time': time.time() - start_time,
                        'embedding_generated': False, 'db_inserted': False
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
            file_size_mb = len(content) / 1024 / 1024

            with self.stats_lock:
                self.stats['total_size_mb'] += file_size_mb

            return {
                'row_number': row_number, 'user_id': user_id, 'name': name,
                'email': email, 'phone': phone, 'resume_url': resume_url,
                'resume_text': resume_text,
                'text_length': len(resume_text) if resume_text else 0,
                'status': 'success',
                'processing_time': time.time() - start_time,
                'file_size_mb': file_size_mb,
                'file_type': file_type,
                'embedding_generated': False,  # will be set after batch embed
                'db_inserted': False            # will be set after bulk insert
            }

        except Exception as e:
            with self.stats_lock:
                self.stats['errors'] += 1
            logger.error(f"Error processing resume (row {row_data.get('row_number', 'N/A')}): {e}")
            return {
                'row_number': row_data.get('row_number', 0), 'user_id': user_id,
                'name': row_data.get('name', ''), 'email': row_data.get('email', ''),
                'phone': row_data.get('phone', ''), 'resume_url': row_data.get('resume_url', ''),
                'resume_text': f'Error: {str(e)}', 'text_length': 0,
                'status': 'error', 'processing_time': 0,
                'embedding_generated': False, 'db_inserted': False
            }
        finally:
            self.return_session(session)
            gc.collect()

    # =========================================================================
    # process_batch — UPDATED
    # Phase 1: Download + extract text in parallel (ThreadPoolExecutor)
    # Phase 2: Batch embed ALL successful texts in one/few OpenAI calls
    # Phase 3: Bulk insert ALL records into Supabase in one call
    # =========================================================================
    def process_batch(self, batch_data: List[Dict], user_id: str, search_id: Optional[int]) -> List[Dict]:
        """
        Process a batch of resumes in three phases:
        1. Parallel download + text extraction (unchanged concurrency)
        2. Batch embedding  — ONE OpenAI call for up to 100 texts (configurable)
        3. Bulk DB insert   — ONE Supabase call for all records in the batch
        """
        results: List[Dict] = []
        if not isinstance(batch_data, list):
            return results

        # ── Phase 1: Parallel download + text extraction ──────────────────────
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_row = {
                executor.submit(self.process_single_resume, row, user_id, search_id): row
                for row in batch_data
            }
            for future in as_completed(future_to_row):
                row_data = future_to_row[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Future error row {row_data.get('row_number', 'N/A')}: {e}")
                    results.append({
                        'row_number': row_data.get('row_number', 0), 'user_id': user_id,
                        'name': row_data.get('name', ''), 'email': row_data.get('email', ''),
                        'phone': row_data.get('phone', ''), 'resume_url': row_data.get('resume_url', ''),
                        'resume_text': f'Processing error: {str(e)}', 'text_length': 0,
                        'status': 'error', 'processing_time': 0,
                        'embedding_generated': False, 'db_inserted': False
                    })

        with self.stats_lock:
            self.stats['processed'] += len(results)

        # ── Phase 2: Batch embedding ───────────────────────────────────────────
        # Collect only the results that have usable text
        embeddable_indices = [
            i for i, r in enumerate(results)
            if r['status'] == 'success'
            and r.get('resume_text')
            and len(r['resume_text'].strip()) > 50
            and not r['resume_text'].startswith('Error')
        ]

        if embeddable_indices:
            texts_to_embed = [results[i]['resume_text'] for i in embeddable_indices]

            logger.info(
                f"Batch embedding {len(texts_to_embed)} texts "
                f"in chunks of {self.embedding_batch_size} → "
                f"~{max(1, len(texts_to_embed) // self.embedding_batch_size)} API call(s)"
            )

            # Single call (or a few calls if > embedding_batch_size texts)
            embeddings = self.generate_embeddings_batch(texts_to_embed)

            # Write embeddings back into results
            for list_pos, result_idx in enumerate(embeddable_indices):
                emb = embeddings[list_pos]
                if emb is not None:
                    results[result_idx]['embedding'] = emb
                    results[result_idx]['embedding_generated'] = True

        # ── Phase 3: Bulk DB insert ────────────────────────────────────────────
        records_to_insert = [
            {
                "user_id": r['user_id'],
                "name": r['name'],
                "email": r['email'],
                "phone": r['phone'],
                "resume_text": r['resume_text'],
                "embedding": r['embedding'],
                "search_id": search_id
            }
            for r in results
            if r.get('embedding_generated') and r.get('embedding') and user_id
        ]

        if records_to_insert:
            logger.info(f"Bulk inserting {len(records_to_insert)} records into Supabase")
            inserted_count = self.bulk_insert_resumes_to_db(records_to_insert)

            # Mark which results were successfully inserted
            inserted_set = set()
            for i, r in enumerate(results):
                if r.get('embedding_generated'):
                    inserted_set.add(i)

            success_iter = iter(range(inserted_count))
            for i in inserted_set:
                try:
                    next(success_iter)
                    results[i]['db_inserted'] = True
                except StopIteration:
                    break

        return results

    def process_drive_spreadsheet_optimized(self, drive_url: str, user_id: str, search_id: Optional[int],
                                            output_file: str = None, progress_callback=None) -> Generator[List[Dict], None, None]:
        self.stats['start_time'] = time.time()
        logger.info("Starting memory-optimized processing with batch embeddings...")

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
                    logger.error(f"Expected at least 4 columns, found {len(columns)}: {columns}")
                    raise ValueError("Insufficient columns in spreadsheet")

                name_col, email_col, phone_col, resume_col = columns[0], columns[1], columns[2], columns[3]
                logger.info(f"Columns → Name:'{name_col}' Email:'{email_col}' Phone:'{phone_col}' Resume:'{resume_col}'")

                batch_data = []
                for idx, (_, row) in enumerate(chunk_df.iterrows()):
                    row_number = total_processed + idx + 1
                    batch_data.append({
                        'row_number': row_number,
                        'name': self.clean_personal_data(row[name_col]),
                        'email': self.clean_personal_data(row[email_col]),
                        'phone': self.clean_personal_data(row[phone_col]),
                        'resume_url': row[resume_col]
                    })

                batch_results = self.process_batch(batch_data, user_id, search_id)

                batch_number += 1
                total_processed += len(batch_results)

                if output_file:
                    self._save_batch_results(batch_results, output_file, batch_number == 1)

                if progress_callback:
                    progress_callback(total_processed, batch_number, self.stats)

                elapsed = time.time() - self.stats['start_time']
                logger.info(
                    f"Batch {batch_number} complete | Total: {total_processed} | "
                    f"Errors: {self.stats['errors']} | DB Inserted: {self.stats['db_inserted']} | "
                    f"Embed Errors: {self.stats['embedding_errors']} | Time: {elapsed:.1f}s"
                )

                yield batch_results

                del chunk_df, batch_data, batch_results
                gc.collect()

        except Exception as e:
            logger.error(f"Error in optimized processing: {e}")
            raise

    def process_excel_file_optimized(self, file_bytes: bytes, filename: str, user_id: str,
                                      search_id: Optional[int],
                                      output_file: str = None,
                                      progress_callback=None) -> Generator[List[Dict], None, None]:
        """
        Process an uploaded Excel (.xlsx) or CSV file the same way as
        process_drive_spreadsheet_optimized.

        Expected column order (by position, not name):
            0 → Name
            1 → Email
            2 → Phone
            3 → Resume URL / Drive link

        Args:
            file_bytes: Raw bytes from the uploaded file (request.files[...].read()).
            filename:   Original filename so we can detect .csv vs .xlsx.
            user_id / search_id: Passed through to process_batch / DB.
        """
        self.stats['start_time'] = time.time()
        logger.info(f"Starting Excel/CSV file processing for '{filename}'...")

        total_processed = 0
        batch_number = 0

        try:
            buf = io.BytesIO(file_bytes)
            fname_lower = filename.lower()

            if fname_lower.endswith('.csv'):
                chunks = pd.read_csv(buf, chunksize=self.batch_size)
            else:
                # .xlsx / .xls
                full_df = pd.read_excel(buf)
                # Simulate chunked iteration to reuse the same processing loop
                chunks = (
                    full_df.iloc[i:i + self.batch_size].copy()
                    for i in range(0, len(full_df), self.batch_size)
                )

            for chunk_df in chunks:
                logger.info(f"Processing chunk with {len(chunk_df)} rows")

                columns = list(chunk_df.columns)
                if len(columns) < 4:
                    logger.error(f"Expected at least 4 columns, found {len(columns)}: {columns}")
                    raise ValueError(
                        f"Uploaded file must have at least 4 columns "
                        f"(Name, Email, Phone, ResumeURL). Found: {columns}"
                    )

                name_col, email_col, phone_col, resume_col = (
                    columns[0], columns[1], columns[2], columns[3]
                )
                logger.info(
                    f"Columns → Name:'{name_col}' Email:'{email_col}' "
                    f"Phone:'{phone_col}' Resume:'{resume_col}'"
                )

                batch_data = []
                for idx, (_, row) in enumerate(chunk_df.iterrows()):
                    row_number = total_processed + idx + 1
                    batch_data.append({
                        'row_number': row_number,
                        'name': self.clean_personal_data(row[name_col]),
                        'email': self.clean_personal_data(row[email_col]),
                        'phone': self.clean_personal_data(row[phone_col]),
                        'resume_url': row[resume_col]
                    })

                batch_results = self.process_batch(batch_data, user_id, search_id)

                batch_number += 1
                total_processed += len(batch_results)

                if output_file:
                    self._save_batch_results(batch_results, output_file, batch_number == 1)

                if progress_callback:
                    progress_callback(total_processed, batch_number, self.stats)

                elapsed = time.time() - self.stats['start_time']
                logger.info(
                    f"Batch {batch_number} complete | Total: {total_processed} | "
                    f"Errors: {self.stats['errors']} | DB Inserted: {self.stats['db_inserted']} | "
                    f"Embed Errors: {self.stats['embedding_errors']} | Time: {elapsed:.1f}s"
                )

                yield batch_results

                del chunk_df, batch_data, batch_results
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing uploaded Excel/CSV file: {e}")
            raise

    def _save_batch_results(self, results: List[Dict], output_file: str, is_first_batch: bool):

        mode = 'w' if is_first_batch else 'a'
        try:
            # Strip the embedding from output to keep files small
            slim_results = [{k: v for k, v in r.items() if k != 'embedding'} for r in results]
            if output_file.endswith('.json'):
                if is_first_batch:
                    with open(output_file, 'w') as f:
                        json.dump(slim_results, f, indent=2)
                else:
                    try:
                        with open(output_file, 'r') as f:
                            existing = json.load(f)
                        existing.extend(slim_results)
                        with open(output_file, 'w') as f:
                            json.dump(existing, f, indent=2)
                    except Exception:
                        with open(output_file, 'w') as f:
                            json.dump(slim_results, f, indent=2)
            else:
                df = pd.DataFrame(slim_results)
                df.to_csv(output_file, mode=mode, header=is_first_batch, index=False)
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")