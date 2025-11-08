from datetime import datetime
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import requests

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
anon_key: str = os.getenv("SUPABASE_KEY")
service_key: str = os.getenv("SERVICE_ROLE_KEY")
supabase: Client = create_client(url, anon_key)
today_date = datetime.now().date()
prev_call = supabase.from_('calls') \
            .select('id, call_summary, status') \
            .eq('candidate_id', 130) \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()

print(prev_call)