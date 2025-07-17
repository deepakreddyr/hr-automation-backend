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

def get_todays_searches(user_id):
    response = supabase.from_('search') \
        .select('id', count='exact') \
        .eq('user_id', user_id) \
        .gte('created_at', f"{today_date}T00:00:00") \
        .lte('created_at', f"{today_date}T23:59:59") \
        .execute()
    return response.count or 0

def get_todays_candidates(user_id):
    response = supabase.from_('candidates') \
        .select('id', count='exact') \
        .eq('user_id', user_id) \
        .gte('created_at', f"{today_date}T00:00:00") \
        .lte('created_at', f"{today_date}T23:59:59") \
        .execute()
    return response.count or 0

def get_new_joinees(user_id):
    response = supabase.from_('candidates') \
        .select('id', count='exact') \
        .eq('user_id', user_id) \
        .eq('join_status', True) \
        .execute()
    return response.count or 0

def get_creds_used(user_id):
    response = supabase.from_('history') \
        .select('creds') \
        .eq('user_id', user_id) \
        .order('created_at', desc=True) \
        .limit(1) \
        .execute()
    if response.data and len(response.data) > 0:
        return response.data[0].get('creds', 0)
    return 0

def get_user_name(user_id):
    response = supabase.from_('users') \
        .select('name') \
        .eq('id', user_id) \
        .single() \
        .execute()
    return response.data.get('name', '') if response.data else ''

def get_people_called(user_id):
    response = supabase.from_('calls') \
        .select('id', count='exact') \
        .eq('user_id', user_id) \
        .gte('created_at', f"{today_date}T00:00:00") \
        .lte('created_at', f"{today_date}T23:59:59") \
        .execute()
    return response.count or 0

from datetime import timedelta

def get_weekly_activity(user_id):
    results = []

    for i in range(7):
        day = today_date - timedelta(days=6 - i)
        start = f"{day}T00:00:00"
        end = f"{day}T23:59:59"

        # Count calls
        calls = supabase.from_('calls').select('id', count='exact') \
            .eq('user_id', user_id).gte('created_at', start).lte('created_at', end).execute().count or 0

        # Count searches
        searches = supabase.from_('search').select('id', count='exact') \
            .eq('user_id', user_id).gte('created_at', start).lte('created_at', end).execute().count or 0

        # Count joinees
        joinees = supabase.from_('candidates').select('id', count='exact') \
            .eq('user_id', user_id).eq('join_status', True) \
            .gte('created_at', start).lte('created_at', end).execute().count or 0

        results.append({
            "day": day.strftime("%a"),  # e.g., Mon, Tue
            "calls": calls,
            "searches": searches,
            "joinees": joinees,
        })
    
    # print(results)
    return results

