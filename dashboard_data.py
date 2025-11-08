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

# ---------- Small helpers ----------

def get_total_credits_used(user_id):
    # Fetch all creds for the user
    response = (
        supabase.from_("credit_logs")
        .select("deductions")
        .eq("user_id", user_id)
        .execute()
    )
    if not response.data:
        return 0

    # Sum all credits
    total_credits = sum(item.get("deductions", 0) for item in response.data)
    return total_credits


def get_user_name(user_id):
    response = (
        supabase.from_("users")
        .select("name")
        .eq("id", user_id)
        .single()
        .execute()
    )
    return response.data.get("name", "") if response.data else ""

from datetime import date, timedelta

today = date.today()

def get_dashboard_data(org_id):
    today = date.today()

    # Today’s stats
    searches = (
        supabase.from_("search")
        .select("id", count="exact")
        .eq("org_id", org_id)
        .gte("created_at", f"{today}T00:00:00")
        .lte("created_at", f"{today}T23:59:59")
        .execute()
    )
    searches_count = searches.count or 0

    candidates = (
        supabase.from_("candidates")
        .select("id", count="exact")
        .eq("org_id", org_id)
        .gte("created_at", f"{today}T00:00:00")
        .lte("created_at", f"{today}T23:59:59")
        .execute()
    )
    candidates_count = candidates.count or 0

    calls = (
        supabase.from_("calls")
        .select("id", count="exact")
        .eq("org_id", org_id)
        .gte("created_at", f"{today}T00:00:00")
        .lte("created_at", f"{today}T23:59:59")
        .execute()
    )
    calls_count = calls.count or 0

    joinees = (
        supabase.from_("candidates")
        .select("id", count="exact")
        .eq("org_id", org_id)
        .eq("join_status", True)
        .execute()
    )
    joinees_count = joinees.count or 0

    # Weekly activity
    start_date = today - timedelta(days=6)

    weekly_calls = (
        supabase.from_("calls")
        .select("created_at")
        .eq("org_id", org_id)
        .gte("created_at", f"{start_date}T00:00:00")
        .execute()
        .data
        or []
    )

    weekly_searches = (
        supabase.from_("search")
        .select("created_at")
        .eq("org_id", org_id)
        .gte("created_at", f"{start_date}T00:00:00")
        .execute()
        .data
        or []
    )

    weekly_joinees = (
        supabase.from_("candidates")
        .select("created_at, join_status")
        .eq("org_id", org_id)
        .gte("created_at", f"{start_date}T00:00:00")
        .execute()
        .data
        or []
    )

    weekly_activity = []
    for i in range(7):
        day = start_date + timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")

        calls_count_day = sum(1 for c in weekly_calls if c["created_at"][:10] == day_str)
        searches_count_day = sum(1 for s in weekly_searches if s["created_at"][:10] == day_str)
        joinees_count_day = sum(
            1 for j in weekly_joinees if j["created_at"][:10] == day_str and j.get("join_status")
        )

        weekly_activity.append(
            {
                "day": day.strftime("%a"),  # Mon, Tue...
                "calls": calls_count_day,
                "searches": searches_count_day,
                "joinees": joinees_count_day,
            }
        )

    # Flatten "today" stats into root level for frontend compatibility
    return {
        "todays_searches": searches_count,
        "todays_candidates": candidates_count,
        "people_called": calls_count,
        "new_joinees": joinees_count,
        "weekly_activity": weekly_activity,
    }