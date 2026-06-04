"""
fetch_youtube_data.py
---------------------
Multi-key rotation version.
Rotates through 8 API keys to maximize daily quota.
Targets 500 movies (proposal minimum).

Usage:
    cd ~/trailer_analysis
    python fetch_youtube_data.py
"""

import csv
import json
import time
import requests
from datetime import date
from pathlib import Path

API_KEYS = [
    "AIzaSyCA98_VUwomQo-aKu-9mLdl_rU_lPGS_Xs",
    "AIzaSyAOQsVLxa5xS_9WlivIXNzEdWs9pyofIhI",
    "AIzaSyAOQsVLxa5xS_9WlivIXNzEdWs9pyofIhI",
    "AIzaSyAasuKBDfg1s_aIUuJoHxYoyNMseOjoMLQ",
    "AIzaSyD_Jgs5SBv0MILPNATieA-QDlL9Gi1-Yjs",
    "AIzaSyC5943CBjdS5cxlDcSm9eK_LP3BpRNuZDc",
    "AIzaSyBaMWiFCKd2e1thSTOFde5gCwGbCtypJ6o",
    "AIzaSyA8fj5R2CFa-0Xd2aGOcOPQA_6T1qpHhZ4",
]

TMDB_CSV  = Path("data/raw/tmdb_metadata.csv")
OUT_PATH  = Path("data/raw/youtube_batches.jsonl")
QUOTA_LOG = Path("logs/quota_usage.json")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)

QUOTA_PER_KEY  = 9500
QUOTA_SEARCH   = 100
QUOTA_COMMENTS = 1
MIN_COMMENTS   = 500
MAX_COMMENTS   = 5000
MAX_MOVIES     = 800
QUOTA_DELAY    = 0.5

BASE = "https://www.googleapis.com/youtube/v3"


# ── QUOTA 

def load_quota_log():
    today = str(date.today())
    if QUOTA_LOG.exists():
        with open(QUOTA_LOG) as f:
            log = json.load(f)
        if log.get("date") == today:
            return log.get("key_usage", {str(i): 0 for i in range(len(API_KEYS))})
    return {str(i): 0 for i in range(len(API_KEYS))}


def save_quota_log(key_usage):
    with open(QUOTA_LOG, "w") as f:
        json.dump({"date": str(date.today()), "key_usage": key_usage}, f)


def get_active_key(key_usage):
    """Return (key_index, api_key) for the next key with quota remaining."""
    for i, key in enumerate(API_KEYS):
        if key_usage[str(i)] + QUOTA_SEARCH <= QUOTA_PER_KEY:
            return i, key
    return None, None


# ── RESUME 

def load_completed_ids():
    if not OUT_PATH.exists():
        return set()
    completed = set()
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                completed.add(json.loads(line)["movie_id"])
            except Exception:
                pass
    return completed


# ── YOUTUBE

def find_trailer_video_id(title, year, api_key):
    params = {
        "key":               api_key,
        "q":                 f"{title} {year} official trailer",
        "part":              "id,snippet",
        "type":              "video",
        "maxResults":        5,
        "relevanceLanguage": "en",
    }
    resp = requests.get(f"{BASE}/search", params=params)
    if resp.status_code != 200:
        print(f"  Search error {resp.status_code}: "
              f"{resp.json().get('error', {}).get('message', '')}")
        return None, None

    items = resp.json().get("items", [])
    for item in items:
        vid_title = item["snippet"].get("title", "").lower()
        channel   = item["snippet"].get("channelTitle", "").lower()
        if "official" in vid_title or "trailer" in vid_title or "official" in channel:
            return item["id"]["videoId"], item["snippet"]["channelId"]

    if items:
        return items[0]["id"]["videoId"], items[0]["snippet"]["channelId"]

    return None, None


def fetch_comments(video_id, api_key, quota_remaining):
    """
    Pull top-level comments using commentThreads.list per proposal spec:
      - part=snippet
      - maxResults=100
      - nextPageToken for pagination
    Only stores: video_id, comment_id, timestamp, like_count, text.
    """
    comments    = []
    page_token  = None
    units_spent = 0
    max_pages   = min(MAX_COMMENTS // 100, quota_remaining)

    while len(comments) < MAX_COMMENTS and units_spent < max_pages:
        params = {
            "key":        api_key,
            "videoId":    video_id,
            "part":       "snippet",
            "maxResults": 100,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(f"{BASE}/commentThreads", params=params)
        units_spent += QUOTA_COMMENTS

        if resp.status_code == 403:
            print("    Comments disabled on this video.")
            break
        if resp.status_code != 200:
            print(f"    Comment error {resp.status_code}")
            break

        data = resp.json()
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text    = snippet.get("textDisplay", "")
            if len(text) > 20:
                comments.append({
                    "video_id":   video_id,
                    "comment_id": item["id"],
                    "timestamp":  snippet.get("publishedAt", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "text":       text,
                })
            if len(comments) >= MAX_COMMENTS:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(QUOTA_DELAY)

    return comments, units_spent


# ── MAIN 

def main():
    if not TMDB_CSV.exists():
        print(f"ERROR: {TMDB_CSV} not found.")
        return

    with open(TMDB_CSV, "r", encoding="utf-8") as f:
        movies = list(csv.DictReader(f))[:MAX_MOVIES]

    completed = load_completed_ids()
    key_usage = load_quota_log()

    total_remaining = sum(
        QUOTA_PER_KEY - key_usage[str(i)] for i in range(len(API_KEYS))
    )
    print(f"Loaded {len(movies)} movies (capped at {MAX_MOVIES}).")
    print(f"Already collected: {len(completed)} movies.")
    print(f"Total quota remaining across all 8 keys: {total_remaining} units\n")

    stats = {"success": 0, "no_trailer": 0, "no_comments": 0,
             "skipped": 0, "quota_stop": 0}

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for i, movie in enumerate(movies):
            movie_id = str(movie["tmdb_id"])

            if movie_id in completed:
                stats["skipped"] += 1
                continue

            key_idx, api_key = get_active_key(key_usage)
            if api_key is None:
                print("\nAll keys exhausted for today. Re-run tomorrow.")
                stats["quota_stop"] += 1
                break

            year = movie["release_date"][:4] if movie["release_date"] else ""
            print(f"[{i+1}/{len(movies)}] {movie['title']} ({year}) "
                  f"| key {key_idx+1} used: {key_usage[str(key_idx)]}")

            video_id, channel_id = find_trailer_video_id(movie["title"], year, api_key)
            key_usage[str(key_idx)] += QUOTA_SEARCH
            save_quota_log(key_usage)
            time.sleep(QUOTA_DELAY)

            if not video_id:
                print("  No trailer found — skipping.")
                stats["no_trailer"] += 1
                continue

            quota_remaining = QUOTA_PER_KEY - key_usage[str(key_idx)]
            comments, comment_units = fetch_comments(video_id, api_key, quota_remaining)
            key_usage[str(key_idx)] += comment_units
            save_quota_log(key_usage)
            time.sleep(QUOTA_DELAY)

            if len(comments) < MIN_COMMENTS:
                print(f"  Only {len(comments)} comments — skipping.")
                stats["no_comments"] += 1
                continue

            record = {
                "movie_id":      movie_id,
                "title":         movie["title"],
                "release_date":  movie["release_date"],
                "video_id":      video_id,
                "channel_id":    channel_id,
                "comment_count": len(comments),
                "comments":      comments,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"  {len(comments)} comments saved ({comment_units} units).")
            stats["success"] += 1

    print("\n── Session complete ──")
    print(f"  Success:      {stats['success']}")
    print(f"  No trailer:   {stats['no_trailer']}")
    print(f"  No comments:  {stats['no_comments']}")
    print(f"  Skipped:      {stats['skipped']} (already collected)")
    print(f"  Quota stop:   {stats['quota_stop']}")
    print(f"\nKey usage summary:")
    for i in range(len(API_KEYS)):
        used = key_usage[str(i)]
        print(f"  Key {i+1}: {used} / {QUOTA_PER_KEY} units used")


if __name__ == "__main__":
    main()