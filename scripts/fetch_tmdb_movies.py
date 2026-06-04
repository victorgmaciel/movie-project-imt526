"""
fetch_tmdb_movies.py
--------------------
Pull English-language movies from TMDB and save as CSV.
Fetches revenue, budget, runtime, and franchise status from
the movie details endpoint. Adds estimated first-day gross
and estimated ticket count as derived columns.

Label note (Risk 1 per proposal): TMDB total revenue is used as
the outcome variable. Estimated first-day gross = revenue * 0.17
(opening day is typically ~17% of total domestic gross for wide
releases). Estimated tickets = first-day gross / $11.75 (avg US
ticket price 2015-2024). Both are documented estimates.

Output: data/raw/tmdb_metadata.csv

Usage:
    python scripts/fetch_tmdb_movies.py
"""

import os
import time
import json
import requests
import csv
from pathlib import Path

TMDB_API_KEY = os.environ["TMDB_API_KEY"]

START_YEAR  = 2015
END_YEAR    = 2024
MIN_VOTES   = 500
MAX_MOVIES  = 1000
OUT_PATH    = Path("data/raw/tmdb_metadata.csv")

# average US ticket price 2015-2024 (NATO estimate)
AVG_TICKET_PRICE       = 11.75
# opening day as share of total domestic gross (industry estimate)
OPENING_DAY_PCT        = 0.17

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "tmdb_id", "title", "release_date", "genre_ids",
    "popularity", "vote_average", "vote_count",
    "revenue", "budget", "runtime", "is_franchise",
    "estimated_first_day_gross", "estimated_tickets",
]


def fetch_movie_list():
    """Step 1: get list of movies from discover endpoint."""
    movies = []
    page = 1
    base_url = "https://api.themoviedb.org/3/discover/movie"

    print(f"Fetching up to {MAX_MOVIES} movies from TMDB ({START_YEAR}-{END_YEAR})...")

    while len(movies) < MAX_MOVIES:
        params = {
            "api_key":                      TMDB_API_KEY,
            "language":                     "en-US",
            "sort_by":                      "popularity.desc",
            "primary_release_date.gte":     f"{START_YEAR}-01-01",
            "primary_release_date.lte":     f"{END_YEAR}-12-31",
            "vote_count.gte":               MIN_VOTES,
            "with_original_language":       "en",
            "page":                         page,
        }

        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        for m in results:
            movies.append({
                "tmdb_id":      m["id"],
                "title":        m["title"],
                "release_date": m.get("release_date", ""),
                "genre_ids":    json.dumps(m.get("genre_ids", [])),
                "popularity":   m.get("popularity", 0),
                "vote_average": m.get("vote_average", 0),
                "vote_count":   m.get("vote_count", 0),
            })
            if len(movies) >= MAX_MOVIES:
                break

        total_pages = data.get("total_pages", 1)
        print(f"  Page {page}/{min(total_pages, MAX_MOVIES // 20 + 1)} "
              f"— {len(movies)} movies collected")

        if page >= total_pages:
            break
        page += 1
        time.sleep(0.25)

    return movies


def fetch_details(tmdb_id):
    """
    Step 2: hit the movie details endpoint.
    Returns revenue, budget, runtime, and franchise status.
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return 0, 0, 0, 0

    data = resp.json()
    revenue     = data.get("revenue", 0) or 0
    budget      = data.get("budget", 0) or 0
    runtime     = data.get("runtime", 0) or 0
    is_franchise = 1 if data.get("belongs_to_collection") else 0

    return revenue, budget, runtime, is_franchise


def main():
    if OUT_PATH.exists():
        print(f"{OUT_PATH} already exists. Delete it to re-fetch.")
        return

    movies = fetch_movie_list()

    print(f"\nFetching details for {len(movies)} movies...")

    for i, movie in enumerate(movies):
        revenue, budget, runtime, is_franchise = fetch_details(movie["tmdb_id"])

        movie["revenue"]     = revenue
        movie["budget"]      = budget
        movie["runtime"]     = runtime
        movie["is_franchise"] = is_franchise

        # derived label columns
        first_day = round(revenue * OPENING_DAY_PCT, 2) if revenue > 0 else 0
        movie["estimated_first_day_gross"] = first_day
        movie["estimated_tickets"]         = round(first_day / AVG_TICKET_PRICE) if first_day > 0 else 0

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(movies)} done")

        time.sleep(0.25)

    with_revenue = sum(1 for m in movies if m["revenue"] > 0)
    print(f"\n{with_revenue}/{len(movies)} movies have revenue data.")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(movies)

    print(f"Done. {len(movies)} movies saved to {OUT_PATH}")
    print("Next step: run fetch_youtube_data.py")


if __name__ == "__main__":
    main()