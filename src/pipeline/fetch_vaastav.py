"""
fetch_vaastav.py
----------------
Downloads historical FPL data from the Vaastav GitHub repository
(https://github.com/vaastav/Fantasy-Premier-League) for all available
seasons and saves raw CSVs to data/raw/.

Files fetched per season:
  - cleaned_players.csv   (season-level player overview)
  - gws/merged_gw.csv     (all gameweeks merged into one file)

Usage:
    python src/pipeline/fetch_vaastav.py

Output layout:
    data/raw/
    └── vaastav/
        ├── 2016-17/
        │   ├── cleaned_players.csv
        │   └── gws/
        │       └── merged_gw.csv
        ├── 2017-18/
        │   └── ...
        └── ...
"""

import csv
import io
import logging
import sys
import time
from pathlib import Path

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# All seasons available in the repo, with their correct encoding.
# Seasons up to 2018-19 use latin-1; the rest are utf-8.
# Source: vaastav/Fantasy-Premier-League global_merger.py
SEASONS = {
    "2016-17": "latin-1",
    "2017-18": "latin-1",
    "2018-19": "latin-1",
    "2019-20": "utf-8",
    "2020-21": "utf-8",
    "2021-22": "utf-8",
    "2022-23": "utf-8",
    "2023-24": "utf-8",
    "2024-25": "utf-8",
}

# Files to download, relative to each season's directory in the repo.
# Local path mirrors the repo structure under data/raw/vaastav/<season>/
FILES = [
    "cleaned_players.csv",
    "gws/merged_gw.csv",
]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "vaastav"

HEADERS = {"User-Agent": "fpl-ai-agent/0.1 (portfolio project)"}
REQUEST_TIMEOUT = 15  # seconds
RETRY_DELAY = 2       # seconds between retries
MAX_RETRIES = 3

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Core functions ────────────────────────────────────────────────────────────

def fetch_csv(url: str, encoding: str) -> list[dict] | None:
    """
    Download a CSV from a URL and return it as a list of row dicts.

    Retries up to MAX_RETRIES times on transient failures.
    Returns None if the file is not found (404) or all retries fail.

    Args:
        url:      Full URL to the raw CSV file.
        encoding: Character encoding to decode the response bytes with.

    Returns:
        List of row dictionaries, or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

            if response.status_code == 404:
                log.warning("Not found (404): %s", url)
                return None

            response.raise_for_status()

            # Decode with the season-appropriate encoding, then parse as CSV
            text = response.content.decode(encoding)
            reader = csv.DictReader(io.StringIO(text))
            return list(reader)

        except requests.exceptions.Timeout:
            log.warning("Timeout on attempt %d/%d: %s", attempt, MAX_RETRIES, url)
        except requests.exceptions.RequestException as e:
            log.warning("Request error on attempt %d/%d: %s — %s", attempt, MAX_RETRIES, url, e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    log.error("All %d attempts failed for: %s", MAX_RETRIES, url)
    return None


def save_csv(rows: list[dict], filepath: Path) -> None:
    """
    Write a list of row dicts to a CSV file, re-encoding as utf-8.

    All saved files use utf-8 regardless of their source encoding,
    so downstream code never needs to worry about encoding.

    Args:
        rows:     List of row dictionaries from csv.DictReader.
        filepath: Destination path (parent directories created as needed).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def fetch_season(season: str, encoding: str) -> dict[str, int]:
    """
    Download all target files for one season and save them locally.

    Args:
        season:   Season string, e.g. "2024-25".
        encoding: Character encoding for this season's files.

    Returns:
        Dictionary with counts: {"fetched": N, "skipped": N, "failed": N}
    """
    log.info("── Season %s (%s) ─────────────────────────────", season, encoding)
    counts = {"fetched": 0, "skipped": 0, "failed": 0}

    for relative_path in FILES:
        url = f"{RAW_BASE}/{season}/{relative_path}"
        local_path = OUTPUT_DIR / season / relative_path

        # Skip if already downloaded — avoids re-fetching on reruns
        if local_path.exists():
            log.info("  SKIP  %s (already exists)", relative_path)
            counts["skipped"] += 1
            continue

        log.info("  GET   %s", relative_path)
        rows = fetch_csv(url, encoding)

        if rows is None:
            counts["failed"] += 1
            continue

        save_csv(rows, local_path)
        log.info("  SAVE  %s — %d rows", relative_path, len(rows))
        counts["fetched"] += 1

    return counts


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Starting Vaastav data fetch — %d seasons", len(SEASONS))
    log.info("Output directory: %s", OUTPUT_DIR)

    totals = {"fetched": 0, "skipped": 0, "failed": 0}

    for season, encoding in SEASONS.items():
        counts = fetch_season(season, encoding)
        for key in totals:
            totals[key] += counts[key]

    log.info("── Summary ────────────────────────────────────")
    log.info("  Fetched : %d", totals["fetched"])
    log.info("  Skipped : %d (already on disk)", totals["skipped"])
    log.info("  Failed  : %d", totals["failed"])
    log.info("───────────────────────────────────────────────")

    if totals["failed"] > 0:
        log.warning("%d file(s) could not be downloaded. Check logs above.", totals["failed"])
        sys.exit(1)


if __name__ == "__main__":
    main()