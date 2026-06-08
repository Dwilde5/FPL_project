"""
parse_vaastav.py
----------------
Reads raw Vaastav CSVs from data/raw/vaastav/, normalises the schema across
all seasons, and writes clean Parquet files to data/processed/.

The Vaastav dataset spans three schema eras:
  Era 1 — 2016-17 to 2020-21: no position/team in merged_gw, no xG/xA cols
  Era 2 — 2021-22 to 2021-22: position/team/xP present, still no xG/xA
  Era 3 — 2022-23 onwards:    full schema including xG, xA, starts, etc.

This script reconciles those differences so downstream models receive a
consistent DataFrame regardless of which seasons they query.

Usage:
    python src/pipeline/parse_vaastav.py

Output:
    data/processed/
    ├── players.parquet      (season-level overview, all seasons stacked)
    └── gameweeks.parquet    (gameweek-level data, all seasons stacked)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "vaastav"
PROCESSED_DIR = ROOT / "data" / "processed"

SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

# Encoding per season (latin-1 for early seasons; utf-8 thereafter)
SEASON_ENCODING = {s: "latin-1" if s <= "2018-19" else "utf-8" for s in SEASONS}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Column definitions ────────────────────────────────────────────────────────

# Columns we keep from cleaned_players.csv, renamed for consistency
PLAYERS_RENAME = {
    "first_name":          "first_name",
    "second_name":         "second_name",
    "goals_scored":        "goals_scored",
    "assists":             "assists",
    "total_points":        "total_points",
    "minutes":             "minutes",
    "goals_conceded":      "goals_conceded",
    "creativity":          "creativity",
    "influence":           "influence",
    "threat":              "threat",
    "bonus":               "bonus",
    "bps":                 "bps",
    "ict_index":           "ict_index",
    "clean_sheets":        "clean_sheets",
    "red_cards":           "red_cards",
    "yellow_cards":        "yellow_cards",
    "selected_by_percent": "selected_by_percent",
    # Present from 2017-18 onwards; filled with NaN for 2016-17
    "now_cost":            "now_cost",
    "element_type":        "position_id",
}

# Columns we keep from merged_gw.csv
# Columns absent in a given era are filled with NaN
GW_KEEP = [
    "name",
    "element",
    "round",
    "GW",
    "kickoff_time",
    "position",          # Era 2+ only
    "team",              # Era 2+ only
    "opponent_team",
    "was_home",
    "minutes",
    "starts",            # Era 3 only
    "total_points",
    "xP",                # Era 2+ only
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",             # Era 3 only
    "expected_assists",           # Era 3 only
    "expected_goal_involvements", # Era 3 only
    "expected_goals_conceded",    # Era 3 only
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "team_h_score",
    "team_a_score",
]

# Columns that should be integers after parsing
GW_INT_COLS = [
    "element", "round", "GW", "minutes", "total_points",
    "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "selected", "transfers_in", "transfers_out", "transfers_balance",
    "team_h_score", "team_a_score",
]

# Columns that should be floats
GW_FLOAT_COLS = [
    "xP", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "value",
]

PLAYERS_FLOAT_COLS = [
    "creativity", "influence", "threat", "ict_index", "selected_by_percent",
]
PLAYERS_INT_COLS = [
    "goals_scored", "assists", "total_points", "minutes", "goals_conceded",
    "bonus", "bps", "clean_sheets", "red_cards", "yellow_cards",
    "now_cost", "position_id",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_csv(path: Path, encoding: str = "utf-8") -> pd.DataFrame | None:
    """
    Read a CSV safely, returning None if the file does not exist.

    fetch_vaastav.py normalises all saved files to utf-8 regardless of their
    original source encoding, so we always read as utf-8 here. The encoding
    parameter is retained for callers that may pass it explicitly, but it is
    ignored in favour of utf-8.

    Args:
        path:     Path to the CSV file.
        encoding: Unused; retained for API compatibility.

    Returns:
        DataFrame, or None if the file is missing.
    """
    if not path.exists():
        log.warning("Missing file: %s", path)
        return None

    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        log.error("Failed to read %s: %s", path, e)
        return None


def enforce_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Restrict a DataFrame to the specified columns, adding missing ones as NaN.

    Args:
        df:      Input DataFrame.
        columns: Ordered list of desired column names.

    Returns:
        DataFrame with exactly the specified columns in order.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def cast_columns(
    df: pd.DataFrame,
    int_cols: list[str],
    float_cols: list[str],
) -> pd.DataFrame:
    """
    Cast columns to their target numeric types, coercing errors to NaN.

    Uses nullable integer dtype (Int64) so integer columns can hold NaN
    without being silently promoted to float.

    Args:
        df:         Input DataFrame.
        int_cols:   Columns to cast to Int64.
        float_cols: Columns to cast to float64.

    Returns:
        DataFrame with corrected dtypes.
    """
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


# ── Per-season parsers ────────────────────────────────────────────────────────

def parse_players(season: str) -> pd.DataFrame | None:
    """
    Parse cleaned_players.csv for one season.

    Renames columns for consistency, adds a season label, and fills
    columns absent in early seasons (now_cost, position_id) with NaN.

    Args:
        season: Season string, e.g. "2024-25".

    Returns:
        Cleaned DataFrame, or None if the file could not be read.
    """
    path = RAW_DIR / season / "cleaned_players.csv"
    enc = SEASON_ENCODING[season]
    df = read_csv(path, enc)
    if df is None:
        return None

    # Keep only columns we have a mapping for; add missing ones as NaN
    available = {k: v for k, v in PLAYERS_RENAME.items() if k in df.columns}
    df = df.rename(columns=available)

    # Ensure all target columns are present
    for target in PLAYERS_RENAME.values():
        if target not in df.columns:
            df[target] = pd.NA

    df = df[list(PLAYERS_RENAME.values())]
    df = cast_columns(df, PLAYERS_INT_COLS, PLAYERS_FLOAT_COLS)
    df.insert(0, "season", season)

    log.info("  players   %s — %d rows, %d cols", season, len(df), df.shape[1])
    return df


def parse_gameweeks(season: str) -> pd.DataFrame | None:
    """
    Parse merged_gw.csv for one season.

    Reconciles the three schema eras by enforcing a fixed column set,
    filling absent columns with NaN and dropping columns not in the set.

    Args:
        season: Season string, e.g. "2024-25".

    Returns:
        Cleaned DataFrame, or None if the file could not be read.
    """
    path = RAW_DIR / season / "gws" / "merged_gw.csv"
    enc = SEASON_ENCODING[season]
    df = read_csv(path, enc)
    if df is None:
        return None

    # Drop manager rows introduced in 2024-25 (name starts with a digit or
    # position is 'MNG') — they follow a different scoring schema
    if "position" in df.columns:
        df = df[df["position"] != "MNG"]

    df = enforce_columns(df, GW_KEEP)
    df = cast_columns(df, GW_INT_COLS, GW_FLOAT_COLS)

    # Parse kickoff_time as a proper datetime (UTC)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")

    # was_home: coerce to boolean
    df["was_home"] = df["was_home"].map(
        {"True": True, "False": False, True: True, False: False}
    ).astype("boolean")

    df.insert(0, "season", season)

    log.info("  gameweeks %s — %d rows, %d cols", season, len(df), df.shape[1])
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    players_frames: list[pd.DataFrame] = []
    gw_frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for season in SEASONS:
        log.info("── Parsing %s ──────────────────────────────────", season)

        players = parse_players(season)
        if players is not None:
            players_frames.append(players)
        else:
            failed.append(f"{season}/cleaned_players")

        gws = parse_gameweeks(season)
        if gws is not None:
            gw_frames.append(gws)
        else:
            failed.append(f"{season}/merged_gw")

    # Stack all seasons and write Parquet
    if players_frames:
        players_all = pd.concat(players_frames, ignore_index=True)
        out = PROCESSED_DIR / "players.parquet"
        players_all.to_parquet(out, index=False)
        log.info("Wrote %s — %d rows", out, len(players_all))
    else:
        log.error("No player data to write")

    if gw_frames:
        gws_all = pd.concat(gw_frames, ignore_index=True)
        out = PROCESSED_DIR / "gameweeks.parquet"
        gws_all.to_parquet(out, index=False)
        log.info("Wrote %s — %d rows", out, len(gws_all))
    else:
        log.error("No gameweek data to write")

    # Summary
    log.info("── Summary ─────────────────────────────────────")
    if players_frames:
        log.info("  players.parquet   : %d rows across %d seasons",
                 len(players_all), len(players_frames))
    if gw_frames:
        log.info("  gameweeks.parquet : %d rows across %d seasons",
                 len(gws_all), len(gw_frames))
    if failed:
        log.warning("  Failed files      : %s", ", ".join(failed))
        sys.exit(1)
    log.info("───────────────────────────────────────────────")


if __name__ == "__main__":
    main()