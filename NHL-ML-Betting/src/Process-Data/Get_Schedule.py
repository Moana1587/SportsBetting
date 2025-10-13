"""
Create 2025–26 NBA season CSV with columns:
Match Number, Round Number, Date, Location, Home Team, Away Team, Result

It reads the official scheduleleaguev2 JSON (either cdn.nba.com or nba.cloud mirror)
and normalizes tip time to UTC ISO8601 (…Z).
"""

import csv, datetime, json
from typing import Dict, Any, List

import requests

# Primary + fallback endpoints (your screenshot shows the nba.cloud mirror)
URLS = [
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
    "http://nba.cloud/league/00/2025-26/scheduleleaguev2?format=json",
]

OUT_CSV = "../../Data/nba-2025-UTC.csv"
SEASON_STR = "2025-26"  # we only keep this season label
KEEP_SEASON_TYPES = {"Preseason", "Regular Season", "NBA Cup", "Playoffs"}  # adjust if needed


def fetch_schedule() -> Dict[str, Any]:
    last_err = None
    for url in URLS:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to fetch schedule from all endpoints: {last_err}")


def to_dd_mm_yy_hh_mm(g: Dict[str, Any]) -> str:
    """
    Best-effort normalization to dd/mm/yy HH:MM format.
    Prefer gameDateTimeUTC if present; otherwise, try to combine date+time fields.
    """
    # 1) Direct UTC datetime if available
    val = g.get("gameDateTimeUTC") or g.get("gameUtc") or g.get("gameTimeUTC")
    if val:
        s = val.replace("Z", "+00:00")  # allow fromisoformat parsing
        dt = datetime.datetime.fromisoformat(s)
        utc_dt = dt.astimezone(datetime.timezone.utc)
        return utc_dt.strftime("%d/%m/%y %H:%M")

    # 2) Fallback: combine separate date/time (UTC first, then EST)
    date_utc = g.get("gameDateUTC")
    time_utc = g.get("gameTimeUTC")
    if date_utc and time_utc:
        s = f"{date_utc} {time_utc}".replace("Z", "").strip()
        # many feeds already ISO; handle simple "YYYY-MM-DDTHH:MM:SS" too
        s = s.replace("T", " ")
        dt = datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
        return dt.strftime("%d/%m/%y %H:%M")

    date_est = g.get("gameDateEst") or g.get("gameDate")
    time_est = g.get("gameTimeEst") or g.get("gameTime")
    if date_est and time_est:
        # Treat as US/Eastern w/o DST db; assume EST≈UTC-5, but many feeds also give ISO strings
        # If it's already ISO with Z/offset, fromisoformat will do the right thing.
        s = f"{date_est} {time_est}".replace("Z", "+00:00").replace("T", " ")
        try:
            dt = datetime.datetime.fromisoformat(s)
            # If no tzinfo, assume ET and convert to UTC; this is a fallback.
            if dt.tzinfo is None:
                # naive: treat as ET (UTC-5 / -4). Without tzdb we assume -5 year-round (best effort).
                dt = dt.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
            utc_dt = dt.astimezone(datetime.timezone.utc)
            return utc_dt.strftime("%d/%m/%y %H:%M")
        except Exception:
            pass

    # 3) If all else fails, return empty
    return ""


def location_str(g: Dict[str, Any]) -> str:
    arena = g.get("arenaName") or ""
    city = g.get("arenaCity") or ""
    state = g.get("arenaState") or g.get("arenaCountry") or ""
    parts = [p for p in [arena, city, state] if p]
    return ", ".join(parts)


def team_name(team_obj: Dict[str, Any]) -> str:
    # Combine city and team name for full team name (e.g., "New Orleans Pelicans")
    city = team_obj.get("teamCity", "")
    name = team_obj.get("teamName", "")
    
    if city and name:
        return f"{city} {name}"
    elif name:
        return name
    elif city:
        return city
    else:
        # Fallback to tricode or slug
        return team_obj.get("teamTricode") or team_obj.get("teamSlug") or ""


def main():
    data = fetch_schedule()

    # The object can be either top-level "leagueSchedule" or nested.
    league = data.get("leagueSchedule") or data.get("leagueScheduleV2") or data
    if not league:
        raise RuntimeError("Missing leagueSchedule in response.")

    season_year = league.get("seasonYear", "")
    if season_year and season_year != SEASON_STR:
        # Some endpoints include multiple seasons; we’ll still filter per-game below.
        pass

    out_rows: List[Dict[str, Any]] = []
    for day in league.get("gameDates", []):
        for g in day.get("games", []):
            # Season filter
            season_label = g.get("season") or league.get("seasonYear", "")
            if season_label and not season_label.startswith(SEASON_STR):
                continue

            # Season type filter (field names vary across mirrors)
            season_type = (
                g.get("seasonType")
                or g.get("seriesText")  # poor fallback; e.g., "Neutral Site" for Abu Dhabi
                or ""
            )
            # Keep if it matches one of expected labels (or keep all if label missing)
            if season_type and season_type not in KEEP_SEASON_TYPES and season_type not in {
                "Neutral Site", "Global Games"
            }:
                # Many global/neutral preseason games do not label "Preseason" directly;
                # allow them through by relaxing filter when unclear:
                pass

            home = team_name(g.get("homeTeam", {}))
            away = team_name(g.get("awayTeam", {}))

            date_iso_utc = to_dd_mm_yy_hh_mm(g)
            loc = location_str(g)

            out_rows.append(
                {
                    "Match Number": 0,           # fill later
                    "Round Number": 1,           # simple default; customize if you group rounds
                    "Date": date_iso_utc,
                    "Location": loc,
                    "Home Team": home,
                    "Away Team": away,
                    "Result": "",                # leave blank; populate post-game if needed
                }
            )

    # sort + enumerate match numbers (ascending by date)
    def sort_key(row):
        date_str = row["Date"]
        if not date_str:
            return datetime.datetime.max
        try:
            # Parse dd/mm/yy HH:MM format back to datetime for proper sorting
            return datetime.datetime.strptime(date_str, "%d/%m/%y %H:%M")
        except ValueError:
            return datetime.datetime.max
    
    out_rows.sort(key=sort_key)
    for i, r in enumerate(out_rows, 1):
        r["Match Number"] = i

    # write CSV in your exact schema
    fieldnames = [
        "Match Number",
        "Round Number",
        "Date",
        "Location",
        "Home Team",
        "Away Team",
        "Result",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {len(out_rows)} games to {OUT_CSV}")


if __name__ == "__main__":
    main()
