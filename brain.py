"""
Clearmap Brain — Real-time Oref alert poller + Telegram intel + Firebase sync.

Polls the Israeli Home Front Command API every 1.5s, manages a state machine
for each alerted city, reads Telegram intel from intel.db, and pushes the
current state to Firebase Realtime Database for the Next.js frontend.

State machine:
  pre_alert → alert → after_alert → (removed)
  
Timings:
  pre_alert:       12 min max, or until alert arrives
  alert:           1.5 min, then auto-transitions to after_alert
  after_alert:     persists until Oref clearance ("הסתיים"/"ניתן לצאת")
"""

import json
import logging
import math
import os
import re
import sys
import sqlite3
import time
import threading
from pathlib import Path

import firebase_admin
import requests
from firebase_admin import credentials, db

# ── Config ──────────────────────────────────────────────────────────────────

OREF_URL = "https://christina-nonreductional-supersilently.ngrok-free.dev/api/alerts"
OREF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.oref.org.il/12481-he/PikudHaoref.aspx",
    "X-Requested-With": "XMLHttpRequest",
}

FIREBASE_DB_URL = "https://clear-map-f20d0-default-rtdb.europe-west1.firebasedatabase.app/"
FIREBASE_NODE = "/public_state/active_alerts"

POLL_INTERVAL = 1.5        # seconds between API polls
REQUEST_TIMEOUT = 5        # HTTP timeout in seconds

# ── Timing constants ────────────────────────────────────────────────────────
PRE_ALERT_TTL = 720         # 12 minutes
ALERT_DURATION = 90         # 1.5 minutes → then after_alert
AFTER_ALERT_SAFETY_TTL = 1800   # 30 min safety net (cleared by Oref signal normally)

# ── Screenshot + Telegram bot config ────────────────────────────────────────

def _load_config_env() -> dict[str, str]:
    """Read key=value pairs from config.env (mounted into container)."""
    result = {}
    for p in (Path(__file__).parent / "config.env", Path("/app/config.env")):
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                result[key.strip()] = val.strip()
            break
    return result

_cfg_env = _load_config_env()
TELEGRAM_BOT_TOKEN = os.environ.get("CLEARMAP_BOT_TOKEN", "") or _cfg_env.get("CLEARMAP_BOT_TOKEN", "")
SCREENSHOT_COOLDOWN = int(os.environ.get("SCREENSHOT_COOLDOWN", "") or _cfg_env.get("SCREENSHOT_COOLDOWN", "120"))
SCREENSHOT_URL = os.environ.get("SCREENSHOT_URL", "") or _cfg_env.get("SCREENSHOT_URL", "https://clearmap.co.il")

POLYGONS_FILE = Path(os.environ.get("POLYGONS_FILE", Path(__file__).parent / "polygons.json"))
SERVICE_ACCOUNT_FILE = Path(os.environ.get("SERVICE_ACCOUNT_FILE", Path(__file__).parent / "serviceAccountKey.json"))

# Region name → list of district names for area mapping
REGION_MAPPING = {
    "צפון": ["מחוז צפון", "מחוז גליל עליון", "מחוז גליל תחתון", "מחוז גולן",
             "מחוז חיפה", "מחוז יערות הכרמל", "מחוז קו העימות", "מחוז העמקים",
             "מחוז בקעת בית שאן"],
    "דרום": ["מחוז דרום הנגב", "מחוז עוטף עזה", "מחוז נגב", "מחוז דרום השפלה",
             "מחוז אילת", "מחוז ים המלח"],
    "מרכז": ["מחוז דן", "מחוז השפלה", "מחוז ירקון", "מחוז שרון", "מחוז חפר",
             "מחוז שומרון"],
    "שפלה": ["מחוז השפלה", "מחוז דרום השפלה"],
    "נגב":  ["מחוז דרום הנגב", "מחוז נגב"],
    "ירושלים": ["מחוז ירושלים", "מחוז יהודה", "מחוז בית שמש", "מחוז בקעה"],
}

# Keyword → region for quick matching
REGION_KEYWORDS = {
    "צפון": "צפון", "גליל": "צפון", "גולן": "צפון", "חיפה": "צפון",
    "דרום": "דרום", "עוטף": "דרום", "נגב": "דרום", "אשקלון": "דרום",
    "מרכז": "מרכז", "דן": "מרכז", "שרון": "מרכז", "גוש דן": "מרכז",
    "שפלה": "שפלה",
    "ירושלים": "ירושלים", "יו\"ש": "ירושלים", "יהודה": "ירושלים", "שומרון": "מרכז",
}

# ── Import District Metadata ────────────────────────────────────────────────
try:
    from district_to_areas import DISTRICT_AREAS
except ImportError:
    DISTRICT_AREAS = {}

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brain")

# ── Types ───────────────────────────────────────────────────────────────────


class CityState:
    """Internal state for a single alerted city."""

    __slots__ = ("state", "started_at", "is_double", "city_name", "city_name_he")

    def __init__(self, city_name_he: str, city_name: str, started_at: float):
        self.state = "alert"
        self.started_at = started_at
        self.is_double = False
        self.city_name_he = city_name_he
        self.city_name = city_name

    def to_firebase(self) -> dict:
        """Serialize to the shape the frontend expects (ActiveAlert interface)."""
        return {
            "id": f"alert_{self.city_name_he}",
            "city_name": self.city_name,
            "city_name_he": self.city_name_he,
            "timestamp": int(self.started_at * 1000),  # JS milliseconds
            "is_double": self.is_double,
            "status": self.state,
        }


# ── Geo Helpers ────────────────────────────────────────────────────────────


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, in degrees [0, 360)."""
    dlon = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _project_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    """Project a point along a bearing by a distance. Returns (lat, lon)."""
    R = 6371.0
    d = dist_km / R
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1),
                              math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))


def _project_onto_line(
    point_lat: float, point_lng: float,
    origin_lat: float, origin_lng: float,
    bearing_deg: float,
) -> tuple[float, float, float]:
    """Project a point onto a line defined by origin + bearing.

    Returns (projected_lat, projected_lng, signed_distance_km).
    Positive distance = forward along bearing, negative = behind origin.
    """
    dist = _haversine_km(origin_lat, origin_lng, point_lat, point_lng)
    if dist < 0.01:  # essentially same point
        return (origin_lat, origin_lng, 0.0)
    brng_to_point = _bearing(origin_lat, origin_lng, point_lat, point_lng)
    angle_diff = math.radians(brng_to_point - bearing_deg)
    along = dist * math.cos(angle_diff)  # signed forward distance
    plat, plng = _project_point(origin_lat, origin_lng, bearing_deg, along)
    return (plat, plng, along)


def _perpendicular_dist_km(
    point_lat: float, point_lng: float,
    origin_lat: float, origin_lng: float,
    bearing_deg: float,
) -> float:
    """Perpendicular distance from a point to a line (origin + bearing) in km."""
    dist = _haversine_km(origin_lat, origin_lng, point_lat, point_lng)
    if dist < 0.01:
        return 0.0
    brng_to_point = _bearing(origin_lat, origin_lng, point_lat, point_lng)
    angle_diff = math.radians(brng_to_point - bearing_deg)
    return abs(dist * math.sin(angle_diff))


def _compute_centroid(polygon_coords: list) -> tuple[float, float]:
    """Average of all boundary points → (lat, lng)."""
    n = len(polygon_coords)
    if n == 0:
        return (0.0, 0.0)
    lat_sum = sum(p[0] for p in polygon_coords)
    lng_sum = sum(p[1] for p in polygon_coords)
    return (lat_sum / n, lng_sum / n)


# ── UAV Flight Path Tracker ───────────────────────────────────────────────

UAV_MIN_SPEED_KMH = 80
UAV_MAX_SPEED_KMH = 300
UAV_DEFAULT_SPEED_KMH = 180


class UavTrack:
    """A single tracked UAV flight path."""

    __slots__ = ("track_id", "raw_centroids", "smoothed_points", "cities",
                 "last_updated", "heading", "speed_estimate", "origin_type")

    def __init__(self, track_id: str, lat: float, lng: float, timestamp: float, city: str):
        self.track_id = track_id
        self.raw_centroids: list[tuple[float, float, float]] = [(lat, lng, timestamp)]
        self.smoothed_points: list[tuple[float, float, float]] = [(lat, lng, timestamp)]
        self.cities: set[str] = {city}
        self.last_updated = timestamp
        self.heading: float = 0.0
        self.speed_estimate: float = UAV_DEFAULT_SPEED_KMH
        self.origin_type: str = "לא ידוע"


class UavTracker:
    """Clusters UAV alerts into tracks and predicts smooth flight paths.

    Instead of using raw city centroids as the flight path (which causes
    teleporting/zigzag), this estimates a straight-line corridor through
    the alert areas and places observations along it.
    """

    CLUSTER_LATERAL_KM = 35.0   # max perpendicular distance from track line
    CLUSTER_FORWARD_KM = 80.0   # max forward distance from last smoothed point
    STALE_SECONDS = 300.0
    PREDICT_SECONDS = [30, 60]
    SPEED_SMOOTHING = 0.3       # exponential smoothing alpha for speed updates

    def __init__(self):
        self.tracks: dict[str, UavTrack] = {}
        self._next_id = 0

    def update(self, uav_cities: set[str], centroids: dict[str, tuple[float, float]], now: float) -> bool:
        """Process current UAV-alerted cities. Returns True if tracks changed.

        When multiple cities alert at once, their centroids are averaged into
        a single observation per track (the UAV is somewhere in that area,
        not at each city center individually).
        """
        changed = self._cleanup_stale(now)

        # Collect new (untracked) cities with valid centroids
        new_cities = []
        for city in uav_cities:
            if city not in centroids:
                continue
            if any(city in t.cities for t in self.tracks.values()):
                continue
            new_cities.append(city)

        if not new_cities:
            return changed

        # Assign each new city to a track (or mark for new track creation)
        # Group by track so simultaneous alerts merge into one observation
        track_batches: dict[str, list[str]] = {}   # track_id → [cities]
        orphans: list[str] = []                      # cities needing new tracks

        for city in new_cities:
            lat, lng = centroids[city]
            best_track = None
            best_score = float("inf")
            for track in self.tracks.values():
                dt_hours = (now - track.last_updated) / 3600.0
                # Maximum logical distance the UAV could have traveled since last update,
                # plus a 15km buffer for the alert polygon size / minor maneuvers.
                max_reachable_dist_km = (track.speed_estimate * dt_hours) + 15.0
                
                if len(track.smoothed_points) < 2:
                    dist = _haversine_km(lat, lng, track.smoothed_points[-1][0], track.smoothed_points[-1][1])
                    if dist < max_reachable_dist_km and dist < best_score:
                        best_score = dist
                        best_track = track
                else:
                    last_sp = track.smoothed_points[-1]
                    perp = _perpendicular_dist_km(lat, lng, last_sp[0], last_sp[1], track.heading)
                    _, _, along = _project_onto_line(lat, lng, last_sp[0], last_sp[1], track.heading)
                    if perp < self.CLUSTER_LATERAL_KM and along > -5 and along < max_reachable_dist_km:
                        score = perp + abs(along) * 0.1
                        if score < best_score:
                            best_score = score
                            best_track = track

            if best_track:
                track_batches.setdefault(best_track.track_id, []).append(city)
            else:
                orphans.append(city)

        # Add batched cities to existing tracks (average centroid per batch)
        for tid, cities in track_batches.items():
            track = self.tracks[tid]
            lats = [centroids[c][0] for c in cities]
            lngs = [centroids[c][1] for c in cities]
            avg_lat = sum(lats) / len(lats)
            avg_lng = sum(lngs) / len(lngs)
            self._add_to_track(track, avg_lat, avg_lng, now, cities)
            changed = True

        # Create new tracks for orphan cities (group nearby orphans together)
        while orphans:
            city = orphans.pop(0)
            lat, lng = centroids[city]
            tid = f"uav_{self._next_id}"
            self._next_id += 1
            # Check if remaining orphans are close enough to group with this one
            grouped = [city]
            remaining = []
            for other in orphans:
                olat, olng = centroids[other]
                if _haversine_km(lat, lng, olat, olng) < self.CLUSTER_LATERAL_KM:
                    grouped.append(other)
                else:
                    remaining.append(other)
            orphans = remaining

            # Average centroid of grouped cities
            lats = [centroids[c][0] for c in grouped]
            lngs = [centroids[c][1] for c in grouped]
            avg_lat = sum(lats) / len(lats)
            avg_lng = sum(lngs) / len(lngs)
            new_track = UavTrack(tid, avg_lat, avg_lng, now, grouped[0])
            for c in grouped[1:]:
                new_track.cities.add(c)
            self.tracks[tid] = new_track
            changed = True

        return changed

    def _add_to_track(self, track: UavTrack, raw_lat: float, raw_lng: float,
                      now: float, cities: list[str]) -> None:
        """Add a new observation to an existing track with smoothing."""
        track.raw_centroids.append((raw_lat, raw_lng, now))
        track.cities.update(cities)
        track.last_updated = now

        first = track.raw_centroids[0]
        UAV_LEAD_TIME_SECONDS = 90.0

        if len(track.raw_centroids) == 2:
            # Second observation — establish heading from first to new centroid
            track.heading = _bearing(first[0], first[1], raw_lat, raw_lng)

            if first[0] > 32.5:
                track.origin_type = "חיזבאללה (מלבנון)"
            elif first[0] < 31.0 or (first[0] < 32.0 and track.heading > 300 and track.heading < 60):
                track.origin_type = "החות'ים (מתימן)"
            elif first[1] > 35.0 or (track.heading > 200 and track.heading < 340):
                track.origin_type = "מיליציות (ממזרח / עיראק)"
            else:
                track.origin_type = "סיכול ממוקד / חמאס"

            # Initial speed estimate from distance and time
            track.speed_estimate = UAV_DEFAULT_SPEED_KMH

            # The current true position is based on the elapsed time from the first centroid
            _, _, proj_dist_from_first = _project_onto_line(raw_lat, raw_lng, first[0], first[1], track.heading)
            
            current_dist = max(proj_dist_from_first, 0.5)
            new_lat, new_lng = _project_point(first[0], first[1], track.heading, current_dist)
            track.smoothed_points.append((new_lat, new_lng, now))
        else:
            # 3+ observations — update heading (first→latest centroid for stability)
            track.heading = _bearing(first[0], first[1], raw_lat, raw_lng)

            # Advance along heading from last smoothed point by estimated travel
            last_sp = track.smoothed_points[-1]
            dt = now - last_sp[2]
            advance_km = (track.speed_estimate / 3600) * dt
            advance_km = max(advance_km, 0.5)  # minimum advance to avoid stacking

            new_lat, new_lng = _project_point(last_sp[0], last_sp[1], track.heading, advance_km)
            track.smoothed_points.append((new_lat, new_lng, now))

            # Update speed with exponential smoothing, but only if enough time has passed
            # since the very first observation (e.g. 15 seconds) to avoid initial jitter spikes.
            _, _, raw_proj_dist = _project_onto_line(
                raw_lat, raw_lng, first[0], first[1], track.heading
            )
            raw_proj_dist = max(raw_proj_dist, 0.0)
            total_dt = now - first[2]
            
            if total_dt > 15.0:
                raw_speed = (raw_proj_dist / total_dt) * 3600
                raw_speed = max(UAV_MIN_SPEED_KMH, min(UAV_MAX_SPEED_KMH, raw_speed))
                a = self.SPEED_SMOOTHING
                track.speed_estimate = a * raw_speed + (1 - a) * track.speed_estimate

    def _cleanup_stale(self, now: float) -> bool:
        stale = [tid for tid, t in self.tracks.items() if now - t.last_updated > self.STALE_SECONDS]
        for tid in stale:
            del self.tracks[tid]
        return len(stale) > 0

    def to_firebase(self) -> dict:
        """Serialize all tracks for the frontend."""
        result = {}
        for tid, track in self.tracks.items():
            observed = [[p[0], p[1]] for p in track.smoothed_points]
            predicted = []

            if len(track.smoothed_points) >= 2:
                last = track.smoothed_points[-1]
                for secs in self.PREDICT_SECONDS:
                    pred_dist = (track.speed_estimate / 3600) * secs
                    if pred_dist > 0:
                        plat, plng = _project_point(last[0], last[1], track.heading, pred_dist)
                        predicted.append([plat, plng])

            result[tid] = {
                "track_id": tid,
                "observed": observed,
                "predicted": predicted,
                "heading_deg": round(track.heading, 1),
                "speed_kmh": round(track.speed_estimate, 0),
                "origin_type": getattr(track, 'origin_type', "לא ידוע"),
                "last_updated": int(track.last_updated * 1000),
            }

        return result


FIREBASE_UAV_NODE = "/public_state/uav_tracks"


# ── Polygon Lookup ──────────────────────────────────────────────────────────


def load_polygons() -> dict:
    """Load the Hebrew-city-name → polygon lookup from polygons.json."""
    if not POLYGONS_FILE.exists():
        log.error("polygons.json not found! Run fetch_polygons.py first.")
        return {}

    with open(POLYGONS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    log.info("Loaded %d municipality polygons.", len(data))
    return data


# ── Firebase Init ───────────────────────────────────────────────────────────


def init_firebase():
    """Initialize Firebase Admin SDK."""
    import base64
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or _cfg_env.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        try:
            if sa_json.strip().startswith("{"):
                cert_dict = json.loads(sa_json)
            else:
                cert_dict = json.loads(base64.b64decode(sa_json).decode("utf-8"))
            cred = credentials.Certificate(cert_dict)
            log.info("Loaded service account from FIREBASE_SERVICE_ACCOUNT_JSON env var.")
        except Exception as e:
            msg = f"Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}. Please ensure it contains valid JSON or Base64."
            log.error(msg)
            raise RuntimeError(msg) from e
    elif SERVICE_ACCOUNT_FILE.exists():
        cred = credentials.Certificate(str(SERVICE_ACCOUNT_FILE))
        log.info("Loaded service account from local file: %s", SERVICE_ACCOUNT_FILE)
    else:
        msg = (
            f"Service account key not found! "
            f"You must either set the 'FIREBASE_SERVICE_ACCOUNT_JSON' environment variable, "
            f"or provide the file at '{SERVICE_ACCOUNT_FILE}'."
        )
        log.error(msg)
        raise RuntimeError(msg)
        
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    log.info("Firebase initialized → %s", FIREBASE_DB_URL)


# ── Oref Polling ────────────────────────────────────────────────────────────


def _classify_alert_object(alert_obj: dict) -> str:
    """Classify a single Oref alert object into a status string.

    Based on real API samples:
    - cat "1"  + "ירי רקטות וטילים"                     → "alert"
    - cat "6"  + "חדירת כלי טיס עוין"                   → "uav"
    - cat "1"  + "חדירת מחבלים"                          → "terrorist"
    - cat "10" + "בדקות הקרובות צפויות להתקבל התרעות"   → "pre_alert"
    - cat "10" + clearance signals                       → "clear"
    """
    cat = str(alert_obj.get("cat", ""))
    title = alert_obj.get("title", "")

    # Clearance signals → remove from map (check before other checks since
    # "חדירת כלי טיס עוין - האירוע הסתיים" contains both keywords)
    if "ניתן לצאת" in title or "להישאר בקרבת" in title or "הסתיים" in title or "החשש הוסר" in title:
        return "clear"

    # Title-based detection (works regardless of cat value)
    if "חדירת כלי טיס עוין" in title:
        return "uav"
    if "חדירת מחבלים" in title:
        return "terrorist"

    if cat in ("1", "6"):
        return "alert"

    if "בדקות הקרובות" in title or "שהייה בסמיכות" in title or "לשפר את המיקום" in title:
        return "pre_alert"

    # Default: treat unknown cat values as alert to be safe
    return "alert"


# Priority: higher number = takes precedence when merging
_STATUS_PRIORITY = {"after_alert": 1, "pre_alert": 2, "alert": 3, "uav": 3, "terrorist": 3}


def fetch_oref() -> list[tuple[str, str]]:
    """
    Fetch the current alert list from Oref.
    
    The API returns an array of alert objects, each with:
      - cat: category string ("1" for rockets, "10" for advisories)
      - title: descriptive Hebrew text
      - data: list of city name strings
    
    Returns a list of (city_name_he, status) tuples with priority merging.
    """
    url = os.environ.get("OREF_URL_OVERRIDE", OREF_URL)
    proxy_url = os.environ.get("OREF_PROXY_URL")
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    resp = requests.get(url, headers=OREF_HEADERS, timeout=REQUEST_TIMEOUT, proxies=proxies)
    
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            log.error("HTTP 403 Forbidden from Oref. If you are deployed outside Israel, Oref's geo-blocking firewall is dropping your requests. Set OREF_PROXY_URL to an Israeli proxy.")
        raise e

    text = resp.text.strip()
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.strip()
    if not text or text in ("null", "[]", "{}"):
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Oref returned non-JSON (first 200 chars): %s", text[:200])
        return []

    # Normalise: API can return a single object or an array of objects
    if isinstance(data, dict):
        alerts = [data]
    elif isinstance(data, list):
        # Could be a list of alert objects or (legacy) a flat list of city strings
        if data and isinstance(data[0], str):
            return [(city, "alert") for city in data]
        alerts = data
    else:
        return []

    # Merge cities from all alert objects with priority
    merged: dict[str, str] = {}

    for alert_obj in alerts:
        if not isinstance(alert_obj, dict):
            continue
        status = _classify_alert_object(alert_obj)
        cities = alert_obj.get("data", [])
        
        log.debug("Oref object: cat=%s title='%s' status=%s cities=%d",
                   alert_obj.get("cat"), alert_obj.get("title", "")[:40], status, len(cities))

        for city in cities:
            if not isinstance(city, str):
                continue
            existing = merged.get(city)
            if existing is None or _STATUS_PRIORITY.get(status, 0) > _STATUS_PRIORITY.get(existing, 0):
                merged[city] = status

    return list(merged.items())


# ── State Machine ───────────────────────────────────────────────────────────


def update_state(
    state: dict[str, CityState],
    oref_data: list[tuple[str, str]],
    polygons: dict,
) -> tuple[bool, bool]:
    """
    Updates the internal state machine.
    
    State machine rules:
    - pre_alert:       12 min timeout, or upgraded to alert
    - alert:           1.5 min duration, then auto → after_alert
    - after_alert:     persists until Oref "clear" signal. Re-alertable (→ alert)
    
    Priority: alert > pre_alert > after_alert
    
    Returns: (changed_flag, has_new_primary_alert_flag)
    """
    now = time.time()
    changed = False
    has_new_primary = False
    
    # ── Step 0: Time-based auto-transitions ──────────────────────────────
    for city_he, cs in list(state.items()):
        elapsed = now - cs.started_at
        
        if cs.state in ("alert", "uav", "terrorist") and elapsed >= ALERT_DURATION:
            # Alert expired → after_alert (shelter)
            cs.state = "after_alert"
            cs.started_at = now
            log.info("⏳ ALERT→AFTER_ALERT: %s (%.0fs elapsed)", city_he, elapsed)
            changed = True
        
        elif cs.state == "pre_alert" and elapsed >= PRE_ALERT_TTL:
            # Pre-alert expired without real alert → remove
            log.info("✅ PRE_ALERT EXPIRED: %s (%.0fs)", city_he, elapsed)
            del state[city_he]
            changed = True
        
        elif cs.state == "after_alert" and elapsed >= AFTER_ALERT_SAFETY_TTL:
            # 24h safety net — normally cleared by Oref "הסתיים" signal
            log.warning("⚠️ SAFETY CLEANUP: %s after_alert for 24h without clearance", city_he)
            del state[city_he]
            changed = True
    
    # ── Step 1: Merge all incoming signals with priority ──────────────────
    incoming: dict[str, str] = {}
    
    # Oref signals override telegram
    for city_he, status in oref_data:
        existing = incoming.get(city_he)
        if existing is None or _STATUS_PRIORITY.get(status, 0) > _STATUS_PRIORITY.get(existing, 0):
            incoming[city_he] = status
    
    oref_cities = {city for city, _ in oref_data}
    
    # ── Step 1b: Handle clearance signals (remove from map) ────────────
    for city_he, status in list(incoming.items()):
        if status == "clear":
            if city_he in state:
                log.info("✅ CLEARED by Oref: %s", city_he)
                del state[city_he]
                changed = True
            del incoming[city_he]

    # ── Step 2: Process incoming signals ──────────────────────────────────
    for city_he, alert_type in incoming.items():
        if city_he in state:
            cs = state[city_he]
            old_state = cs.state
            new_priority = _STATUS_PRIORITY.get(alert_type, 0)
            old_priority = _STATUS_PRIORITY.get(old_state, 0)
            
            # Only upgrade state (higher priority), never downgrade via incoming signal
            # Exception: after_alert can be re-promoted to alert
            if new_priority > old_priority or (old_state == "after_alert" and alert_type in ("alert", "uav", "terrorist")):
                log.info("🔄 %s → %s: %s", old_state, alert_type, city_he)
                cs.state = alert_type
                cs.started_at = now
                cs.is_double = (old_state in ("alert", "uav", "terrorist", "after_alert") and alert_type in ("alert", "uav", "terrorist"))
                changed = True
                if alert_type in ("alert", "uav", "terrorist", "pre_alert"):
                    has_new_primary = True
            elif old_state == alert_type and alert_type in ("alert", "uav", "terrorist", "pre_alert"):
                # Same state from Oref — refresh timer only for active oref states
                if city_he in oref_cities:
                    cs.started_at = now
        else:
            # New city
            poly_data = polygons.get(city_he)
            if not poly_data:
                if alert_type != "telegram_intel":
                    log.warning("No polygon data for '%s' — skipping.", city_he)
                continue
            
            cs = CityState(city_he, poly_data["city_name"], now)
            cs.state = alert_type
            state[city_he] = cs
            
            emoji = {"pre_alert": "🟠", "alert": "🔴", "uav": "🟣", "terrorist": "🔶", "after_alert": "⚫"}.get(alert_type, "❓")
            log.info("%s NEW %s: %s (%s)", emoji, alert_type.upper(), city_he, poly_data["city_name"])
            changed = True
            if alert_type in ("alert", "uav", "terrorist", "pre_alert"):
                has_new_primary = True
    
    # ── Step 3: Oref signals that STOPPED ─────────────────────────────────
    # If a city was in pre_alert/alert from Oref but is no longer in Oref response,
    # let the timer handle the transition (Step 0 on next tick).
    # We do NOT force-transition here — the Oref API might just be between polls.
    
    return changed, has_new_primary


# ── Firebase Sync ───────────────────────────────────────────────────────────


def _sanitize_fb_key(key: str) -> str:
    """Sanitize a string for use as a Firebase RTDB key.

    Firebase keys cannot contain: . $ # [ ] /
    Replace them with underscores.
    """
    return re.sub(r'[.$/\[\]#]', '_', key)


def sync_to_firebase(state: dict[str, CityState]):
    """Push the current state to Firebase Realtime Database."""
    ref = db.reference(FIREBASE_NODE)

    if not state:
        ref.set({})
        log.info("Firebase synced: no active alerts.")
        return

    payload = {_sanitize_fb_key(city_he): cs.to_firebase() for city_he, cs in state.items()}
    ref.set(payload)
    log.info("Firebase synced: %d alerts.", len(payload))


def sync_uav_tracks(tracker: UavTracker):
    """Push UAV flight path tracks to Firebase."""
    uav_ref = db.reference(FIREBASE_UAV_NODE)
    payload = tracker.to_firebase()
    uav_ref.set(payload if payload else {})
    if payload:
        log.info("Firebase UAV tracks synced: %d tracks.", len(payload))


# ── Screenshot + Telegram Bot Broadcast ─────────────────────────────────────

_STATUS_EMOJI = {
    "alert": "🔴", "uav": "🟣", "terrorist": "🔶",
    "pre_alert": "🟠", "after_alert": "⚫", "telegram_intel": "🔵",
}
_STATUS_LABEL = {
    "alert": "התרעות ירי רקטות וטילים", "uav": "התרעות חדירת כלי טיס עוין",
    "terrorist": "חדירת מחבלים", "pre_alert": "התרעות מוקדמות",
    "after_alert": "להישאר בממ\"ד", "telegram_intel": "מודיעין",
}

_screenshot_lock = threading.Lock()

def _bot_send_message(chat_id: str, text: str):
    """Send a text message via Telegram Bot API."""
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            log.warning("Bot sendMessage failed for %s: %s", chat_id, resp.text[:150])
    except Exception as e:
        log.error("Bot sendMessage error for %s: %s", chat_id, e)


def _handle_bot_command(chat_id: str, text: str, first_name: str):
    """Handle an incoming bot command."""
    cmd = text.strip().lower().split()[0] if text else ""

    if cmd in ("/start", "/subscribe"):
        _bot_send_message(
            chat_id,
            "✅ הרישום עבר לערוץ המרכזי: @clearmapchannel"
        )
    elif cmd == "/unsubscribe":
        _bot_send_message(chat_id, "❌ כדי להפסיק לקבל עדכונים, יש לעזוב את הערוץ @clearmapchannel")
    elif cmd == "/status":
        _bot_send_message(chat_id, "🤖 הבוט פעיל ומשדר לערוץ המרכזי.")
    else:
        _bot_send_message(
            chat_id,
            "🤖 <b>ClearMap Bot</b>\n\n"
            "העדכונים עברו לערוץ: @clearmapchannel"
        )


def _bot_poller():
    """Background thread: poll Telegram getUpdates for bot commands."""
    log.info("🤖 Bot poller started (token=%s...)", TELEGRAM_BOT_TOKEN[:12])
    offset = 0

    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 30, "allowed_updates": '["message"]'},
                timeout=35,
            )
            if not resp.ok:
                log.warning("🤖 getUpdates HTTP %d: %s", resp.status_code, resp.text[:150])
                time.sleep(5)
                continue

            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))
                first_name = msg.get("from", {}).get("first_name", "?")

                if chat_id and text.startswith("/"):
                    log.debug("🤖 Command from %s (%s): %s", first_name, chat_id, text)
                    _handle_bot_command(chat_id, text, first_name)

        except requests.exceptions.Timeout:
            continue  # long-poll timeout is normal
        except requests.exceptions.ConnectionError:
            log.warning("🤖 Bot poller: network unreachable — retrying in 30s")
            time.sleep(30)
        except Exception as e:
            log.error("🤖 Bot poller error: %s", e)
            time.sleep(5)


def _build_caption(state: dict) -> str:
    """Build a Hebrew caption with alert locations, times, and website link."""
    from collections import defaultdict
    from datetime import datetime
    try:
        from district_to_areas import DISTRICT_AREAS
    except ImportError:
        DISTRICT_AREAS = {}

    if not state:
        return "אין התרעות פעילות"

    # Build reverse mapping from city -> region
    city_to_region = {}
    for region, cities_list in DISTRICT_AREAS.items():
        for c in cities_list:
            city_to_region[c] = region

    # Group cities by status
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for city_he, cs in state.items():
        groups[cs.state].append((cs.city_name_he, cs.started_at))

    lines = []
    # Priority order for display
    for status in ("alert", "uav", "terrorist", "pre_alert", "after_alert"):
        cities = groups.get(status)
        if not cities:
            continue
        emoji = _STATUS_EMOJI.get(status, "❓")
        label = _STATUS_LABEL.get(status, status)
        
        if status == "after_alert":
            lines.append(f"{emoji} {len(cities)} מקומות להישאר במרחב מוגן:")
        else:
            lines.append(f"{emoji} {len(cities)} {label}:")

        # If too many alerts of this type, group by region
        if len(cities) > 8 and DISTRICT_AREAS:
            reg_dict: dict[str, float] = {}
            for city_name, ts in cities:
                reg = city_to_region.get(city_name, city_name)
                # Keep the newest timestamp for the region
                if reg not in reg_dict or ts > reg_dict[reg]:
                    reg_dict[reg] = ts
            
            reg_list = sorted(reg_dict.items(), key=lambda x: x[1], reverse=True)
            for reg, ts in reg_list:
                t = datetime.fromtimestamp(ts).strftime("%H:%M")
                lines.append(f"  • {reg} ({t})")
        else:
            # Sort by time (newest first) and list city names with time
            cities.sort(key=lambda x: x[1], reverse=True)
            for city_name, ts in cities:
                t = datetime.fromtimestamp(ts).strftime("%H:%M")
                lines.append(f"  • {city_name} ({t})")

    lines.append("")
    lines.append("🗺 clearmap.co.il")

    caption = "\n".join(lines)

    # Telegram photo captions are limited to 1024 characters
    if len(caption) > 1024:
        # Truncate lists but keep the summary + link
        summary_parts = []
        for status in ("alert", "uav", "terrorist", "pre_alert", "after_alert"):
            cities = groups.get(status)
            if not cities:
                continue
            emoji = _STATUS_EMOJI.get(status, "❓")
            label = _STATUS_LABEL.get(status, status)
            if status == "after_alert":
                summary_parts.append(f"{emoji} {len(cities)} מקומות להישאר במרחב מוגן")
            else:
                summary_parts.append(f"{emoji} {len(cities)} {label}")
        caption = " | ".join(summary_parts) + "\n\n🗺 clearmap.co.il"

    return caption


def _send_photo_to_chat(chat_id: str, photo_path: Path, caption: str) -> bool:
    """Send a photo to a single chat. Returns True on success."""
    try:
        with open(photo_path, "rb") as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": ("alert.png", f, "image/png")},
                timeout=30,
            )
        if resp.ok:
            return True
        else:
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            # Auto-remove subscribers who blocked the bot
            if body.get("error_code") in (403, 400):
                log.warning("📸 Removing blocked/invalid subscriber %s: %s",
                            chat_id, body.get("description", ""))
                with _subscribers_lock:
                    _subscribers.discard(chat_id)
                    _save_subscribers(_subscribers)
            else:
                log.warning("📸 sendPhoto failed for %s: %s", chat_id, resp.text[:150])
            return False
    except Exception as e:
        log.error("📸 sendPhoto error for %s: %s", chat_id, e)
        return False


TARGET_CHANNEL_ID = "-1003879479829"

def capture_and_broadcast(state: dict):
    """Capture a screenshot and broadcast it to the main channel.

    Runs in a background thread — must not raise.
    """
    import subprocess
    
    if not _screenshot_lock.acquire(blocking=False):
        log.info("📸 Screenshot already in progress — skipping.")
        return

    try:
        log.info("📸 Capturing screenshot for broadcast channel via subprocess...")

        caption = _build_caption(state)
        log.info("📸 Caption: %s", caption)

        output_dir = Path(__file__).parent / "screenshots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_path = output_dir / "broadcast_latest.png"
        script_path = Path(__file__).parent / "screenshot_alerts.py"
        
        url = SCREENSHOT_URL
        if "?" in url:
            url += "&screenshot=true"
        else:
            url += "?screenshot=true"
            
        cmd = [
            sys.executable, str(script_path),
            "--url", url,
            "--output-file", str(final_path),
            "--theme", "dark",
            "--size", "1080"
        ]
        
        def _lower_priority():
            if sys.platform != "win32":
                try:
                    os.nice(15)
                except Exception:
                    pass

        t0 = time.time()
        log.info("📸 Running: %s", " ".join(cmd))
        
        # Run playwright screenshot job in an isolated, low-priority process
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            preexec_fn=_lower_priority if sys.platform != "win32" else None,
            timeout=120
        )
        
        if result.returncode != 0:
            log.error("📸 Screenshot subprocess failed (code=%d):\nSTDOUT: %s\nSTDERR: %s", 
                      result.returncode, result.stdout, result.stderr)
            return
            
        if not final_path.exists():
            log.error("📸 Screenshot subprocess completed but output file missing!")
            return

        log.info("📸 Final screenshot ready (%.1fs, %.1fKB) — broadcasting to channel...",
                 time.time() - t0, final_path.stat().st_size / 1024)

        # Broadcast to channel
        if _send_photo_to_chat(TARGET_CHANNEL_ID, final_path, caption):
             log.info("📸 Broadcast sent to channel successfully")
        else:
             log.error("📸 Broadcast to channel failed")

    except Exception as e:
        log.error("📸 Screenshot/broadcast error: %s", e, exc_info=True)
    finally:
        _screenshot_lock.release()


# ── Main Loop ───────────────────────────────────────────────────────────────


def main():
    log.info("=== Clearmap Brain starting ===")

    polygons = load_polygons()
    if not polygons:
        log.error("Cannot start without polygon data. Exiting.")
        return

    init_firebase()

    state: dict[str, CityState] = {}
    centroids = {name: _compute_centroid(p["polygon"]) for name, p in polygons.items() if p.get("polygon")}
    uav_tracker = UavTracker()
    last_screenshot_time = 0.0
    sliding_window_end_time = 0.0
    pending_screenshot = False  # True when a screenshot was deferred due to cooldown

    # Start bot command poller in background
    if TELEGRAM_BOT_TOKEN:
        threading.Thread(target=_bot_poller, daemon=True).start()
        log.info("📸 Screenshot broadcast enabled (cooldown=%ds)", SCREENSHOT_COOLDOWN)
    else:
        log.info("📸 Screenshot broadcast disabled (set CLEARMAP_BOT_TOKEN)")

    # Clear any stale data on startup
    sync_to_firebase(state)
    sync_uav_tracks(uav_tracker)

    log.info("Polling Oref every %.1fs | alert=%ds pre_alert=%ds after_alert=until_clear",
             POLL_INTERVAL, ALERT_DURATION, PRE_ALERT_TTL)

    while True:
        try:
            oref_data = fetch_oref()
            changed, has_new_primary = update_state(state, oref_data, polygons)

            uav_cities = {c for c, cs in state.items() if cs.state == "uav"}
            tracks_changed = uav_tracker.update(uav_cities, centroids, time.time())

            if changed or tracks_changed:
                sync_to_firebase(state)
                sync_uav_tracks(uav_tracker)

            # Slide the window if we got new actual alerts
            now_ts = time.time()
            if has_new_primary and TELEGRAM_BOT_TOKEN:
                # Delay the screenshot by 10s. If it arrives within 10s of another alert, it extends the batch clock!
                sliding_window_end_time = now_ts + 10.0
                pending_screenshot = True
                log.info("📸 New primary alert received! Sliding window set/pushed to 10s from now.")

            # Send pending screenshot once both the sliding window and global cooldown have elapsed
            if pending_screenshot and TELEGRAM_BOT_TOKEN:
                if now_ts >= sliding_window_end_time:
                    time_since_last = now_ts - last_screenshot_time
                    if time_since_last >= SCREENSHOT_COOLDOWN:
                        if state:
                            log.info("📸 Cooldown & Sliding window elapsed! Triggering screenshot for %d alerts", len(state))
                            last_screenshot_time = now_ts
                            pending_screenshot = False
                            state_snapshot = {k: v for k, v in state.items()}
                            threading.Thread(
                                target=capture_and_broadcast,
                                args=(state_snapshot,),
                                daemon=True,
                            ).start()
                        else:
                            log.info("📸 Window elapsed but alerts cleared — cancelling pending screenshot")
                            pending_screenshot = False
                    else:
                        remaining_cool = SCREENSHOT_COOLDOWN - time_since_last
                        if int(remaining_cool) % 30 == 0 or remaining_cool > SCREENSHOT_COOLDOWN - 2:
                            log.info("📸 Screenshot ready from batching, but waiting on global cooldown: %.0fs left", remaining_cool)
                else:
                    remaining_window = sliding_window_end_time - now_ts
                    if int(remaining_window) % 3 == 0 or remaining_window > 8:
                        log.info("📸 Waiting for alert barrage to settle... (%.1fs sliding window left)", remaining_window)

        except requests.exceptions.RequestException as e:
            log.error("HTTP error: %s", e)
        except json.JSONDecodeError as e:
            log.error("JSON parse error: %s", e)
        except Exception as e:
            log.error("Unexpected error: %s", e, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

