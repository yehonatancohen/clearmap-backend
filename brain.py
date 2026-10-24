"""
Clearmap Brain — Real-time Oref alert poller + Firebase sync.

Polls the Israeli Home Front Command API every 1.5s, manages a state machine
for each alerted city, and pushes the
current state to Firebase Realtime Database for the Next.js frontend.

State machine:
  pre_alert → alert → after_alert → (removed)
  
Timings:
  pre_alert:       12 min max, or until alert arrives
  alert:           1.5 min, then auto-transitions to after_alert
  after_alert:     persists until Oref clearance ("הסתיים"/"ניתן לצאת")
"""

import inspect
import json
import logging
import math
import os
import queue
import random
import re
import socket
import sys
import sqlite3
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import firebase_admin
import requests
from firebase_admin import credentials, db

try:
    from pywebpush import webpush, WebPushException
    HAS_WEBPUSH = True
    # Newer pywebpush accepts a requests session, which lets each worker reuse
    # its TLS connection to the push service instead of handshaking per message.
    _WEBPUSH_HAS_SESSION = "requests_session" in inspect.signature(webpush).parameters
except ImportError:
    HAS_WEBPUSH = False
    _WEBPUSH_HAS_SESSION = False

# ── Config ──────────────────────────────────────────────────────────────────

OREF_URL = "https://backend.clearmap.co.il/api/alerts"
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

POLL_INTERVAL = 0.5        # seconds between API polls
REQUEST_TIMEOUT = 5        # HTTP timeout in seconds

# Global session to reuse TCP/TLS connections (Keep-Alive) for faster polling
http_session = requests.Session()


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

# ── Web Push (VAPID) config ──────────────────────────────────────────────────
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "") or _cfg_env.get("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "") or _cfg_env.get("VAPID_PUBLIC_KEY", "")
VAPID_SUBJECT = os.environ.get("VAPID_SUBJECT", "") or _cfg_env.get("VAPID_SUBJECT", "mailto:yoncohenyon@gmail.com")
FIREBASE_PUSH_SUBS_NODE = "/push_subscriptions"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, "") or _cfg_env.get(key, default)


# ── Scale / resilience config ───────────────────────────────────────────────

# Extra RTDB instances (same project) that mirror /public_state. Each instance
# holds ~200k concurrent clients, so N mirrors ≈ N×200k viewers. The frontend
# picks one at random from NEXT_PUBLIC_FIREBASE_DATABASE_URLS.
FIREBASE_MIRROR_URLS = [u.strip() for u in _env("FIREBASE_MIRROR_URLS").split(",") if u.strip()]

# Tried in order when the primary alert source fails (e.g. a second Israeli bridge).
OREF_URL_FALLBACKS = [u.strip() for u in _env("OREF_URL_FALLBACKS").split(",") if u.strip()]

FIREBASE_HEARTBEAT_NODE = "/public_state/heartbeat"
FIREBASE_CLEARS_NODE = "/public_state/clears"
FIREBASE_HISTORY_NODE = "/public_state/history"
FIREBASE_LEADER_NODE = "/system/leader"

HEARTBEAT_INTERVAL = 5          # seconds between heartbeat writes
OREF_STALE_SECONDS = 15         # heartbeat reports oref_ok=False after this long without a good fetch
CLEARS_TTL = 1800               # keep official "event ended" records for 30 min
HISTORY_MAX_EVENTS = 500

# Leader election lets a standby brain run alongside the primary: both poll,
# only the lease holder writes to Firebase and sends pushes.
LEADER_ELECTION = _env("LEADER_ELECTION", "1") != "0"
LEADER_LEASE_TTL = 12           # seconds; a standby takes over after this much silence
LEADER_RENEW_INTERVAL = 3
INSTANCE_ID = f"{socket.gethostname()}-{os.getpid()}-{random.randrange(16**4):04x}"

PUSH_WORKERS = int(_env("PUSH_WORKERS", "48"))
PUSH_ALERT_MAX_AGE = 120        # drop undelivered alert pushes older than this (seconds)
PUSH_CLEAR_MAX_AGE = 900

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

# ── City → Region mapping (for pre-alert area grouping) ───────────────────
_CITY_REGIONS_FILE = Path(__file__).parent / "city-regions.json"
try:
    CITY_REGIONS: dict[str, str] = json.loads(_CITY_REGIONS_FILE.read_text(encoding="utf-8"))
except Exception:
    CITY_REGIONS = {}

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
    env_keys = ["FIREBASE_SERVICE_ACCOUNT_JSON", "FIREBASE_SERVICE_ACCOUNT", "SERVICE_ACCOUNT_JSON"]
    sa_json = None
    for key in env_keys:
        val = os.environ.get(key) or _cfg_env.get(key)
        if val:
            sa_json = val
            break

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
        
    # Bounded HTTP timeout: the default (120s) could freeze the polling loop.
    # Streaming listeners don't use it.
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL, "httpTimeout": 10})
    log.info("Firebase initialized → %s", FIREBASE_DB_URL)


# ── Oref Polling ────────────────────────────────────────────────────────────


def _classify_alert_object(alert_obj: dict) -> str:
    """Classify a single Oref alert object into a status string.

    Based on real API samples:
    - cat "1"  + "ירי רקטות וטילים"                     → "alert"
    - cat "6"  + "חדירת כלי טיס עוין"                   → "uav"
    - category 2 + "כלי טייס עוין - היכנסו מייד למרחב המוגן" → "uav" (Oct 2026 wording)
    - cat "1"  + "חדירת מחבלים"                          → "terrorist"
    - cat "10" + "בדקות הקרובות צפויות להתקבל התרעות"   → "pre_alert"
    - cat "10" + clearance signals                       → "clear"
    """
    # Support both old API format (cat: str) and new format (category: int)
    cat = str(alert_obj.get("cat", alert_obj.get("category", "")))
    title = alert_obj.get("title", "")

    # Clearance signals → remove from map (check before other checks since
    # "חדירת כלי טיס עוין - האירוע הסתיים" contains both keywords).
    # Terror clears read "הוסר החשש לחדירת מחבלים" (Oct 2026), older ones "החשש הוסר".
    if ("ניתן לצאת" in title or "להישאר בקרבת" in title or "הסתיים" in title
            or "החשש הוסר" in title or "הוסר החשש" in title):
        return "clear"

    # Advisory phrasing wins over threat keywords, so an early warning that
    # mentions the threat type isn't shown as a live alert.
    if "בדקות הקרובות" in title or "שהייה בסמיכות" in title or "לשפר את המיקום" in title or "התקרבו למרחב" in title:
        return "pre_alert"

    # Title-based detection (works regardless of cat value).
    # Oref spells it both "כלי טיס" and "כלי טייס", with or without "חדירת".
    if "כלי טיס" in title or "כלי טייס" in title:
        return "uav"
    if "מחבלים" in title:
        return "terrorist"

    # Category fallback for UAV when the title wording is unrecognised:
    # "cat" carries the matrix id (6), "category" carries the history id (2).
    if cat == "6" or ("cat" not in alert_obj and cat == "2"):
        return "uav"

    if cat == "1":
        return "alert"

    # Category 14 = "התקרבו למרחב מוגן" advisory
    if cat == "14":
        return "pre_alert"

    # Unrecognised wording: "category" 13 is the history id for "event ended".
    # Last, so a threat title never loses to it (matrix id 13 is terror).
    if "cat" not in alert_obj and cat == "13":
        return "clear"

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
    urls = [os.environ.get("OREF_URL_OVERRIDE", OREF_URL), *OREF_URL_FALLBACKS]
    last_error: Exception | None = None
    for i, url in enumerate(urls):
        try:
            data = _fetch_oref_json(url)
            break
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            last_error = e
            if i < len(urls) - 1:
                log.warning("Alert source %s failed (%s) — trying fallback", url, e)
    else:
        raise last_error  # type: ignore[misc]

    return _parse_oref_payload(data)


def _fetch_oref_json(url: str):
    proxy_url = os.environ.get("OREF_PROXY_URL")
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    max_retries = 2
    for attempt in range(max_retries):
        resp = http_session.get(url, headers=OREF_HEADERS, timeout=REQUEST_TIMEOUT, proxies=proxies)
        if resp.status_code in (502, 503, 504) and attempt < max_retries - 1:
            log.warning("Oref fetch got %d, retrying in 1s... (attempt %d/%d)",
                        resp.status_code, attempt + 1, max_retries)
            time.sleep(1)
            continue
        break

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            log.error("HTTP 403 Forbidden from Oref (or bridge proxy). Set OREF_PROXY_URL to an Israeli proxy if hitting Oref directly.")
        raise e

    # Oref returns an empty body (sometimes just a BOM) when there are no alerts
    text = resp.content.decode("utf-8-sig", errors="replace").strip()
    if not text:
        return []
    return json.loads(text)


def _parse_oref_payload(data) -> list[tuple[str, str]]:
    """Normalise an Oref (or bridge) response into (city_name_he, status) tuples."""

    if not data:
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
        raw_data = alert_obj.get("data", [])
        # New API returns data as a single city string; old API returns a list
        cities = [raw_data] if isinstance(raw_data, str) else raw_data

        log.debug("Oref object: cat=%s title='%s' status=%s cities=%d",
                   alert_obj.get("cat", alert_obj.get("category")), alert_obj.get("title", "")[:40], status, len(cities))

        for city in cities:
            if not isinstance(city, str):
                continue
            existing = merged.get(city)
            if existing is None or _STATUS_PRIORITY.get(status, 0) > _STATUS_PRIORITY.get(existing, 0):
                merged[city] = status

    return list(merged.items())


# ── State Machine ───────────────────────────────────────────────────────────


class StateUpdate:
    """What changed in one tick of the state machine. Side effects (Firebase,
    pushes, history) are applied by the caller, and only by the leader."""

    __slots__ = ("changed", "new_primary_cities", "cleared_cities", "history")

    def __init__(self):
        self.changed = False
        self.new_primary_cities: list[str] = []
        self.cleared_cities: list[str] = []
        self.history: dict[str, list[str]] = {}   # status → cities, one history event each

    def log(self, status: str, cities: list[str]):
        self.history.setdefault(status, []).extend(cities)


def update_state(
    state: dict[str, CityState],
    oref_data: list[tuple[str, str]],
    polygons: dict,
) -> StateUpdate:
    """
    Updates the internal state machine.

    State machine rules:
    - pre_alert:       12 min timeout, or upgraded to alert
    - alert:           1.5 min duration, then auto → after_alert
    - after_alert:     persists until Oref "clear" signal. Re-alertable (→ alert)

    Priority: alert > pre_alert > after_alert
    """
    now = time.time()
    result = StateUpdate()
    changed = False
    new_primary_cities = result.new_primary_cities

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
    cleared_cities = []
    for city_he, status in list(incoming.items()):
        if status == "clear":
            if city_he in state:
                log.info("✅ CLEARED by Oref: %s", city_he)
                cleared_cities.append(city_he)
                del state[city_he]
                changed = True
            del incoming[city_he]

    if cleared_cities:
        result.cleared_cities = cleared_cities
        result.log("clear", cleared_cities)

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
                    new_primary_cities.append(city_he)
                    result.log(alert_type, [city_he])
            elif old_state == alert_type and alert_type in ("alert", "uav", "terrorist", "pre_alert"):
                # Same state from Oref — refresh timer only for active oref states
                if city_he in oref_cities:
                    cs.started_at = now
        else:
            # New city
            poly_data = polygons.get(city_he)
            if not poly_data:
                log.warning("No polygon data for '%s' — skipping.", city_he)
                continue

            cs = CityState(city_he, poly_data["city_name"], now)
            cs.state = alert_type
            state[city_he] = cs

            emoji = {"pre_alert": "🟠", "alert": "🔴", "uav": "🟣", "terrorist": "🔶", "after_alert": "⚫"}.get(alert_type, "❓")
            log.info("%s NEW %s: %s (%s)", emoji, alert_type.upper(), city_he, poly_data["city_name"])
            changed = True
            if alert_type in ("alert", "uav", "terrorist", "pre_alert"):
                new_primary_cities.append(city_he)
                result.log(alert_type, [city_he])

    # ── Step 3: Oref signals that STOPPED ─────────────────────────────────
    # If a city was in pre_alert/alert from Oref but is no longer in Oref response,
    # let the timer handle the transition (Step 0 on next tick).
    # We do NOT force-transition here — the Oref API might just be between polls.

    result.changed = changed
    return result


# ── Firebase Sync ───────────────────────────────────────────────────────────


def _sanitize_fb_key(key: str) -> str:
    """Sanitize a string for use as a Firebase RTDB key.

    Firebase keys cannot contain: . $ # [ ] /
    Replace them with underscores.
    """
    return re.sub(r'[.$/\[\]#]', '_', key)


def _live_write(path: str, value: dict, method: str = "set") -> bool:
    """Write a /public_state node to the primary RTDB, then hand the result to
    the mirrors. Only the primary is awaited: a slow or dead mirror must never
    stall the polling loop."""
    try:
        ref = db.reference(path)
        if method == "update":
            ref.update(value)
        else:
            ref.set(value)
        return True
    except Exception as e:
        log.error("Firebase write to %s failed: %s", path, e)
        return False


class MirrorTarget:
    """Keeps one mirror RTDB instance in step with the primary on its own thread.

    Holds the latest full value per path and its own baseline of what the
    mirror has, so it ships deltas, and only this mirror re-sends in full
    (with backoff) after a failure.
    """

    MAX_BACKOFF = 30

    def __init__(self, url: str):
        self.url = url
        self._desired: dict[str, tuple] = {}     # path → (value, always_set)
        self._synced: dict[str, dict | None] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        threading.Thread(target=self._run, name=f"mirror-{url[:30]}", daemon=True).start()

    def reset(self):
        """Forget the baseline (another instance may have written meanwhile)."""
        with self._lock:
            self._synced = {}
        self._wake.set()

    def publish(self, path: str, value: dict, always_set: bool = False):
        with self._lock:
            self._desired[path] = (value, always_set)
        self._wake.set()

    def _run(self):
        backoff = 1
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                items = list(self._desired.items())
            failed = False
            for path, (value, always_set) in items:
                base = self._synced.get(path)
                try:
                    ref = db.reference(path, url=self.url)
                    if base is None or always_set:
                        ref.set(value)
                    else:
                        diff: dict = {k: v for k, v in value.items() if base.get(k) != v}
                        diff.update({k: None for k in base if k not in value})
                        if diff:
                            ref.update(diff)
                    self._synced[path] = None if always_set else value
                    if always_set:
                        with self._lock:
                            if self._desired.get(path, (None,))[0] is value:
                                del self._desired[path]
                except Exception as e:
                    failed = True
                    self._synced[path] = None
                    log.warning("Mirror %s write to %s failed: %s", self.url, path, e)
            if failed:
                time.sleep(backoff)
                backoff = min(backoff * 2, self.MAX_BACKOFF)
                self._wake.set()
            else:
                backoff = 1


_mirrors: list[MirrorTarget] = []


def _publish_mirrors(path: str, value: dict, always_set: bool = False):
    for m in _mirrors:
        m.publish(path, value, always_set)


class FirebaseSync:
    """Pushes live state as deltas instead of rewriting whole nodes.

    A ref.set() of active_alerts re-sends every city to every connected viewer;
    during a 500-city salvo that is ~75KB × every viewer × every change. A
    multi-path update only ships the cities that actually changed.
    """

    def __init__(self):
        self._alerts: dict | None = None     # last payload every instance accepted
        self._clears: dict[str, dict] = {}   # key → {city_name_he, ts, source}
        self._clears_synced: dict | None = None
        self.needs_resync = True

    def reset(self):
        """Forget what was synced; the next sync rewrites every node in full."""
        self._alerts = None
        self._clears_synced = None
        self.needs_resync = True

    def sync(self, state: dict[str, CityState], tracker: "UavTracker"):
        ok = self._sync_alerts(state)
        tracks = tracker.to_firebase() or {}
        ok = _live_write(FIREBASE_UAV_NODE, tracks) and ok
        _publish_mirrors(FIREBASE_UAV_NODE, tracks)
        ok = self._sync_clears(state) and ok
        self.needs_resync = not ok

    def _sync_alerts(self, state: dict[str, CityState]) -> bool:
        payload = {_sanitize_fb_key(city_he): cs.to_firebase() for city_he, cs in state.items()}
        _publish_mirrors(FIREBASE_NODE, payload)
        if self._alerts is None:
            ok = _live_write(FIREBASE_NODE, payload)
        else:
            diff: dict = {k: v for k, v in payload.items() if self._alerts.get(k) != v}
            diff.update({k: None for k in self._alerts if k not in payload})
            if not diff:
                return True
            ok = _live_write(FIREBASE_NODE, diff, "update")
        self._alerts = payload if ok else None
        if ok:
            log.info("Firebase synced: %d alerts.", len(payload))
        return ok

    def record_clears(self, cities: list[str]):
        """Remember an official Oref "event ended" for each city."""
        now_ms = int(time.time() * 1000)
        for city_he in cities:
            self._clears[_sanitize_fb_key(city_he)] = {
                "city_name_he": city_he,
                "ts": now_ms,
                "source": "oref",
            }

    def _sync_clears(self, state: dict[str, CityState]) -> bool:
        # A city that is alerted again (or expired long ago) no longer has a
        # valid "you can leave" message.
        cutoff = (time.time() - CLEARS_TTL) * 1000
        active = {_sanitize_fb_key(c) for c in state}
        self._clears = {k: v for k, v in self._clears.items() if v["ts"] >= cutoff and k not in active}
        _publish_mirrors(FIREBASE_CLEARS_NODE, dict(self._clears))

        if self._clears_synced is None:
            ok = _live_write(FIREBASE_CLEARS_NODE, dict(self._clears))
        else:
            diff: dict = {k: v for k, v in self._clears.items() if self._clears_synced.get(k) != v}
            diff.update({k: None for k in self._clears_synced if k not in self._clears})
            if not diff:
                return True
            ok = _live_write(FIREBASE_CLEARS_NODE, diff, "update")
        self._clears_synced = dict(self._clears) if ok else None
        return ok

    def has_expiring_clears(self) -> bool:
        cutoff = (time.time() - CLEARS_TTL) * 1000
        return any(v["ts"] < cutoff for v in self._clears.values())


# ── History log ─────────────────────────────────────────────────────────────

_PUSH_CHARS = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"


class HistoryWriter:
    """Batches history events and writes them off the main loop.

    The old code did one blocking push() per city before the map was synced,
    so a 300-city salvo meant ~300 round-trips to Europe before anyone saw it.
    """

    FLUSH_INTERVAL = 1.0
    CLEANUP_INTERVAL = 300

    def __init__(self):
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._last_ms = 0
        self._last_rand: list[int] = []

    def start(self):
        threading.Thread(target=self._run, name="history", daemon=True).start()

    def add(self, events: dict[str, list[str]]):
        with self._lock:
            for status, cities in events.items():
                if cities:
                    self._pending.append({
                        "status": status,
                        "cities": list(dict.fromkeys(cities)),
                        "timestamp": {".sv": "timestamp"},
                    })

    def _push_id(self) -> str:
        """Chronologically sortable key, same scheme as RTDB push()."""
        now = int(time.time() * 1000)
        if now == self._last_ms:
            for i in range(11, -1, -1):
                if self._last_rand[i] != 63:
                    self._last_rand[i] += 1
                    break
                self._last_rand[i] = 0
        else:
            self._last_ms = now
            self._last_rand = [random.randrange(64) for _ in range(12)]
        ts = ""
        for _ in range(8):
            ts = _PUSH_CHARS[now % 64] + ts
            now //= 64
        return ts + "".join(_PUSH_CHARS[r] for r in self._last_rand)

    def _run(self):
        last_cleanup = 0.0
        while True:
            time.sleep(self.FLUSH_INTERVAL)
            with self._lock:
                batch, self._pending = self._pending, []
            if batch:
                try:
                    db.reference(FIREBASE_HISTORY_NODE).update({self._push_id(): e for e in batch})
                except Exception as e:
                    log.error("Failed to log history (%d events): %s", len(batch), e)
            if time.time() - last_cleanup > self.CLEANUP_INTERVAL:
                last_cleanup = time.time()
                self._cleanup()

    def _cleanup(self):
        try:
            keys = sorted((db.reference(FIREBASE_HISTORY_NODE).get(shallow=True) or {}).keys())
            excess = keys[:-HISTORY_MAX_EVENTS] if len(keys) > HISTORY_MAX_EVENTS else []
            if excess:
                db.reference(FIREBASE_HISTORY_NODE).update({k: None for k in excess})
                log.info("History cleanup: removed %d old events", len(excess))
        except Exception as e:
            log.warning("History cleanup failed: %s", e)


# ── Leader election + heartbeat ─────────────────────────────────────────────


class _LeaseHeld(Exception):
    pass


class LeaderElection:
    """Lease in /system/leader so a standby brain can run next to the primary.

    Every instance polls Oref and keeps its state machine warm; only the lease
    holder writes to Firebase and sends pushes. If the leader stops renewing
    for LEADER_LEASE_TTL seconds, a standby takes over.
    """

    def __init__(self):
        self._valid_until = float("inf") if not LEADER_ELECTION else 0.0
        self._was_leader = False

    def start(self):
        if LEADER_ELECTION:
            threading.Thread(target=self._run, name="leader", daemon=True).start()

    def is_leader(self) -> bool:
        return time.time() < self._valid_until

    def became_leader(self) -> bool:
        """True exactly once each time this instance gains the lease."""
        now_leader = self.is_leader()
        promoted = now_leader and not self._was_leader
        self._was_leader = now_leader
        return promoted

    def _run(self):
        while True:
            started = time.time()
            now_ms = int(started * 1000)

            def claim(current):
                if (isinstance(current, dict) and current.get("id") != INSTANCE_ID
                        and now_ms - int(current.get("ts", 0)) < LEADER_LEASE_TTL * 1000):
                    raise _LeaseHeld()
                return {"id": INSTANCE_ID, "ts": now_ms}

            try:
                db.reference(FIREBASE_LEADER_NODE).transaction(claim)
                if not self.is_leader():
                    log.info("👑 Acquired leader lease (%s)", INSTANCE_ID)
                # Stop acting as leader a few seconds before a standby may take over
                self._valid_until = started + LEADER_LEASE_TTL - 4
            except _LeaseHeld:
                if self.is_leader():
                    log.warning("👑 Lost leader lease — switching to standby")
                self._valid_until = 0.0
            except Exception as e:
                # Can't reach Firebase: keep the current lease until it runs out
                log.warning("👑 Lease renewal failed: %s", e)
            time.sleep(LEADER_RENEW_INTERVAL)


class Heartbeat:
    """Lets the frontend tell "no alerts" apart from "the poller is down"."""

    def __init__(self, election: LeaderElection):
        self._election = election
        self.last_oref_ok = 0.0

    def start(self):
        threading.Thread(target=self._run, name="heartbeat", daemon=True).start()

    def _run(self):
        while True:
            if self._election.is_leader():
                beat = {
                    "ts": {".sv": "timestamp"},
                    "oref_ok": time.time() - self.last_oref_ok < OREF_STALE_SECONDS,
                }
                _live_write(FIREBASE_HEARTBEAT_NODE, beat)
                _publish_mirrors(FIREBASE_HEARTBEAT_NODE, dict(beat), always_set=True)
            time.sleep(HEARTBEAT_INTERVAL)


# ── Web Push Notifications ──────────────────────────────────────────────────

def _normalize_coords(raw) -> list | None:
    """Normalize userCoords from Firebase.
    Firebase serializes JS arrays as {"0": lat, "1": lng} (string keys),
    so we must handle both list and dict formats.
    """
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return [float(raw[0]), float(raw[1])]
    if isinstance(raw, dict) and len(raw) == 2:
        try:
            keys = sorted(raw.keys(), key=lambda k: int(k))
            return [float(raw[keys[0]]), float(raw[keys[1]])]
        except (ValueError, TypeError):
            return None
    return None


def _as_list(raw) -> list:
    """Firebase returns arrays as lists, or as {"0": ..} dicts when sparse."""
    if isinstance(raw, list):
        return [v for v in raw if v is not None]
    if isinstance(raw, dict):
        return list(raw.values())
    return []


_STATUS_LABELS = {
    "alert": "צבע אדום!",
    "uav": "חדירת כלי טיס עוין",
    "terrorist": "חדירת מחבלים",
    "pre_alert": "התרעה מוקדמת (מודיעין)",
    "after_alert": "להישאר במרחב המוגן",
}


def _region_body(cities: list[str]) -> str:
    """"Region: City1, City2 +3 · Region2: ..." for a notification body."""
    by_region: dict[str, list[str]] = {}
    for c in cities:
        by_region.setdefault(CITY_REGIONS.get(c, "אחר"), []).append(c)
    parts = []
    for region, r_cities in by_region.items():
        cities_str = ", ".join(r_cities[:8])
        if len(r_cities) > 8:
            cities_str += f" +{len(r_cities)-8}"
        parts.append(f"{region}: {cities_str}")
    return " · ".join(parts)


def _match_subscriber(sub_info: dict, cities: list[str], city_set: set[str], centroids: dict) -> tuple[list[str], list[str]]:
    """Which of `cities` this subscriber wants, and the names of their people hit.

    Returns (matched_cities, people_names). People ("my people") always match,
    even when the subscriber filters by city or location. Runs once per
    subscriber per job, so the common all-Israel case avoids per-city work.
    """
    settings = sub_info.get("settings") or {}
    all_israel = settings.get("allIsrael", True)

    people_by_city: dict[str, list[str]] = {}
    for p in _as_list(settings.get("people")):
        if isinstance(p, dict) and p.get("city") in city_set and p.get("name"):
            people_by_city.setdefault(p["city"], []).append(str(p["name"])[:24])
    people_hit: list[str] = []
    for names in people_by_city.values():
        people_hit.extend(n for n in names if n not in people_hit)

    if all_israel:
        return cities, people_hit

    selected = set(_as_list(settings.get("selectedCities"))) & city_set
    user_coords = _normalize_coords(sub_info.get("userCoords"))
    if not selected and not people_by_city and not user_coords:
        return [], []

    matched: list[str] = []
    for city_he in cities:
        if city_he in selected or city_he in people_by_city:
            matched.append(city_he)
        elif user_coords:
            city_pos = centroids.get(city_he)
            if city_pos and _haversine_km(user_coords[0], user_coords[1], city_pos[0], city_pos[1]) <= 15.0:
                matched.append(city_he)
    return matched, people_hit


def _with_people(title: str, people: list[str]) -> str:
    if not people:
        return title
    shown = ", ".join(people[:3]) + (f" +{len(people)-3}" if len(people) > 3 else "")
    return f"{title} · {shown}"


class PushDispatcher:
    """Fans Web Push notifications out to every subscriber in parallel.

    - Subscriptions are mirrored in memory via a Firebase stream instead of
      downloading the whole node for every alert.
    - Work is never dropped because a previous batch is still sending (the old
      semaphore silently skipped whole salvos and clearance messages).
    - Newest alerts go first; undelivered pushes past their max age are skipped.
    """

    RELISTEN_INTERVAL = 1800

    def __init__(self):
        self._subs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._work: queue.PriorityQueue = queue.PriorityQueue()
        self._jobs: queue.Queue = queue.Queue()
        self._seq = 0
        self._local = threading.local()

    @property
    def enabled(self) -> bool:
        return HAS_WEBPUSH and bool(VAPID_PRIVATE_KEY)

    def start(self):
        if not self.enabled:
            return
        threading.Thread(target=self._subs_loop, name="push-subs", daemon=True).start()
        threading.Thread(target=self._planner, name="push-plan", daemon=True).start()
        for i in range(PUSH_WORKERS):
            threading.Thread(target=self._worker, name=f"push-{i}", daemon=True).start()

    # ── subscription mirror ──

    def _subs_loop(self):
        while True:
            registration = None
            try:
                registration = db.reference(FIREBASE_PUSH_SUBS_NODE).listen(self._on_event)
            except Exception as e:
                log.error("📱 Push subscription stream failed: %s", e)
                time.sleep(30)
                continue
            # Re-open periodically in case the stream died silently
            time.sleep(self.RELISTEN_INTERVAL)
            try:
                registration.close()
            except Exception:
                pass

    def _on_event(self, event):
        path = [p for p in (event.path or "/").split("/") if p]
        data = event.data
        is_patch = event.event_type == "patch"
        with self._lock:
            if not path:
                if is_patch:
                    for k, v in (data or {}).items():
                        self._set(self._subs, [k], v)
                else:
                    self._subs = dict(data) if isinstance(data, dict) else {}
                    log.info("📱 Push subscriptions loaded: %d", len(self._subs))
            elif is_patch:
                for k, v in (data or {}).items():
                    self._set(self._subs, path + [k], v)
            else:
                self._set(self._subs, path, data)

    @staticmethod
    def _set(root: dict, path: list[str], value):
        node = root
        for key in path[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                if value is None:
                    return
                child = node[key] = {}
            node = child
        if value is None:
            node.pop(path[-1], None)
        else:
            node[path[-1]] = value

    # ── job submission ──

    def notify_alerts(self, state: dict[str, CityState], new_cities: list[str], centroids: dict):
        if not self.enabled:
            return
        by_status: dict[str, list[str]] = {}
        for city_he in new_cities:
            cs = state.get(city_he)
            if cs:
                by_status.setdefault(cs.state, []).append(city_he)
        if not by_status:
            return
        log.info("📱 Push: queueing %d cities: %s", len(new_cities), ", ".join(new_cities[:5]))
        stamp = int(time.time())
        city_sets = {status: set(cities) for status, cities in by_status.items()}

        def build(sub_info: dict) -> list[dict]:
            settings = sub_info.get("settings") or {}
            payloads = []
            for status, cities in by_status.items():
                if status == "pre_alert" and not settings.get("earlyAlerts", True):
                    continue
                matched, people = _match_subscriber(sub_info, cities, city_sets[status], centroids)
                if not matched:
                    continue
                payloads.append({
                    "title": _with_people(_STATUS_LABELS.get(status, "התרעה"), people),
                    "body": _region_body(matched),
                    "tag": f"push_{status}_{stamp}",
                    "url": "https://clearmap.co.il",
                })
            return payloads

        self._enqueue(build, ttl=300, max_age=PUSH_ALERT_MAX_AGE, label="alert")

    def notify_clearance(self, cities: list[str], centroids: dict):
        if not self.enabled or not cities:
            return
        log.info("📱 Push (Clearance): queueing %d cities", len(cities))
        stamp = int(time.time())
        city_set = set(cities)

        def build(sub_info: dict) -> list[dict]:
            settings = sub_info.get("settings") or {}
            if not settings.get("leaveShelterAlerts", True):
                return []
            matched, people = _match_subscriber(sub_info, cities, city_set, centroids)
            if not matched:
                return []
            return [{
                "title": _with_people("האירוע הסתיים", people),
                "body": f"ניתן לצאת מהמרחב המוגן:\n{_region_body(matched)}",
                "tag": f"clearance_{stamp}",
                "url": "https://clearmap.co.il",
            }]

        self._enqueue(build, ttl=1800, max_age=PUSH_CLEAR_MAX_AGE, label="clearance")

    def _enqueue(self, build, ttl: int, max_age: int, label: str):
        # Matching every subscriber is too slow for the polling loop; a planner
        # thread does it and feeds the delivery workers.
        self._jobs.put((build, ttl, max_age, label, time.time()))

    def _planner(self):
        while True:
            build, ttl, max_age, label, created = self._jobs.get()
            try:
                self._plan(build, ttl, created + max_age, label)
            except Exception as e:
                log.error("📱 Push planning failed (%s): %s", label, e, exc_info=True)

    def _plan(self, build, ttl: int, deadline: float, label: str):
        with self._lock:
            subs = list(self._subs.items())
            self._seq += 1
            seq = self._seq
        queued = 0
        for sub_key, sub_info in subs:
            if not isinstance(sub_info, dict) or "endpoint" not in sub_info:
                continue
            if not (sub_info.get("settings") or {}).get("enabled", True):
                continue
            payloads = build(sub_info)
            if not payloads:
                continue
            # Newest job first (-seq); ties broken by insertion order
            self._work.put((-seq, queued, (sub_key, sub_info, payloads, ttl, deadline)))
            queued += 1
        log.info("📱 Push %s job #%d: %d subscribers queued (%d total)", label, seq, queued, len(subs))

    # ── delivery ──

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
        return s

    def _worker(self):
        while True:
            _, _, (sub_key, sub_info, payloads, ttl, deadline) = self._work.get()
            if time.time() > deadline:
                continue
            subscription_info = {"endpoint": sub_info["endpoint"], "keys": sub_info.get("keys", {})}
            extra = {"requests_session": self._session()} if _WEBPUSH_HAS_SESSION else {}
            for payload in payloads:
                try:
                    webpush(
                        subscription_info=subscription_info,
                        data=json.dumps(payload),
                        vapid_private_key=VAPID_PRIVATE_KEY,
                        vapid_claims={"sub": VAPID_SUBJECT},
                        timeout=10,
                        ttl=ttl,
                        headers={"Urgency": "high"},
                        **extra,
                    )
                except WebPushException as e:
                    if e.response is not None and e.response.status_code in (404, 410):
                        self._expire(sub_key)
                        break
                    log.debug("📱 Push failed for %s: %s", sub_key, e)
                except requests.exceptions.RequestException as e:
                    log.warning("📱 Push network error for %s: %s", sub_key, e)
                    break
                except Exception as e:
                    log.warning("📱 Push error for %s: %s", sub_key, e)
                    break

    def _expire(self, sub_key: str):
        with self._lock:
            self._subs.pop(sub_key, None)
        try:
            db.reference(f"{FIREBASE_PUSH_SUBS_NODE}/{sub_key}").delete()
        except Exception:
            pass


# ── Telegram Bot ────────────────────────────────────────────────────────────

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



# ── Main Loop ───────────────────────────────────────────────────────────────


def restore_state(state: dict[str, CityState], polygons: dict, fb_sync: FirebaseSync):
    """Adopt what is currently published, so a restart or failover doesn't wipe
    cities that are still in after_alert (Oref no longer lists those)."""
    try:
        published = db.reference(FIREBASE_NODE).get() or {}
        clears = db.reference(FIREBASE_CLEARS_NODE).get() or {}
    except Exception as e:
        log.warning("Could not restore published state: %s", e)
        return
    adopted = 0
    for entry in published.values() if isinstance(published, dict) else []:
        if not isinstance(entry, dict) or entry.get("is_test"):
            continue
        city_he = entry.get("city_name_he")
        if not city_he or city_he in state or city_he not in polygons:
            continue
        cs = CityState(city_he, polygons[city_he]["city_name"], float(entry.get("timestamp", 0)) / 1000 or time.time())
        cs.state = entry.get("status", "alert")
        cs.is_double = bool(entry.get("is_double"))
        state[city_he] = cs
        adopted += 1
    for entry in clears.values() if isinstance(clears, dict) else []:
        if isinstance(entry, dict) and entry.get("city_name_he") and entry.get("ts"):
            fb_sync._clears.setdefault(_sanitize_fb_key(entry["city_name_he"]), entry)
    if adopted:
        log.info("Restored %d published alerts from Firebase", adopted)


def main():
    log.info("=== Clearmap Brain starting (%s) ===", INSTANCE_ID)

    polygons = load_polygons()
    if not polygons:
        log.error("Cannot start without polygon data. Exiting.")
        return

    init_firebase()

    state: dict[str, CityState] = {}
    centroids = {name: _compute_centroid(p["polygon"]) for name, p in polygons.items() if p.get("polygon")}
    uav_tracker = UavTracker()

    fb_sync = FirebaseSync()
    history = HistoryWriter()
    election = LeaderElection()
    heartbeat = Heartbeat(election)
    pushes = PushDispatcher()

    restore_state(state, polygons, fb_sync)
    for component in (history, election, heartbeat, pushes):
        component.start()

    if pushes.enabled:
        log.info("📱 Web Push ENABLED (%d workers, session reuse=%s)", PUSH_WORKERS, _WEBPUSH_HAS_SESSION)
    else:
        log.warning("📱 Web Push DISABLED — HAS_WEBPUSH=%s, VAPID_PRIVATE_KEY=%s",
                     HAS_WEBPUSH, "SET" if VAPID_PRIVATE_KEY else "EMPTY")
    if FIREBASE_MIRROR_URLS:
        _mirrors.extend(MirrorTarget(url) for url in FIREBASE_MIRROR_URLS)
        log.info("Mirroring live state to %d extra RTDB instance(s)", len(FIREBASE_MIRROR_URLS))
    log.info("Leader election: %s", "ON" if LEADER_ELECTION else "OFF (always leader)")

    log.info("Polling Oref every %.1fs | alert=%ds pre_alert=%ds after_alert=until_clear",
             POLL_INTERVAL, ALERT_DURATION, PRE_ALERT_TTL)

    while True:
        try:
            oref_data = fetch_oref()
            heartbeat.last_oref_ok = time.time()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            log.error("Alert source error: %s", e)
            oref_data = []  # still run the time-based transitions below

        try:
            upd = update_state(state, oref_data, polygons)

            uav_cities = {c for c, cs in state.items() if cs.state == "uav"}
            tracks_changed = uav_tracker.update(uav_cities, centroids, time.time())

            if election.became_leader():
                restore_state(state, polygons, fb_sync)
                fb_sync.reset()
                for m in _mirrors:
                    m.reset()

            if election.is_leader():
                # Map first: everything else is slower and less urgent
                if upd.cleared_cities:
                    fb_sync.record_clears(upd.cleared_cities)
                if (upd.changed or tracks_changed or upd.cleared_cities
                        or fb_sync.needs_resync or fb_sync.has_expiring_clears()):
                    fb_sync.sync(state, uav_tracker)

                # Send push notifications only for genuinely new/upgraded cities
                if upd.new_primary_cities:
                    pushes.notify_alerts(state, upd.new_primary_cities, centroids)
                if upd.cleared_cities:
                    pushes.notify_clearance(upd.cleared_cities, centroids)
                history.add(upd.history)

        except Exception as e:
            log.error("Unexpected error: %s", e, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
