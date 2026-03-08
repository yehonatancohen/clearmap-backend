#!/usr/bin/env python3
"""
Interactive test tool — write mock alerts to Firebase RTDB
and inject mock Telegram messages into intel.db.

Usage:
    python test_alerts.py
"""

import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, db

# Import district mapping if available
try:
    from district_to_areas import DISTRICT_AREAS
except ImportError:
    DISTRICT_AREAS = {}

FIREBASE_DB_URL = "https://clear-map-f20d0-default-rtdb.europe-west1.firebasedatabase.app/"
FIREBASE_NODE = "/public_state/active_alerts"
FIREBASE_UAV_NODE = "/public_state/uav_tracks"
SERVICE_ACCOUNT = Path(__file__).parent / "serviceAccountKey.json"
POLYGONS_FILE = Path(__file__).parent / "polygons.json"

STATUSES = ["alert", "pre_alert", "after_alert", "uav", "terrorist"]

# ── Screenshot config (loaded from config.env) ──────────────────────────────
def _load_screenshot_config() -> str:
    """Read bot token from config.env."""
    for p in (Path(__file__).parent / "config.env",
              Path(__file__).parent.parent / "config.env"):
        if p.exists():
            cfg = {}
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                cfg[key.strip()] = val.strip()
            return cfg.get("CLEARMAP_BOT_TOKEN", "")
    return ""

_BOT_TOKEN = _load_screenshot_config()

SUBSCRIBERS_FILE = Path(__file__).parent / "subscribers.json"

# When running from test_alerts.py, screenshots go only to the developer
_DEV_CHAT_ID = "985770181"
_LOCALHOST_URL = "http://localhost:3000"
_PROD_URL = "https://clearmap.co.il"


def _load_subscribers() -> list[str]:
    """Load subscriber chat IDs from disk."""
    if SUBSCRIBERS_FILE.exists():
        try:
            return json.loads(SUBSCRIBERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _auto_send_screenshot():
    """Capture screenshot from localhost and send only to developer."""
    if not _BOT_TOKEN:
        return
    try:
        from screenshot_alerts import quick_capture_and_send
        print(f"\n[📸] Sending test screenshot to dev ({_DEV_CHAT_ID})...")
        quick_capture_and_send(
            _BOT_TOKEN, _DEV_CHAT_ID,
            caption="🧪 Test alert screenshot",
            url=_LOCALHOST_URL,
        )
    except Exception as e:
        print(f"  [warn] Screenshot send failed: {e}")


REGION_MAPPING = {
    "צפון": ["מחוז צפון", "מחוז גליל עליון", "מחוז גליל תחתון", "מחוז גולן",
             "מחוז חיפה", "מחוז יערות הכרמל", "מחוז קו העימות", "מחוז העמקים",
             "מחוז בקעת בית שאן", "מחוז ואדי ערה", "מחוז קצרין", "מחוז תבור", "מחוז קריות"],
    "דרום": ["מחוז דרום הנגב", "מחוז עוטף עזה", "מחוז נגב", "מחוז דרום השפלה",
             "מחוז אילת", "מחוז ים המלח", "מחוז לכיש", "מחוז מערב הנגב", "מחוז מרכז הנגב", "מחוז מערב לכיש"],
    "מרכז": ["מחוז דן", "מחוז השפלה", "מחוז ירקון", "מחוז שרון", "מחוז חפר",
             "מחוז שומרון", "מחוז מנשה"],
    "ירושלים": ["מחוז ירושלים", "מחוז יהודה", "מחוז בית שמש", "מחוז בקעה"],
}

UAV_FLIGHT_PRESETS = {
    "Lebanon → Haifa (Direct)": [
        ["ראש הנקרה", "לימן"],
        ["נהריה", "געתון"],
        ["עכו", "שמרת"],
        ["קריות", "עכו - אזור תעשייה"],
        ["חיפה - מפרץ", "נשר"],
        ["חיפה - כרמל, הדר ועיר תחתית"],
    ],
    "Lebanon → Central Galilee → Tiberias": [
        ["זרעית", "שתולה"],
        ["פסוטה", "אבירים"],
        ["מעלות תרשיחא", "מעונה"],
        ["כרמיאל", "בענה"],
        ["מע'אר", "עילבון"],
        ["טבריה", "מגדל"],
    ],
    "Gaza → Tel Aviv (Coast)": [
        ["נתיב העשרה", "זיקים"],
        ["אשקלון - צפון", "ניצנים"],
        ["אשדוד - א,ב,ד,ה", "גן יבנה"],
        ["יבנה", "פלמחים"],
        ["ראשון לציון - מערב", "חולון"],
        ["תל אביב - דרום העיר ויפו", "בת ים"],
    ],
    "Yemen → Eilat → Dead Sea": [
        ["אילת"],
        ["אילות"],
        ["צופר", "פארן"],
        ["עין יהב", "חצבה"],
        ["נאות הכיכר", "עין תמר"],
        ["עין גדי", "בתי מלון ים המלח"],
    ],
}

SALVO_PRESETS = {
    "North Salvo (Galilee)": [
        "מחוז גליל עליון", "מחוז גליל תחתון", "מחוז קו העימות"
    ],
    "Center Salvo (Gush Dan)": [
        "מחוז דן", "מחוז השפלה", "מחוז שרון"
    ],
    "South Salvo (Otef Gaza + Negev)": [
        "מחוז עוטף עזה", "מחוז נגב", "מחוז דרום הנגב"
    ],
}

def _sanitize_fb_key(key: str) -> str:
    return re.sub(r'[.$/\[\]#]', '_', key)


def _make_payload(city_he: str, city_name: str, status: str, is_double: bool = False, ts: float = None) -> dict:
    return {
        "id": f"alert_{city_he}",
        "city_name": city_name,
        "city_name_he": city_he,
        "timestamp": int((ts or time.time()) * 1000),
        "is_double": is_double,
        "status": status,
        "is_test": True,
    }


def _send_batch(cities: list[str], polygons: dict, status: str = None):
    if not status:
        print("Status:")
        for i, s in enumerate(STATUSES):
            print(f"  {i}: {s}")
        try:
            status = STATUSES[int(input("Pick status: ").strip())]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

    batch = {}
    for city_he in cities:
        if city_he not in polygons:
            continue
        key = _sanitize_fb_key(city_he)
        batch[key] = _make_payload(city_he, polygons[city_he]["city_name"], status)

    ref = db.reference(FIREBASE_NODE)
    ref.update(batch)
    print(f"Written {len(batch)} alerts with status={status}")
    _auto_send_screenshot()


def cmd_add_alert(polygons: dict):
    query = input("Search city (Hebrew): ").strip()
    if not query:
        return
    matches = [k for k in polygons if query in k][:20]
    if not matches:
        print("No matches found.")
        return
    for i, m in enumerate(matches):
        print(f"  {i}: {m}")
    try:
        idx = int(input("Pick index: ").strip())
        city_he = matches[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    print("Status:")
    for i, s in enumerate(STATUSES):
        print(f"  {i}: {s}")
    try:
        status = STATUSES[int(input("Pick status: ").strip())]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    is_double = input("Double/salvo? (y/N): ").strip().lower() == "y"

    payload = _make_payload(city_he, polygons[city_he]["city_name"], status, is_double)
    ref = db.reference(f"{FIREBASE_NODE}/{_sanitize_fb_key(city_he)}")
    ref.set(payload)
    print(f"Written: {city_he} = {status}")
    _auto_send_screenshot()


def cmd_batch_alert(polygons: dict):
    query = input("Search cities containing (Hebrew): ").strip()
    if not query:
        return
    matches = [k for k in polygons if query in k]
    print(f"Found {len(matches)} cities matching '{query}'")
    if not matches:
        return

    limit = input(f"How many to alert? (max {len(matches)}, Enter=all): ").strip()
    limit = int(limit) if limit else len(matches)
    matches = matches[:limit]

    _send_batch(matches, polygons)


def cmd_district_alert(polygons: dict):
    if not DISTRICT_AREAS:
        print("DISTRICT_AREAS not loaded.")
        return
    
    districts = sorted(list(DISTRICT_AREAS.keys()))
    for i, d in enumerate(districts):
        print(f"  {i}: {d}")
    
    try:
        idx = int(input("Pick district: ").strip())
        district = districts[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    cities = DISTRICT_AREAS[district]
    print(f"Alerting {len(cities)} cities in {district}")
    _send_batch(cities, polygons)


def cmd_region_alert(polygons: dict):
    regions = sorted(list(REGION_MAPPING.keys()))
    for i, r in enumerate(regions):
        print(f"  {i}: {r}")
    
    try:
        idx = int(input("Pick region: ").strip())
        region = regions[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    districts = REGION_MAPPING[region]
    cities = []
    for d in districts:
        cities.extend(DISTRICT_AREAS.get(d, []))
    
    print(f"Alerting {len(cities)} cities in region {region}")
    _send_batch(cities, polygons)


def cmd_clear_all():
    db.reference(FIREBASE_NODE).set({})
    db.reference(FIREBASE_UAV_NODE).set({})
    print("All alerts and UAV tracks cleared.")


def cmd_clear_test_alerts():
    # Clear active alerts
    alerts_ref = db.reference(FIREBASE_NODE)
    alerts = alerts_ref.get()
    if alerts:
        test_keys = {k: None for k, v in alerts.items() if v.get("is_test")}
        if test_keys:
            alerts_ref.update(test_keys)
            print(f"Cleared {len(test_keys)} test alerts.")
        else:
            print("No test alerts found.")
    else:
        print("No active alerts.")

    # Clear UAV tracks
    uav_ref = db.reference(FIREBASE_UAV_NODE)
    tracks = uav_ref.get()
    if tracks:
        test_keys = {k: None for k, v in tracks.items() if v.get("is_test")}
        if test_keys:
            uav_ref.update(test_keys)
            print(f"Cleared {len(test_keys)} test UAV tracks.")
        else:
            print("No test UAV tracks found.")
    else:
        print("No active UAV tracks.")


def cmd_clear_one(polygons: dict):
    query = input("Search city to clear (Hebrew): ").strip()
    if not query:
        return
    matches = [k for k in polygons if query in k][:20]
    if not matches:
        print("No matches found.")
        return
    for i, m in enumerate(matches):
        print(f"  {i}: {m}")
    try:
        idx = int(input("Pick index: ").strip())
        city_he = matches[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    ref = db.reference(f"{FIREBASE_NODE}/{_sanitize_fb_key(city_he)}")
    ref.delete()
    print(f"Cleared: {city_he}")


def cmd_show_active():
    data = db.reference(FIREBASE_NODE).get()
    if not data:
        print("No active alerts.")
        return
    print(f"\n{'City':<25} {'Status':<18} {'Double'}")
    print("-" * 55)
    for key, val in data.items():
        print(f"{val.get('city_name_he', key):<25} {val.get('status', '?'):<18} {val.get('is_double', False)}")
    print(f"\nTotal: {len(data)}")





def _centroid(polygon_coords: list) -> tuple[float, float]:
    """Average of boundary points → (lat, lng)."""
    n = len(polygon_coords)
    if n == 0:
        return (0.0, 0.0)
    return (sum(p[0] for p in polygon_coords) / n, sum(p[1] for p in polygon_coords) / n)


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlon = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _project_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    R = 6371.0
    d = dist_km / R
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1),
                              math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))


def _project_onto_line(point_lat, point_lng, origin_lat, origin_lng, bearing_deg):
    """Project a point onto a line defined by origin + bearing."""
    dist = _haversine_km(origin_lat, origin_lng, point_lat, point_lng)
    if dist < 0.01:
        return (origin_lat, origin_lng, 0.0)
    brng_to_point = _bearing(origin_lat, origin_lng, point_lat, point_lng)
    angle_diff = math.radians(brng_to_point - bearing_deg)
    along = dist * math.cos(angle_diff)
    plat, plng = _project_point(origin_lat, origin_lng, bearing_deg, along)
    return (plat, plng, along)


UAV_MIN_SPEED_KMH = 80
UAV_MAX_SPEED_KMH = 300
UAV_DEFAULT_SPEED_KMH = 180
SPEED_SMOOTHING = 0.3


class _TestUavTrack:
    """Mirrors brain.py UavTrack for test simulation."""
    def __init__(self, track_id, lat, lng, ts):
        self.track_id = track_id
        self.raw_centroids = [(lat, lng, ts)]
        self.smoothed_points = [(lat, lng, ts)]
        self.heading = 0.0
        self.speed_estimate = UAV_DEFAULT_SPEED_KMH

    def add_point(self, raw_lat, raw_lng, now):
        self.raw_centroids.append((raw_lat, raw_lng, now))
        first = self.raw_centroids[0]
        UAV_LEAD_TIME_SECONDS = 90.0

        if len(self.raw_centroids) == 2:
            self.heading = _bearing(first[0], first[1], raw_lat, raw_lng)

            offset_km = (UAV_DEFAULT_SPEED_KMH / 3600) * UAV_LEAD_TIME_SECONDS
            back_h = (self.heading + 180) % 360
            fixed_first_lat, fixed_first_lng = _project_point(first[0], first[1], back_h, offset_km)
            self.smoothed_points[0] = (fixed_first_lat, fixed_first_lng, first[2])

            _, _, proj_dist = _project_onto_line(raw_lat, raw_lng, first[0], first[1], self.heading)
            _, _, proj_dist_from_fixed = _project_onto_line(raw_lat, raw_lng, fixed_first_lat, fixed_first_lng, self.heading)
            
            current_dist = max(proj_dist_from_fixed - offset_km, 0.5)
            new_lat, new_lng = _project_point(fixed_first_lat, fixed_first_lng, self.heading, current_dist)
            self.smoothed_points.append((new_lat, new_lng, now))

            dt = now - first[2]
            if dt > 0:
                self.speed_estimate = max(UAV_MIN_SPEED_KMH, min(UAV_MAX_SPEED_KMH, (proj_dist / dt) * 3600))
        else:
            self.heading = _bearing(first[0], first[1], raw_lat, raw_lng)
            last_sp = self.smoothed_points[-1]
            dt = now - last_sp[2]
            advance_km = max((self.speed_estimate / 3600) * dt, 0.5)
            new_lat, new_lng = _project_point(last_sp[0], last_sp[1], self.heading, advance_km)
            self.smoothed_points.append((new_lat, new_lng, now))
            _, _, raw_proj_dist = _project_onto_line(raw_lat, raw_lng, last_sp[0], last_sp[1], self.heading)
            raw_proj_dist = max(raw_proj_dist, 0.0)
            if dt > 0:
                raw_speed = max(UAV_MIN_SPEED_KMH, min(UAV_MAX_SPEED_KMH, (raw_proj_dist / dt) * 3600))
                self.speed_estimate = SPEED_SMOOTHING * raw_speed + (1 - SPEED_SMOOTHING) * self.speed_estimate

    def to_firebase(self):
        observed = [[p[0], p[1]] for p in self.smoothed_points]
        predicted = []
        if len(self.smoothed_points) >= 2:
            last = self.smoothed_points[-1]
            for secs in [30, 60]:
                pred_dist = (self.speed_estimate / 3600) * secs
                if pred_dist > 0:
                    plat, plng = _project_point(last[0], last[1], self.heading, pred_dist)
                    predicted.append([plat, plng])
        return {
            "track_id": self.track_id,
            "observed": observed,
            "predicted": predicted,
            "heading_deg": round(self.heading, 1),
            "speed_kmh": round(self.speed_estimate, 0),
            "last_updated": int(self.smoothed_points[-1][2] * 1000),
            "is_test": True,
        }


def cmd_simulate_uav(polygons: dict):
    """Simulate a UAV flight path — writes directly to uav_tracks (no brain.py needed)."""
    print("\nUAV flight path presets:")
    preset_names = list(UAV_FLIGHT_PRESETS.keys())
    for i, name in enumerate(preset_names):
        waves = UAV_FLIGHT_PRESETS[name]
        total_cities = sum(len(w) for w in waves)
        print(f"  {i}: {name} ({total_cities} cities, {len(waves)} waves)")

    try:
        idx = int(input("Pick preset: ").strip())
        preset_name = preset_names[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    waves = UAV_FLIGHT_PRESETS[preset_name]
    # Filter each wave to valid cities
    valid_waves = []
    for wave in waves:
        valid_cities = [c for c in wave if c in polygons and polygons[c].get("polygon")]
        if valid_cities:
            valid_waves.append(valid_cities)

    if not valid_waves:
        print("No valid cities in preset!")
        return

    total_cities = sum(len(w) for w in valid_waves)

    delay = input("Delay between waves in seconds (Enter=3, 0=instant): ").strip()
    delay = float(delay) if delay else 3.0

    print(f"\nSimulating UAV flight: {total_cities} cities in {len(valid_waves)} waves, {delay}s apart")
    for i, wave in enumerate(valid_waves):
        print(f"  Wave {i+1}: {', '.join(wave)}")
    print("Starting in 2 seconds...\n")
    time.sleep(2)

    alert_ref = db.reference(FIREBASE_NODE)
    uav_ref = db.reference(FIREBASE_UAV_NODE)
    track_id = f"uav_test_{int(time.time())}"
    track: _TestUavTrack | None = None
    city_count = 0

    for wi, wave in enumerate(valid_waves):
        now = time.time()

        # Compute average centroid of all cities in this wave
        lats, lngs = [], []
        batch = {}
        for city_he in wave:
            lat, lng = _centroid(polygons[city_he]["polygon"])
            lats.append(lat)
            lngs.append(lng)

            # Write the alert polygon for each city
            payload = _make_payload(city_he, polygons[city_he]["city_name"], "uav")
            key = _sanitize_fb_key(city_he)
            batch[key] = payload

        alert_ref.update(batch)

        # Use average centroid of the wave as the single observation point
        avg_lat = sum(lats) / len(lats)
        avg_lng = sum(lngs) / len(lngs)

        if track is None:
            track = _TestUavTrack(track_id, avg_lat, avg_lng, now)
        else:
            track.add_point(avg_lat, avg_lng, now)

        # Push track data
        track_payload = track.to_firebase()
        uav_ref.update({track_id: track_payload})

        city_count += len(wave)
        cities_str = ", ".join(wave)
        print(f"  Wave {wi+1}/{len(valid_waves)}: [{cities_str}]  →  track updated ({len(track.smoothed_points)} pts)")

        if wi < len(valid_waves) - 1 and delay > 0:
            time.sleep(delay)

    print(f"\nDone! {city_count} UAV alerts in {len(valid_waves)} waves written to Firebase.")


def cmd_send_screenshot():
    """Manually capture and send a screenshot to the developer."""
    if not _BOT_TOKEN:
        print("Screenshot not configured. Set CLEARMAP_BOT_TOKEN in config.env")
        return

    print("\nCapture from:")
    print(f"  1) Localhost ({_LOCALHOST_URL})")
    print(f"  2) Production ({_PROD_URL})")
    choice = input("> ").strip()
    if choice == "2":
        url = _PROD_URL
    else:
        url = _LOCALHOST_URL

    try:
        from screenshot_alerts import quick_capture_and_send
        print(f"Capturing from {url} → sending to {_DEV_CHAT_ID}...")
        quick_capture_and_send(
            _BOT_TOKEN, _DEV_CHAT_ID,
            caption="📸 Manual screenshot",
            url=url,
        )
    except Exception as e:
        print(f"Failed: {e}")


def cmd_salvo_scenario(polygons: dict):
    print("\nSalvo presets:")
    preset_names = list(SALVO_PRESETS.keys())
    for i, name in enumerate(preset_names):
        print(f"  {i}: {name}")
    
    try:
        idx = int(input("Pick preset: ").strip())
        preset_name = preset_names[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    districts = SALVO_PRESETS[preset_name]
    cities = []
    for d in districts:
        cities.extend(DISTRICT_AREAS.get(d, []))
    
    print(f"Executing SALVO: {len(cities)} cities in {len(districts)} districts")
    _send_batch(cities, polygons, status="alert")


def main():
    # Load polygons
    with open(POLYGONS_FILE, encoding="utf-8") as f:
        polygons = json.load(f)

    # Init Firebase
    cred = credentials.Certificate(str(SERVICE_ACCOUNT))
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

    print(f"Loaded {len(polygons)} cities.")
    print(f"Screenshot bot: {'✅ configured' if _BOT_TOKEN else '❌ not configured (set CLEARMAP_BOT_TOKEN in config.env)'}")
    subs = _load_subscribers()
    if subs:
        print(f"  📋 {len(subs)} subscribers will receive screenshots")

    while True:
        print("\n--- Alerts ---")
        print("1) Add single   2) Search batch   3) Alert District  4) Alert Region")
        print("5) Clear all    6) Clear test     7) Clear one       8) Show active")
        print("--- Scenarios ---")
        print("11) Simulate UAV (Real-time/Fast-forward)")
        print("12) Salvo Attack")
        print("--- Screenshot ---")
        print("13) Send screenshot now")
        print("0) Exit")
        
        choice = input("> ").strip()

        if choice == "1":
            cmd_add_alert(polygons)
        elif choice == "2":
            cmd_batch_alert(polygons)
        elif choice == "3":
            cmd_district_alert(polygons)
        elif choice == "4":
            cmd_region_alert(polygons)
        elif choice == "5":
            cmd_clear_all()
        elif choice == "6":
            cmd_clear_test_alerts()
        elif choice == "7":
            cmd_clear_one(polygons)
        elif choice == "8":
            cmd_show_active()
        elif choice == "11":
            cmd_simulate_uav(polygons)
        elif choice == "12":
            cmd_salvo_scenario(polygons)
        elif choice == "13":
            cmd_send_screenshot()
        elif choice == "0":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
