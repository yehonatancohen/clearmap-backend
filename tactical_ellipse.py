"""
Tactical Ellipse — group nearby rocket alerts into a fitted enclosing ellipse.

Given an array of Pikud HaOref alert zone names, resolves each to a centroid
+ radius via city_db.json, then computes a minimum enclosing ellipse using PCA.

Output dict:
  center_lat, center_lon   — geographic center
  major_axis_km, minor_axis_km — semi-axis lengths
  rotation_deg             — clockwise from North
  point_count              — number of resolved cities
"""

import json
import math
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

R_EARTH = 6371.0          # km
MIN_AXIS_KM = 1.5         # minimum semi-axis length
PADDING_FACTOR = 1.2       # 20 % buffer around PCA extents

# ── Load city database ───────────────────────────────────────────────────────

_CITY_DB_PATH = Path(__file__).parent / "city_db.json"
_city_db: dict[str, dict] = {}


def _load_city_db() -> dict[str, dict]:
    global _city_db
    if not _city_db:
        with open(_CITY_DB_PATH, "r", encoding="utf-8") as f:
            _city_db = json.load(f)
    return _city_db


# ── Projection helpers ───────────────────────────────────────────────────────

def _to_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Equirectangular projection to (x_east_km, y_north_km)."""
    cos_ref = math.cos(math.radians(ref_lat))
    x = math.radians(lon - ref_lon) * cos_ref * R_EARTH
    y = math.radians(lat - ref_lat) * R_EARTH
    return (x, y)


def _angle_from_north(vx: float, vy: float) -> float:
    """Angle of vector (vx, vy) measured clockwise from North (positive-y).
    Returns degrees in [0, 360).
    """
    # atan2(east, north) gives clockwise-from-north
    deg = math.degrees(math.atan2(vx, vy))
    return deg % 360


# ── Core algorithm ───────────────────────────────────────────────────────────

def compute_ellipse(zone_names: list[str]) -> Optional[dict]:
    """Compute a tactical ellipse for a list of Pikud HaOref zone names.

    Returns None if no zones can be resolved.
    """
    db = _load_city_db()

    # Resolve zone names to city records
    cities = []
    for name in zone_names:
        rec = db.get(name)
        if rec:
            cities.append(rec)

    n = len(cities)
    if n == 0:
        return None

    # ── N = 1: simple circle ─────────────────────────────────────────────
    if n == 1:
        c = cities[0]
        r = max(c["radius_km"], MIN_AXIS_KM)
        return {
            "center_lat": c["lat"],
            "center_lon": c["lon"],
            "major_axis_km": r,
            "minor_axis_km": r,
            "rotation_deg": 0,
            "point_count": 1,
        }

    # ── N = 2: line between two cities ───────────────────────────────────
    if n == 2:
        a, b = cities[0], cities[1]
        center_lat = (a["lat"] + b["lat"]) / 2
        center_lon = (a["lon"] + b["lon"]) / 2

        half_dist = _haversine(a["lat"], a["lon"], b["lat"], b["lon"]) / 2
        larger_r = max(a["radius_km"], b["radius_km"])
        smaller_r = min(a["radius_km"], b["radius_km"])

        major = half_dist + larger_r
        minor = max(smaller_r, MIN_AXIS_KM)

        # Bearing from a → b, clockwise from North
        rotation = _bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])

        return {
            "center_lat": round(center_lat, 6),
            "center_lon": round(center_lon, 6),
            "major_axis_km": round(major, 2),
            "minor_axis_km": round(minor, 2),
            "rotation_deg": round(rotation, 2),
            "point_count": 2,
        }

    # ── N ≥ 3: PCA ellipse ──────────────────────────────────────────────

    # 1. Mean lat/lon → projection origin
    mean_lat = sum(c["lat"] for c in cities) / n
    mean_lon = sum(c["lon"] for c in cities) / n

    # 2. Project to local flat (x=East km, y=North km)
    points = []
    for c in cities:
        x, y = _to_xy(c["lat"], c["lon"], mean_lat, mean_lon)
        points.append((x, y, c["radius_km"]))

    # 3. Covariance matrix of (x, y) positions
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n

    cov_xx = sum((p[0] - mx) ** 2 for p in points) / n
    cov_yy = sum((p[1] - my) ** 2 for p in points) / n
    cov_xy = sum((p[0] - mx) * (p[1] - my) for p in points) / n

    # 4. Eigenvalues of 2×2 symmetric matrix [[cov_xx, cov_xy],[cov_xy, cov_yy]]
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy ** 2
    discriminant = max(trace ** 2 - 4 * det, 0)
    sqrt_disc = math.sqrt(discriminant)

    lambda1 = (trace + sqrt_disc) / 2   # larger eigenvalue
    lambda2 = (trace - sqrt_disc) / 2   # smaller eigenvalue

    # 5. Eigenvector for lambda1 (major axis direction)
    if abs(cov_xy) > 1e-12:
        ev1_x = lambda1 - cov_yy
        ev1_y = cov_xy
    elif cov_xx >= cov_yy:
        ev1_x, ev1_y = 1.0, 0.0
    else:
        ev1_x, ev1_y = 0.0, 1.0

    # Normalize
    mag = math.sqrt(ev1_x ** 2 + ev1_y ** 2)
    if mag > 1e-12:
        ev1_x /= mag
        ev1_y /= mag

    # Perpendicular eigenvector (minor axis)
    ev2_x, ev2_y = -ev1_y, ev1_x

    # 6. Project points onto principal axes, extend by city radius
    max_major = 0.0
    max_minor = 0.0
    for x, y, r in points:
        dx, dy = x - mx, y - my
        proj_major = abs(dx * ev1_x + dy * ev1_y)
        proj_minor = abs(dx * ev2_x + dy * ev2_y)
        max_major = max(max_major, proj_major + r)
        max_minor = max(max_minor, proj_minor + r)

    major_km = max(max_major * PADDING_FACTOR, MIN_AXIS_KM)
    minor_km = max(max_minor * PADDING_FACTOR, MIN_AXIS_KM)

    # 7. Rotation: angle of major eigenvector clockwise from North
    rotation = _angle_from_north(ev1_x, ev1_y)

    return {
        "center_lat": round(mean_lat, 6),
        "center_lon": round(mean_lon, 6),
        "major_axis_km": round(major_km, 2),
        "minor_axis_km": round(minor_km, 2),
        "rotation_deg": round(rotation, 2),
        "point_count": n,
    }


# ── Internal geo helpers (standalone, no brain.py dependency) ────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point1 → point2, clockwise from North [0, 360)."""
    dlon = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(la2)
    y = (math.cos(la1) * math.sin(la2)
         - math.sin(la1) * math.cos(la2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360
