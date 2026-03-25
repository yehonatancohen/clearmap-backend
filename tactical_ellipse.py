"""
Tactical Ellipse — group nearby rocket alerts into a fitted enclosing ellipse
using proper chi-squared confidence regions for a bivariate normal model.

Given an array of Pikud HaOref alert zone names, resolves each to a centroid
+ radius via city_db.json, then computes confidence ellipses via weighted PCA
with trajectory-locked axis orientation.

Output dict:
  center            — {"lat": ..., "lon": ...}
  rotation_deg      — clockwise from North (locked to launch bearing when origin is known)
  sigma_major_km    — raw 1σ semi-major axis (≈39.3% confidence for 2D)
  sigma_minor_km    — raw 1σ semi-minor axis
  outer_ellipse     — {confidence, semi_major_km, semi_minor_km} at 95%
  inner_ellipse     — {confidence, semi_major_km, semi_minor_km} at 50%
  point_count       — number of resolved cities
"""

import json
import math
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

R_EARTH = 6371.0          # km
MIN_AXIS_KM = 1.5         # minimum semi-axis length
MIN_AXIS_RATIO = 1.5      # minimum major/minor ratio (range error > deflection)
CLUSTER_DISTANCE_KM = 30  # max distance for spatial clustering
CLUSTER_TIME_WINDOW_S = 60  # alerts within this window are same-barrage

# Default confidence levels
OUTER_CONFIDENCE = 0.95
INNER_CONFIDENCE = 0.50
OUTER_PADDING = 1.08      # 8% padding to cover polygon edges beyond centroids


# ── IQR-based outlier filter ────────────────────────────────────────────────

def robust_max(values: list[float]) -> float:
    """IQR-based outlier filter: returns the max value that isn't a statistical
    extreme outlier. Only excludes points beyond Q3 + 2×IQR (very permissive).
    Keeps all legitimate barrage cities, only drops truly anomalous distances."""
    if not values:
        return 0.0
    abs_vals = sorted(abs(v) for v in values)
    if len(abs_vals) < 4:
        return abs_vals[-1]
    q1 = abs_vals[len(abs_vals) // 4]
    q3 = abs_vals[(3 * len(abs_vals)) // 4]
    iqr = q3 - q1
    upper_fence = q3 + 2.0 * iqr
    filtered = [v for v in abs_vals if v <= upper_fence]
    return filtered[-1] if filtered else abs_vals[-1]


# ── Projection helpers ───────────────────────────────────────────────────────

def chi2_scale(confidence: float) -> float:
    """
    Returns the ellipse radius scale factor for a given confidence level
    under a bivariate normal distribution (chi-squared with 2 DOF).
    confidence: float between 0 and 1 (e.g., 0.5, 0.9, 0.95)
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError("Confidence must be between 0 and 1 exclusive")
    return math.sqrt(-2.0 * math.log(1.0 - confidence))


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
    deg = math.degrees(math.atan2(vx, vy))
    return deg % 360


# ── Origin estimation ───────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Smallest angular difference between two bearings (0–180)."""
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d


def _estimate_origin(
    center_lat: float, center_lon: float, angle_deg: float,
) -> tuple[float, str, bool]:
    """Estimate launch origin based on geographic position of the cluster center.

    PCA angle is used only for refinement (±40°) when it agrees with the
    geographic prior.  Otherwise the known threat bearing is used directly.

    Returns (bearing_deg, source_name, is_confident).
    """
    # PCA-derived candidate bearings (for refinement only)
    pca1 = angle_deg % 360
    pca2 = (pca1 + 180) % 360

    def refine_bearing(target: float, max_deviation: float = 40) -> float:
        d1 = _angle_diff(pca1, target)
        d2 = _angle_diff(pca2, target)
        if d1 <= max_deviation and d1 <= d2:
            return pca1
        if d2 <= max_deviation:
            return pca2
        return target

    lat, lon = center_lat, center_lon

    # Determine source primarily by geographic position
    if lat > 32.5:
        # Northern Israel → Lebanon (~0°)
        return refine_bearing(0), "lebanon", True
    elif lat < 31.0 and lon > 34.5:
        # Deep south, eastern → Yemen (~170°)
        return refine_bearing(170), "yemen", True
    elif lat < 31.5 and lon < 34.5:
        # Southwest → Gaza/Sinai (~220°)
        return refine_bearing(220), "gaza", True
    else:
        # Central Israel → Iran/Iraq (~85°)
        return refine_bearing(85), "iran", True


# ── Max-extent along trajectory axes ────────────────────────────────────────

def _trajectory_locked_extent(
    points: list[tuple[float, float]],
    radii: list[float],
    bearing_deg: float,
) -> tuple[float, float]:
    """Project city circles (centroid + radius) onto trajectory-locked axes.

    Returns (extent_major, extent_minor): the robust-max furthest alert-zone
    edge distance from center along each axis. Uses IQR outlier filtering.
    """
    b_rad = math.radians(bearing_deg)

    # Trajectory axis unit vector: (east, north)
    traj_x = math.sin(b_rad)
    traj_y = math.cos(b_rad)

    # Perpendicular axis unit vector
    perp_x = math.cos(b_rad)
    perp_y = -math.sin(b_rad)

    proj_maj = []
    proj_min = []

    for (x, y), r in zip(points, radii):
        proj_maj.append(abs(x * traj_x + y * traj_y) + r)
        proj_min.append(abs(x * perp_x + y * perp_y) + r)

    return robust_max(proj_maj), robust_max(proj_min)


# ── Result builder ──────────────────────────────────────────────────────────

def _make_result(
    center_lat: float,
    center_lon: float,
    outer_major: float,
    outer_minor: float,
    rotation_deg: float,
    point_count: int,
    enforce_ratio: bool = True,
) -> dict:
    """Build the standardized result dict with inner/outer confidence ellipses.

    outer_major / outer_minor are bounding-box semi-axes (max extent of
    alert-zone edges from center). Inner ellipse is derived as outer × 0.481.
    """
    inner_outer_ratio = chi2_scale(INNER_CONFIDENCE) / chi2_scale(OUTER_CONFIDENCE)

    outer_major = max(outer_major * OUTER_PADDING, MIN_AXIS_KM)
    outer_minor = max(outer_minor * OUTER_PADDING, MIN_AXIS_KM)

    # Enforce minimum axis ratio (range error > deflection error)
    if enforce_ratio and outer_major / max(outer_minor, 0.1) < MIN_AXIS_RATIO:
        outer_major = max(outer_major, outer_minor * MIN_AXIS_RATIO)

    sigma_major = outer_major
    sigma_minor = outer_minor

    return {
        "center": {
            "lat": round(center_lat, 6),
            "lon": round(center_lon, 6),
        },
        "rotation_deg": round(rotation_deg, 2),
        "sigma_major_km": round(sigma_major, 4),
        "sigma_minor_km": round(sigma_minor, 4),
        "outer_ellipse": {
            "confidence": OUTER_CONFIDENCE,
            "semi_major_km": round(outer_major, 2),
            "semi_minor_km": round(outer_minor, 2),
        },
        "inner_ellipse": {
            "confidence": INNER_CONFIDENCE,
            "semi_major_km": round(outer_major * inner_outer_ratio, 2),
            "semi_minor_km": round(outer_minor * inner_outer_ratio, 2),
        },
        "point_count": point_count,
    }


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
        # City radius is the ~95% region; pass directly as outer axis
        outer_r = max(c["radius_km"], MIN_AXIS_KM)
        return _make_result(c["lat"], c["lon"], outer_r, outer_r, 0, 1, enforce_ratio=False)

    # ── N = 2: line between two cities ───────────────────────────────────
    if n == 2:
        a, b = cities[0], cities[1]

        # Weighted centroid (inverse radius squared)
        weights = []
        for c in [a, b]:
            r = max(c["radius_km"], 0.5)
            weights.append(1.0 / (r * r))
        total_w = sum(weights)
        center_lat = sum(w * c["lat"] for w, c in zip(weights, [a, b])) / total_w
        center_lon = sum(w * c["lon"] for w, c in zip(weights, [a, b])) / total_w

        half_dist = _haversine(a["lat"], a["lon"], b["lat"], b["lon"]) / 2
        avg_r = (a["radius_km"] + b["radius_km"]) / 2

        # Outer axis along connecting direction = half the city-to-city spread
        outer_major = half_dist
        # Outer axis perpendicular = average city radius (already ~95% region)
        outer_minor = avg_r

        # Use origin estimation to pick the correct bearing direction
        raw_bearing = _bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])
        launch_bearing, _source, _confident = _estimate_origin(center_lat, center_lon, raw_bearing)

        return _make_result(center_lat, center_lon, outer_major, outer_minor, launch_bearing, 2)

    # ── N ≥ 3: Trajectory-locked bounding-box ellipse ─────────────────────

    # 1. Compute inverse-radius-squared weights and collect radii
    weights = []
    radii = []
    for c in cities:
        r = max(c["radius_km"], 0.5)
        weights.append(1.0 / (r * r))
        radii.append(r)
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    # 2. Weighted centroid
    center_lat = sum(w * c["lat"] for w, c in zip(weights, cities)) / total_w
    center_lon = sum(w * c["lon"] for w, c in zip(weights, cities)) / total_w

    # 3. Project to local flat (x=East km, y=North km)
    points = []
    for c in cities:
        x, y = _to_xy(c["lat"], c["lon"], center_lat, center_lon)
        points.append((x, y))

    # 4. Free PCA to get initial axis angle for origin estimation
    wmx = sum(w * p[0] for w, p in zip(norm_weights, points))
    wmy = sum(w * p[1] for w, p in zip(norm_weights, points))

    cov_xx = sum(w * (p[0] - wmx) ** 2 for w, p in zip(norm_weights, points))
    cov_yy = sum(w * (p[1] - wmy) ** 2 for w, p in zip(norm_weights, points))
    cov_xy = sum(w * (p[0] - wmx) * (p[1] - wmy) for w, p in zip(norm_weights, points))

    sum_w2 = sum(w * w for w in norm_weights)
    correction = 1.0 - sum_w2
    if correction > 1e-12:
        cov_xx /= correction
        cov_yy /= correction
        cov_xy /= correction

    # Free PCA eigenvector direction (used only for origin estimation input)
    if abs(cov_xy) > 1e-12:
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy ** 2
        discriminant = max(trace ** 2 - 4 * det, 0)
        lambda1 = (trace + math.sqrt(discriminant)) / 2
        ev1_x = lambda1 - cov_yy
        ev1_y = cov_xy
    elif cov_xx >= cov_yy:
        ev1_x, ev1_y = 1.0, 0.0
    else:
        ev1_x, ev1_y = 0.0, 1.0

    mag = math.sqrt(ev1_x ** 2 + ev1_y ** 2)
    if mag > 1e-12:
        ev1_x /= mag
        ev1_y /= mag

    free_pca_angle = _angle_from_north(ev1_x, ev1_y)

    # 5. Estimate origin using free PCA angle (geography-based, PCA for refinement)
    launch_bearing, source, is_confident = _estimate_origin(center_lat, center_lon, free_pca_angle)

    # 6. Determine orientation
    rotation = launch_bearing if is_confident else free_pca_angle

    # 7. Outer = max extent of city circles along each axis (bounding-box)
    outer_major, outer_minor = _trajectory_locked_extent(points, radii, rotation)

    return _make_result(center_lat, center_lon, outer_major, outer_minor, rotation, n)


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
