import json
import os
from pathlib import Path

# Path to the source area_to_polygon.json from amitfin/oref_alert
SOURCE_FILE = Path(__file__).parent / "oref_alert/custom_components/oref_alert/metadata/area_to_polygon.json"
# Output file for brain.py
OUTPUT_FILE = Path(__file__).parent / "polygons.json"

def convert_coordinates(coords: list[list[float]]) -> list[list[float]]:
    """Convert Leaflet/Oref [lat, lng] to Leaflet [lat, lng] or verify they are correct."""
    # Oref source JSON has them as [lat, lng] already: [31.811685030330167, 35.102242650898944]
    # In fetch_polygons.py we did: return [[pt[1], pt[0]] for pt in coords] because GeoJSON was [lng, lat]
    # But area_to_polygon.json is [lat, lon], so we just return them as-is.
    return coords

def main():
    if not SOURCE_FILE.exists():
        print(f"Source file not found: {SOURCE_FILE}")
        print("Please extract area_to_polygon.json from area_to_polygon.json.zip in oref_alert/custom_components/oref_alert/metadata/")
        return

    print("Loading Oref exact polygons...")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        oref_data = json.load(f)

    print(f"Loaded {len(oref_data)} alert zones.")

    lookup = {}
    skipped = 0

    for area_name_he, polygon_coords in oref_data.items():
        if not polygon_coords:
            skipped += 1
            print(f"[SKIP] No polygon for {area_name_he}")
            continue

        # The coordinates from area_to_polygon are already [lat, lon]
        # brain.py expects `city_name` (used for english or display) and `city_name_he`.
        # Since we only have the Hebrew name for the zones, we'll use it for both, 
        # or we could map them to English using other metadata. But brain.py only really needs city_name to pass to the frontend.
        # It's better to pass the Hebrew name to the frontend anyway since it's an exact zone like "ירושלים - מרכז".
        
        lookup[area_name_he] = {
            "city_name": area_name_he, # Using Hebrew name as English fallback since we lack a strict dictionary
            "city_name_he": area_name_he,
            "polygon": polygon_coords
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)

    print(f"\nDone! Saved {len(lookup)} alert zones to {OUTPUT_FILE}")
    if skipped:
        print(f"Skipped {skipped} entries without polygons.")

if __name__ == "__main__":
    main()
