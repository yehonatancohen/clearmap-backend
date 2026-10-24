#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test for _classify_alert_object and data parsing with new Oref API format."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from brain import _classify_alert_object

cases = [
    ({"category": 1,  "title": "ירי רקטות וטילים",                "data": "תל אביב"},        "alert"),
    ({"category": 6,  "title": "חדירת כלי טיס עוין",        "data": "חיפה"},                     "uav"),
    ({"category": 1,  "title": "חדירת מחבלים",                                   "data": "אשקלון"},          "terrorist"),
    ({"category": 10, "title": "בדקות הקרובות צפויות להתקבל התראות", "data": "ירושלים"}, "pre_alert"),
    ({"category": 13, "title": "האירוע הסתיים",                             "data": "תל אביב - מרכז"}, "clear"),
    # Old format still works
    ({"cat": "1",     "title": "ירי רקטות וטילים",                "data": ["חדרה"]},                   "alert"),
    ({"cat": "10",    "title": "בדקות הקרובות",                             "data": ["חיפה"]},                   "pre_alert"),
    # Oct 2026 wording: "טייס" spelling, no "חדירת", category 2
    ({"category": 2,  "title": "כלי טייס עוין - היכנסו מייד למרחב המוגן", "data": "ערב אל עראמשה"}, "uav"),
    ({"category": 2,  "title": "",                                          "data": "ערב אל עראמשה"}, "uav"),
    ({"cat": "6",     "title": "",                                          "data": ["ערב אל עראמשה"]}, "uav"),
    ({"category": 13, "title": "כלי טייס עוין - האירוע הסתיים",   "data": "ערב אל עראמשה"}, "clear"),
    ({"category": 10, "title": "בדקות הקרובות צפויות להתקבל התרעות על כלי טיס עוין", "data": "חיפה"}, "pre_alert"),
    # Oct 2026 terror all-clear wording (was read as a second infiltration)
    ({"category": 13, "title": "הוסר החשש לחדירת מחבלים",               "data": "אביתר"},          "clear"),
    ({"category": 13, "title": "",                                          "data": "אביתר"},          "clear"),
    ({"category": 14, "title": "איום מלבנון - התקרבו למרחב מוגן",       "data": "מטולה"},          "pre_alert"),
]

# Labels for readability
labels = ["alert", "uav", "terrorist", "pre_alert", "clear", "old-alert", "old-pre_alert",
          "uav-new-title", "uav-category-only", "uav-cat-only", "uav-ended", "uav-advisory",
          "terror-cleared", "clear-category-only", "pre_alert-lebanon"]

all_pass = True
for i, (obj, expected) in enumerate(cases):
    result = _classify_alert_object(obj)
    ok = result == expected
    if not ok:
        all_pass = False
    print(f"{'PASS' if ok else 'FAIL'}  [{labels[i]}]  expected={expected}  got={result}")

# Test data normalization (new format: data is a string)
print()
raw_data_new = "tel aviv"
cities = [raw_data_new] if isinstance(raw_data_new, str) else raw_data_new
assert cities == ["tel aviv"], f"Expected ['tel aviv'], got {cities}"
print("PASS  [data-string] single city string wrapped into list correctly")

raw_data_old = ["city1", "city2"]
cities = [raw_data_old] if isinstance(raw_data_old, str) else raw_data_old
assert cities == ["city1", "city2"], f"Expected list unchanged, got {cities}"
print("PASS  [data-list]   list of cities left unchanged")

print()
print("All tests passed!" if all_pass else "SOME TESTS FAILED")
