# localities.py
#
# Gazetteer of localities within ~5 km of NIT Hamirpur (31.7115, 76.5117),
# Himachal Pradesh. Used to:
#   1. Tag simulated complaints with a realistic, geographically constrained location.
#   2. Extract an approximate location from real tweet text by keyword matching.
#
# Coordinates are approximate (sourced from public map data) and good enough
# for a proof-of-concept location radius check — not survey-grade.

NIT_HAMIRPUR = {"name": "NIT Hamirpur Campus", "lat": 31.7115, "lon": 76.5117}

# Each entry: name, representative lat/lon, and keywords used to match
# against complaint text (lowercase, no punctuation — see clean_text()).
LOCALITIES = [
    {
        "name": "NIT Hamirpur Campus",
        "lat": 31.7115, "lon": 76.5117,
        "keywords": ["nit hamirpur", "nit campus", "hostel", "shivalik hostel", "college campus"],
    },
    {
        "name": "Degree College Chowk",
        "lat": 31.7040, "lon": 76.5170,
        "keywords": ["degree college chowk", "college chowk", "bus stand", "chowk"],
    },
    {
        "name": "Hamirpur Town Center",
        "lat": 31.6908, "lon": 76.5177,
        "keywords": ["town center", "town centre", "hamirpur town", "main bazaar", "bazaar"],
    },
    {
        "name": "Green Park Colony",
        "lat": 31.6985, "lon": 76.5140,
        "keywords": ["green park colony", "green park"],
    },
    {
        "name": "Patel Nagar",
        "lat": 31.7005, "lon": 76.5210,
        "keywords": ["patel nagar"],
    },
    {
        "name": "Vegetable Market (Sabzi Mandi)",
        "lat": 31.6950, "lon": 76.5155,
        "keywords": ["vegetable market", "sabzi mandi", "mandi", "market"],
    },
    {
        "name": "District Library Area",
        "lat": 31.6930, "lon": 76.5190,
        "keywords": ["district library", "library"],
    },
    {
        "name": "Sarahkar",
        "lat": 31.7050, "lon": 76.5050,
        "keywords": ["sarahkar"],
    },
    {
        "name": "Majhog Sultani",
        "lat": 31.7300, "lon": 76.5320,
        "keywords": ["majhog", "majhog sultani"],
    },
    {
        "name": "Daruhi",
        "lat": 31.6870, "lon": 76.5340,
        "keywords": ["daruhi"],
    },
    {
        "name": "NH-88 Bypass Road",
        "lat": 31.7150, "lon": 76.5220,
        "keywords": ["nh88", "nh 88", "national highway", "bypass road", "highway"],
    },
]


def find_locality_for_text(text: str):
    """Return the first matching locality dict for the given (lowercased,
    cleaned) text, or None if no keyword matches."""
    if not text:
        return None
    t = text.lower()
    for loc in LOCALITIES:
        for kw in loc["keywords"]:
            if kw in t:
                return loc
    return None


def random_locality(rng=None):
    """Return a random locality (for the simulated complaint generator)."""
    import random
    r = rng or random
    return r.choice(LOCALITIES)
