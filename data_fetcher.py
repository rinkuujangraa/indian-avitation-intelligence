"""
data_fetcher.py
---------------
Fetches live flight data from the AirLabs API and returns a clean,
named Pandas DataFrame.

AirLabs API docs: https://airlabs.co/docs/

Setup:
    1. Create a .env file in your project root
    2. Add this line: AIRLABS_API_KEY=your_key_here
    3. Run: pip install python-dotenv requests pandas
"""

import os
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from notifier import send_alert

# ── Load environment variables ─────────────────────────────────────────────────
# load_dotenv() reads your .env file and loads the variables into the
# environment. This means os.getenv("AIRLABS_API_KEY") will find your key
# WITHOUT you ever hardcoding it in the source code.
load_dotenv()

# ── Logger setup ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s"
)
logger = logging.getLogger(__name__)


# ── AirLabs base URL ───────────────────────────────────────────────────────────
BASE_URL = "https://airlabs.co/api/v9"
SCHEDULE_CACHE_TTL = 900
_MAX_CACHE_ENTRIES = 200
_SCHEDULE_CACHE: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_FLIGHT_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}   # params → (ts, df) stale fallback on 429
_MAX_FLIGHT_CACHE_ENTRIES = 20  # at most 20 region/filter combos kept in memory


# ── Bounding boxes for common regions ─────────────────────────────────────────
# Format: "min_lat,min_lon,max_lat,max_lon"
# AirLabs expects the bbox as a single comma-separated string.
REGIONS = {
    "world":         None,
    "europe":        "36,-15,72,40",
    "north_america": "24,-130,72,-60",
    "middle_east":   "12,25,42,65",
    "south_asia":    "5,60,40,95",
    "india":         "8,68,37,97",
    "gulf":          "22,50,27,57",
}

# Map region → map centre (lat, lon) + zoom level for Folium
REGION_MAP_SETTINGS = {
    "world":         {"center": (20,   0),  "zoom": 2},
    "europe":        {"center": (51,  10),  "zoom": 4},
    "north_america": {"center": (40, -98),  "zoom": 3},
    "middle_east":   {"center": (25,  45),  "zoom": 5},
    "south_asia":    {"center": (20,  78),  "zoom": 4},
    "india":         {"center": (22,  80),  "zoom": 5},
    "gulf":          {"center": (24,  54),  "zoom": 6},
}

INDIAN_AIRPORTS = [
    # ── Tier 1: Metro International Hubs ──
    {"iata": "DEL", "name": "Indira Gandhi International", "city": "Delhi", "lat": 28.5562, "lng": 77.1000},
    {"iata": "BOM", "name": "Chhatrapati Shivaji Maharaj International", "city": "Mumbai", "lat": 19.0896, "lng": 72.8656},
    {"iata": "BLR", "name": "Kempegowda International", "city": "Bengaluru", "lat": 13.1986, "lng": 77.7066},
    {"iata": "HYD", "name": "Rajiv Gandhi International", "city": "Hyderabad", "lat": 17.2403, "lng": 78.4294},
    {"iata": "MAA", "name": "Chennai International", "city": "Chennai", "lat": 12.9941, "lng": 80.1709},
    {"iata": "CCU", "name": "Netaji Subhas Chandra Bose International", "city": "Kolkata", "lat": 22.6547, "lng": 88.4467},
    # ── Tier 2: Major Domestic Airports ──
    {"iata": "AMD", "name": "Sardar Vallabhbhai Patel International", "city": "Ahmedabad", "lat": 23.0772, "lng": 72.6347},
    {"iata": "PNQ", "name": "Pune Airport", "city": "Pune", "lat": 18.5821, "lng": 73.9197},
    {"iata": "GOI", "name": "Goa International", "city": "Goa", "lat": 15.3808, "lng": 73.8314},
    {"iata": "GOX", "name": "Manohar International", "city": "Goa", "lat": 15.7443, "lng": 73.8607},
    {"iata": "COK", "name": "Cochin International", "city": "Kochi", "lat": 10.1520, "lng": 76.4019},
    {"iata": "TRV", "name": "Trivandrum International", "city": "Thiruvananthapuram", "lat": 8.4821, "lng": 76.9201},
    {"iata": "CJB", "name": "Coimbatore International", "city": "Coimbatore", "lat": 11.0300, "lng": 77.0434},
    {"iata": "IXM", "name": "Madurai Airport", "city": "Madurai", "lat": 9.8345, "lng": 78.0934},
    {"iata": "TRZ", "name": "Tiruchirappalli International", "city": "Tiruchirappalli", "lat": 10.7654, "lng": 78.7097},
    {"iata": "VGA", "name": "Vijayawada International", "city": "Vijayawada", "lat": 16.5304, "lng": 80.7968},
    {"iata": "VTZ", "name": "Visakhapatnam Airport", "city": "Visakhapatnam", "lat": 17.7212, "lng": 83.2245},
    {"iata": "BBI", "name": "Biju Patnaik International", "city": "Bhubaneswar", "lat": 20.2444, "lng": 85.8178},
    {"iata": "PAT", "name": "Jay Prakash Narayan", "city": "Patna", "lat": 25.5913, "lng": 85.0870},
    {"iata": "LKO", "name": "Chaudhary Charan Singh International", "city": "Lucknow", "lat": 26.7606, "lng": 80.8893},
    {"iata": "JAI", "name": "Jaipur International", "city": "Jaipur", "lat": 26.8242, "lng": 75.8122},
    {"iata": "UDR", "name": "Maharana Pratap", "city": "Udaipur", "lat": 24.6177, "lng": 73.8961},
    {"iata": "ATQ", "name": "Sri Guru Ram Dass Jee International", "city": "Amritsar", "lat": 31.7096, "lng": 74.7973},
    {"iata": "IXC", "name": "Chandigarh International", "city": "Chandigarh", "lat": 30.6735, "lng": 76.7885},
    {"iata": "SXR", "name": "Srinagar Airport", "city": "Srinagar", "lat": 33.9871, "lng": 74.7743},
    {"iata": "JDH", "name": "Jodhpur Airport", "city": "Jodhpur", "lat": 26.2511, "lng": 73.0489},
    {"iata": "IDR", "name": "Devi Ahilyabai Holkar", "city": "Indore", "lat": 22.7218, "lng": 75.8011},
    {"iata": "NAG", "name": "Dr. Babasaheb Ambedkar International", "city": "Nagpur", "lat": 21.0922, "lng": 79.0472},
    {"iata": "RAJ", "name": "Rajkot International", "city": "Rajkot", "lat": 22.3092, "lng": 70.7795},
    {"iata": "IXR", "name": "Birsa Munda", "city": "Ranchi", "lat": 23.3143, "lng": 85.3217},
    # ── Tier 3: Regional & Smaller Airports ──
    {"iata": "GAU", "name": "Lokpriya Gopinath Bordoloi International", "city": "Guwahati", "lat": 26.1061, "lng": 91.5859},
    {"iata": "IMF", "name": "Tulihal Airport", "city": "Imphal", "lat": 24.7600, "lng": 93.8967},
    {"iata": "DIB", "name": "Dibrugarh Airport", "city": "Dibrugarh", "lat": 27.4839, "lng": 95.0169},
    {"iata": "IXB", "name": "Bagdogra Airport", "city": "Siliguri", "lat": 26.6812, "lng": 88.3286},
    {"iata": "AJL", "name": "Lengpui Airport", "city": "Aizawl", "lat": 23.8406, "lng": 92.6197},
    {"iata": "IXA", "name": "Agartala Airport", "city": "Agartala", "lat": 23.8870, "lng": 91.2404},
    {"iata": "RPR", "name": "Swami Vivekananda Airport", "city": "Raipur", "lat": 21.1804, "lng": 81.7388},
    {"iata": "VNS", "name": "Lal Bahadur Shastri International", "city": "Varanasi", "lat": 25.4524, "lng": 82.8593},
    {"iata": "GAY", "name": "Gaya Airport", "city": "Gaya", "lat": 24.7443, "lng": 84.9512},
    {"iata": "IXJ", "name": "Jammu Airport", "city": "Jammu", "lat": 32.6891, "lng": 74.8374},
    {"iata": "IXL", "name": "Kushok Bakula Rimpochee", "city": "Leh", "lat": 34.1359, "lng": 77.5465},
    {"iata": "DHM", "name": "Gaggal Airport", "city": "Dharamshala", "lat": 32.1651, "lng": 76.2634},
    {"iata": "DED", "name": "Jolly Grant Airport", "city": "Dehradun", "lat": 30.1897, "lng": 78.1803},
    {"iata": "KUU", "name": "Bhuntar Airport", "city": "Kullu", "lat": 31.8767, "lng": 77.1544},
    {"iata": "AGR", "name": "Agra Airport", "city": "Agra", "lat": 27.1557, "lng": 77.9609},
    {"iata": "GWL", "name": "Rajmata Vijaya Raje Scindia", "city": "Gwalior", "lat": 26.2933, "lng": 78.2278},
    {"iata": "BHO", "name": "Raja Bhoj Airport", "city": "Bhopal", "lat": 23.2875, "lng": 77.3374},
    {"iata": "JBP", "name": "Jabalpur Airport", "city": "Jabalpur", "lat": 23.1778, "lng": 80.0520},
    {"iata": "STV", "name": "Surat Airport", "city": "Surat", "lat": 21.1141, "lng": 72.7418},
    {"iata": "BDQ", "name": "Vadodara Airport", "city": "Vadodara", "lat": 22.3362, "lng": 73.2263},
    {"iata": "BHJ", "name": "Bhuj Airport", "city": "Bhuj", "lat": 23.2878, "lng": 69.6702},
    {"iata": "DIU", "name": "Diu Airport", "city": "Diu", "lat": 20.7131, "lng": 70.9211},
    {"iata": "PBD", "name": "Porbandar Airport", "city": "Porbandar", "lat": 21.6487, "lng": 69.6572},
    {"iata": "IXE", "name": "Mangalore International", "city": "Mangalore", "lat": 12.9614, "lng": 74.8901},
    {"iata": "HBX", "name": "Hubli Airport", "city": "Hubli", "lat": 15.3617, "lng": 75.0849},
    {"iata": "IXG", "name": "Belgaum Airport", "city": "Belgaum", "lat": 15.8593, "lng": 74.6183},
    {"iata": "MYQ", "name": "Mysore Airport", "city": "Mysore", "lat": 12.2300, "lng": 76.6556},
    {"iata": "IXZ", "name": "Vir Savarkar International", "city": "Port Blair", "lat": 11.6412, "lng": 92.7297},
    {"iata": "CNN", "name": "Kannur International", "city": "Kannur", "lat": 11.9190, "lng": 75.5474},
    {"iata": "TCR", "name": "Tuticorin Airport", "city": "Tuticorin", "lat": 8.7244, "lng": 78.0258},
    {"iata": "SLV", "name": "Salem Airport", "city": "Salem", "lat": 11.7834, "lng": 78.0656},
    {"iata": "PNY", "name": "Puducherry Airport", "city": "Puducherry", "lat": 11.9680, "lng": 79.8100},
    {"iata": "IXU", "name": "Chikkalthana Airport", "city": "Aurangabad", "lat": 19.8627, "lng": 75.3981},
    {"iata": "KLH", "name": "Kolhapur Airport", "city": "Kolhapur", "lat": 16.6647, "lng": 74.2894},
    {"iata": "NDC", "name": "Nanded Airport", "city": "Nanded", "lat": 19.1833, "lng": 77.3167},
    {"iata": "ISK", "name": "Nashik Airport", "city": "Nashik", "lat": 20.1191, "lng": 73.9129},
    {"iata": "SAG", "name": "Shirdi Airport", "city": "Shirdi", "lat": 19.6886, "lng": 74.3789},
]


def get_airports_for_region(region: str) -> list[dict]:
    """Return reference airport markers for the selected region."""
    if region == "india":
        return INDIAN_AIRPORTS
    return _REGION_AIRPORTS.get(region, [])


def get_global_airport_lookup() -> dict[str, dict]:
    """Return a combined IATA→airport dict covering Indian + all regional hubs."""
    combined: dict[str, dict] = {}
    for ap in INDIAN_AIRPORTS:
        combined[ap["iata"]] = ap
    for airports in _REGION_AIRPORTS.values():
        for ap in airports:
            if ap["iata"] not in combined:
                combined[ap["iata"]] = ap
    return combined


# ── Hub airports for non-India regions ────────────────────────────────────────
_REGION_AIRPORTS: dict[str, list[dict]] = {
    "europe": [
        {"iata": "LHR", "name": "Heathrow",               "city": "London",        "lat": 51.4775, "lng": -0.4614},
        {"iata": "CDG", "name": "Charles de Gaulle",       "city": "Paris",         "lat": 49.0097, "lng":  2.5478},
        {"iata": "AMS", "name": "Schiphol",                "city": "Amsterdam",     "lat": 52.3086, "lng":  4.7639},
        {"iata": "FRA", "name": "Frankfurt Airport",       "city": "Frankfurt",     "lat": 50.0333, "lng":  8.5706},
        {"iata": "MAD", "name": "Adolfo Suárez",           "city": "Madrid",        "lat": 40.4719, "lng": -3.5626},
        {"iata": "BCN", "name": "El Prat",                 "city": "Barcelona",     "lat": 41.2971, "lng":  2.0785},
        {"iata": "MUC", "name": "Munich Airport",          "city": "Munich",        "lat": 48.3537, "lng": 11.7750},
        {"iata": "FCO", "name": "Leonardo da Vinci",       "city": "Rome",          "lat": 41.8003, "lng": 12.2389},
        {"iata": "IST", "name": "Istanbul Airport",        "city": "Istanbul",      "lat": 41.2608, "lng": 28.7418},
        {"iata": "ZRH", "name": "Zurich Airport",          "city": "Zurich",        "lat": 47.4647, "lng":  8.5492},
        {"iata": "VIE", "name": "Vienna International",    "city": "Vienna",        "lat": 48.1103, "lng": 16.5697},
        {"iata": "CPH", "name": "Copenhagen Airport",      "city": "Copenhagen",    "lat": 55.6180, "lng": 12.6561},
        {"iata": "OSL", "name": "Oslo Gardermoen",         "city": "Oslo",          "lat": 60.1939, "lng": 11.1004},
        {"iata": "ARN", "name": "Stockholm Arlanda",       "city": "Stockholm",     "lat": 59.6519, "lng": 17.9186},
        {"iata": "HEL", "name": "Helsinki-Vantaa",         "city": "Helsinki",      "lat": 60.3172, "lng": 24.9633},
    ],
    "north_america": [
        {"iata": "ATL", "name": "Hartsfield-Jackson",      "city": "Atlanta",       "lat": 33.6407, "lng": -84.4277},
        {"iata": "ORD", "name": "O'Hare International",    "city": "Chicago",       "lat": 41.9742, "lng": -87.9073},
        {"iata": "LAX", "name": "Los Angeles International","city": "Los Angeles",  "lat": 33.9425, "lng":-118.4081},
        {"iata": "DFW", "name": "Dallas/Fort Worth",       "city": "Dallas",        "lat": 32.8998, "lng": -97.0403},
        {"iata": "DEN", "name": "Denver International",    "city": "Denver",        "lat": 39.8561, "lng":-104.6737},
        {"iata": "JFK", "name": "John F. Kennedy",         "city": "New York",      "lat": 40.6413, "lng": -73.7781},
        {"iata": "SFO", "name": "San Francisco International","city": "San Francisco","lat": 37.6213, "lng":-122.3790},
        {"iata": "SEA", "name": "Seattle-Tacoma",          "city": "Seattle",       "lat": 47.4502, "lng":-122.3088},
        {"iata": "YYZ", "name": "Toronto Pearson",         "city": "Toronto",       "lat": 43.6777, "lng": -79.6248},
        {"iata": "YVR", "name": "Vancouver International", "city": "Vancouver",     "lat": 49.1947, "lng":-123.1792},
        {"iata": "MEX", "name": "Benito Juárez International","city": "Mexico City","lat": 19.4363, "lng": -99.0721},
        {"iata": "MIA", "name": "Miami International",     "city": "Miami",         "lat": 25.7959, "lng": -80.2870},
    ],
    "middle_east": [
        {"iata": "DXB", "name": "Dubai International",     "city": "Dubai",         "lat": 25.2532, "lng": 55.3657},
        {"iata": "AUH", "name": "Abu Dhabi International", "city": "Abu Dhabi",     "lat": 24.4330, "lng": 54.6511},
        {"iata": "DOH", "name": "Hamad International",     "city": "Doha",          "lat": 25.2732, "lng": 51.6082},
        {"iata": "RUH", "name": "King Khalid International","city": "Riyadh",       "lat": 24.9576, "lng": 46.6988},
        {"iata": "JED", "name": "King Abdulaziz International","city": "Jeddah",    "lat": 21.6796, "lng": 39.1565},
        {"iata": "KWI", "name": "Kuwait International",    "city": "Kuwait City",   "lat": 29.2267, "lng": 47.9689},
        {"iata": "BAH", "name": "Bahrain International",   "city": "Manama",        "lat": 26.2708, "lng": 50.6336},
        {"iata": "MCT", "name": "Muscat International",    "city": "Muscat",        "lat": 23.5933, "lng": 58.2844},
        {"iata": "AMM", "name": "Queen Alia International","city": "Amman",         "lat": 31.7226, "lng": 35.9932},
        {"iata": "BEY", "name": "Rafic Hariri International","city": "Beirut",      "lat": 33.8209, "lng": 35.4883},
    ],
    "south_asia": [
        {"iata": "DEL", "name": "Indira Gandhi International","city": "Delhi",      "lat": 28.5562, "lng": 77.1000},
        {"iata": "BOM", "name": "Chhatrapati Shivaji Maharaj International","city": "Mumbai","lat": 19.0896, "lng": 72.8656},
        {"iata": "KHI", "name": "Jinnah International",    "city": "Karachi",       "lat": 24.9065, "lng": 67.1608},
        {"iata": "LHE", "name": "Allama Iqbal International","city": "Lahore",      "lat": 31.5216, "lng": 74.4036},
        {"iata": "ISB", "name": "Islamabad International", "city": "Islamabad",     "lat": 33.5490, "lng": 72.8574},
        {"iata": "DAC", "name": "Hazrat Shahjalal International","city": "Dhaka",   "lat": 23.8433, "lng": 90.3978},
        {"iata": "CMB", "name": "Bandaranaike International","city": "Colombo",     "lat":  7.1807, "lng": 79.8841},
        {"iata": "KTM", "name": "Tribhuvan International", "city": "Kathmandu",     "lat": 27.6966, "lng": 85.3591},
        {"iata": "MLE", "name": "Velana International",    "city": "Malé",          "lat":  4.1918, "lng": 73.5290},
    ],
    "gulf": [
        {"iata": "DXB", "name": "Dubai International",     "city": "Dubai",         "lat": 25.2532, "lng": 55.3657},
        {"iata": "AUH", "name": "Abu Dhabi International", "city": "Abu Dhabi",     "lat": 24.4330, "lng": 54.6511},
        {"iata": "DOH", "name": "Hamad International",     "city": "Doha",          "lat": 25.2732, "lng": 51.6082},
        {"iata": "SHJ", "name": "Sharjah International",   "city": "Sharjah",       "lat": 25.3286, "lng": 55.5172},
        {"iata": "RKT", "name": "Ras Al Khaimah International","city": "Ras Al Khaimah","lat": 25.6135, "lng": 55.9388},
        {"iata": "BAH", "name": "Bahrain International",   "city": "Manama",        "lat": 26.2708, "lng": 50.6336},
        {"iata": "KWI", "name": "Kuwait International",    "city": "Kuwait City",   "lat": 29.2267, "lng": 47.9689},
        {"iata": "MCT", "name": "Muscat International",    "city": "Muscat",        "lat": 23.5933, "lng": 58.2844},
    ],
    "world": [
        {"iata": "LHR", "name": "Heathrow",               "city": "London",        "lat": 51.4775, "lng": -0.4614},
        {"iata": "CDG", "name": "Charles de Gaulle",       "city": "Paris",         "lat": 49.0097, "lng":  2.5478},
        {"iata": "FRA", "name": "Frankfurt Airport",       "city": "Frankfurt",     "lat": 50.0333, "lng":  8.5706},
        {"iata": "DXB", "name": "Dubai International",     "city": "Dubai",         "lat": 25.2532, "lng": 55.3657},
        {"iata": "ATL", "name": "Hartsfield-Jackson",      "city": "Atlanta",       "lat": 33.6407, "lng": -84.4277},
        {"iata": "ORD", "name": "O'Hare International",    "city": "Chicago",       "lat": 41.9742, "lng": -87.9073},
        {"iata": "LAX", "name": "Los Angeles International","city": "Los Angeles",  "lat": 33.9425, "lng":-118.4081},
        {"iata": "JFK", "name": "John F. Kennedy",         "city": "New York",      "lat": 40.6413, "lng": -73.7781},
        {"iata": "PEK", "name": "Beijing Capital",         "city": "Beijing",       "lat": 40.0799, "lng": 116.6031},
        {"iata": "HND", "name": "Tokyo Haneda",            "city": "Tokyo",         "lat": 35.5533, "lng": 139.7811},
        {"iata": "SIN", "name": "Singapore Changi",        "city": "Singapore",     "lat":  1.3644, "lng": 103.9915},
        {"iata": "SYD", "name": "Sydney Airport",          "city": "Sydney",        "lat": -33.9461, "lng": 151.1772},
        {"iata": "GRU", "name": "São Paulo Guarulhos",     "city": "São Paulo",     "lat": -23.4356, "lng": -46.4731},
        {"iata": "JNB", "name": "O.R. Tambo International","city": "Johannesburg",  "lat": -26.1392, "lng":  28.2460},
        {"iata": "DEL", "name": "Indira Gandhi International","city": "Delhi",      "lat": 28.5562, "lng": 77.1000},
        {"iata": "BOM", "name": "Chhatrapati Shivaji Maharaj International","city": "Mumbai","lat": 19.0896, "lng": 72.8656},
        {"iata": "YYZ", "name": "Toronto Pearson",         "city": "Toronto",       "lat": 43.6777, "lng": -79.6248},
        {"iata": "MEX", "name": "Benito Juárez International","city": "Mexico City","lat": 19.4363, "lng": -99.0721},
        # Southeast Asia (high traffic from India)
        {"iata": "BKK", "name": "Suvarnabhumi Airport",   "city": "Bangkok",       "lat": 13.6811, "lng": 100.7472},
        {"iata": "DMK", "name": "Don Mueang International","city": "Bangkok",       "lat": 13.9126, "lng": 100.6067},
        {"iata": "HKT", "name": "Phuket International",   "city": "Phuket",        "lat":  8.1132, "lng": 98.3169},
        {"iata": "KUL", "name": "Kuala Lumpur International","city": "Kuala Lumpur","lat":  2.7456, "lng": 101.7099},
        {"iata": "HAN", "name": "Noi Bai International",  "city": "Hanoi",         "lat": 21.2212, "lng": 105.8072},
        {"iata": "SGN", "name": "Tan Son Nhat International","city": "Ho Chi Minh City","lat": 10.8188, "lng": 106.6520},
        {"iata": "HKG", "name": "Hong Kong International","city": "Hong Kong",     "lat": 22.3080, "lng": 113.9185},
        {"iata": "PQC", "name": "Phu Quoc International", "city": "Phu Quoc",      "lat": 10.2270, "lng": 103.9670},
        {"iata": "CXR", "name": "Cam Ranh International", "city": "Nha Trang",     "lat": 11.9982, "lng": 109.2192},
        # Europe extras
        {"iata": "BRU", "name": "Brussels Airport",       "city": "Brussels",      "lat": 50.9014, "lng":  4.4844},
        {"iata": "MXP", "name": "Milan Malpensa",         "city": "Milan",         "lat": 45.6306, "lng":  8.7281},
        # Central Asia / Africa
        {"iata": "ALA", "name": "Almaty International",   "city": "Almaty",        "lat": 43.3521, "lng": 77.0405},
        {"iata": "ADD", "name": "Addis Ababa Bole",       "city": "Addis Ababa",   "lat":  8.9779, "lng": 38.7993},
        {"iata": "DMM", "name": "King Fahd International","city": "Dammam",        "lat": 26.4712, "lng": 49.7979},
        # Indian airports sometimes spelled differently
        {"iata": "CCJ", "name": "Calicut International",  "city": "Kozhikode",     "lat": 11.1368, "lng": 75.9553},
    ],
}


# ── AirLabs response columns we care about ────────────────────────────────────
# AirLabs returns a JSON list of flight objects. Each object is a dict
# with named keys — much nicer than OpenSky's unnamed index-based list.
KEEP_COLUMNS = [
    "hex",            # ICAO24 transponder code (unique aircraft ID)
    "reg_number",     # Aircraft registration e.g. "A6-ENY"
    "flag",           # Country flag of origin
    "lat",            # Latitude (decimal degrees)
    "lng",            # Longitude (decimal degrees)
    "alt",            # Altitude in metres
    "dir",            # Heading in degrees (0-360)
    "speed",          # Ground speed in km/h
    "v_speed",        # Vertical speed in m/s
    "squawk",         # Transponder squawk code
    "flight_number",  # Full flight number e.g. "EK203"
    "flight_icao",    # ICAO flight number e.g. "UAE203"
    "flight_iata",    # IATA flight number e.g. "EK203"
    "dep_icao",       # Departure airport ICAO code e.g. "OMDB"
    "dep_iata",       # Departure airport IATA code e.g. "DXB"
    "arr_icao",       # Arrival airport ICAO code e.g. "EGLL"
    "arr_iata",       # Arrival airport IATA code e.g. "LHR"
    "airline_icao",   # Airline ICAO code e.g. "UAE"
    "airline_iata",   # Airline IATA code e.g. "EK"
    "aircraft_icao",  # Aircraft type e.g. "B77W" (Boeing 777)
    "updated",        # Unix timestamp of last update
    "status",         # Flight status: "en-route", "landed", etc.
]


# ── Main fetch function ────────────────────────────────────────────────────────
def get_flight_data(
    region: str = "india",
    airline_iata: str = None,
    dep_iata: str = None,
    arr_iata: str = None,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch live flight data from AirLabs API.

    Unlike OpenSky, AirLabs returns airline names, departure airports,
    arrival airports, and flight numbers with every flight.

    Parameters
    ----------
    region : str
        Bounding box region. One of the keys in REGIONS dict above.
    airline_iata : str, optional
        Filter by airline IATA code. e.g. "EK" for Emirates only.
    dep_iata : str, optional
        Filter by departure airport. e.g. "DXB" for Dubai departures.
    arr_iata : str, optional
        Filter by arrival airport. e.g. "LHR" for London arrivals.
    max_retries : int
        Number of retry attempts on failure.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with named columns, unit conversions applied.

    Examples
    --------
    # All flights over India
    df = get_flight_data(region="india")

    # Only Air India flights over India
    df = get_flight_data(region="india", airline_iata="AI")

    # Only flights departing Delhi
    df = get_flight_data(region="world", dep_iata="DEL")
    """

    # ── Read API key securely from environment ─────────────────────────────────
    api_key = os.getenv("AIRLABS_API_KEY")

    if not api_key:
        logger.error(
            "AIRLABS_API_KEY not found! "
            "Create a .env file with: AIRLABS_API_KEY=your_key_here"
        )
        return pd.DataFrame()

    # ── Build query parameters ─────────────────────────────────────────────────
    # params is a dictionary. requests.get() automatically converts it
    # into a query string: ?api_key=...&bbox=...&airline_iata=...
    params = {"api_key": api_key}

    bbox = REGIONS.get(region)
    if bbox:
        params["bbox"] = bbox
        logger.info(f"Region: '{region}' | bbox: {bbox}")
    else:
        logger.info("Fetching global flights (no bounding box filter)")

    if airline_iata:
        params["airline_iata"] = airline_iata.upper()
        logger.info(f"Airline filter: {airline_iata.upper()}")

    if dep_iata:
        params["dep_iata"] = dep_iata.upper()
        logger.info(f"Departure airport filter: {dep_iata.upper()}")

    if arr_iata:
        params["arr_iata"] = arr_iata.upper()
        logger.info(f"Arrival airport filter: {arr_iata.upper()}")

    # ── Retry loop ─────────────────────────────────────────────────────────────
    _cache_key = (region, airline_iata or "", dep_iata or "", arr_iata or "")
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"API request attempt {attempt}/{max_retries}...")

            response = requests.get(
                f"{BASE_URL}/flights",
                params=params,
                timeout=15
            )
            response.raise_for_status()

            data = response.json()

            # AirLabs wraps results in a "response" key:
            # {"response": [...list of flight dicts...]}
            
            # Check for API-level errors (e.g. quota exceeded)
            if "error" in data:
                err_msg = data["error"].get("message", "Unknown API error")
                err_code = data["error"].get("code", "")
                logger.error(f"AirLabs API error: {err_msg} (code: {err_code})")
                _quota_keywords = ("limit", "quota", "exceeded", "exhausted", "requests")
                if any(kw in err_msg.lower() or kw in err_code.lower() for kw in _quota_keywords):
                    send_alert(
                        subject="AirLabs API quota exhausted",
                        body=(
                            f"Your AirLabs request quota has been exhausted.\n\n"
                            f"Message : {err_msg}\n"
                            f"Code    : {err_code}\n\n"
                            "Action  : Log in to https://airlabs.co/ and renew\n"
                            "          your plan or wait for the quota to reset,\n"
                            "          then update AIRLABS_API_KEY in your .env file."
                        ),
                        event_key="airlabs_quota_exhausted",
                    )
                else:
                    send_alert(
                        subject=f"AirLabs API error — {err_code or 'unknown'}",
                        body=(
                            f"The AirLabs API returned an error.\n\n"
                            f"Message : {err_msg}\n"
                            f"Code    : {err_code}\n\n"
                            "Check https://airlabs.co/ for your account status."
                        ),
                        event_key=f"airlabs_api_error_{err_code}",
                    )
                return _fetch_opensky_fallback(region)

            flights = data.get("response", [])

            if not flights:
                logger.warning("No flights returned. Region may be empty or filters too strict.")
                return pd.DataFrame(columns=KEEP_COLUMNS)

            logger.info(f"Raw data received: {len(flights)} flights")

            df = _parse_flights(flights)
            logger.info(f"After cleaning: {len(df)} valid flights")
            _FLIGHT_CACHE[_cache_key] = (time.time(), df.copy())
            # Cap cache size — evict oldest entry when limit is reached
            if len(_FLIGHT_CACHE) > _MAX_FLIGHT_CACHE_ENTRIES:
                oldest_key = min(_FLIGHT_CACHE, key=lambda k: _FLIGHT_CACHE[k][0])
                _FLIGHT_CACHE.pop(oldest_key, None)
            return df

        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out on attempt {attempt}.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                logger.warning("Rate limited (429) — returning stale cache if available.")
                send_alert(
                    subject="AirLabs API rate limited (429)",
                    body=(
                        "The AirLabs API returned HTTP 429 Too Many Requests.\n\n"
                        "This usually means you've hit the per-minute or per-day\n"
                        "request cap for your current plan.\n\n"
                        "Action  : Log in to https://airlabs.co/ to check your\n"
                        "          usage and consider upgrading your plan."
                    ),
                    event_key="airlabs_429",
                )
                _stale = _FLIGHT_CACHE.get(_cache_key)
                if _stale is not None:
                    logger.info("Returning %d stale flights from cache (%.0fs old).",
                                len(_stale[1]), time.time() - _stale[0])
                    return _stale[1].copy()
                return pd.DataFrame(columns=KEEP_COLUMNS)
            elif status == 401:
                logger.error("Invalid API key. Check your .env file.")
                send_alert(
                    subject="AirLabs API key invalid (401)",
                    body=(
                        "The AirLabs API rejected your key with HTTP 401 Unauthorized.\n\n"
                        "Action  : Open your .env file and update AIRLABS_API_KEY\n"
                        "          with a valid key from https://airlabs.co/"
                    ),
                    event_key="airlabs_401",
                )
                return pd.DataFrame()
        except requests.exceptions.ConnectionError:
            logger.error("No internet connection or AirLabs is unreachable.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        if attempt < max_retries:
            wait = min(2 ** attempt * 5, 60)
            logger.info(f"Waiting {wait}s before retry...")
            time.sleep(wait)

    logger.error("All retry attempts failed. Trying OpenSky fallback...")
    fallback_df = _fetch_opensky_fallback(region)
    if not fallback_df.empty:
        return fallback_df
    return pd.DataFrame(columns=KEEP_COLUMNS)


# ── OpenSky live REST fallback ─────────────────────────────────────────────────
# Used automatically when AirLabs returns empty (quota exhausted / error).
# OpenSky is free & unauthenticated for ~400 req/day; credentials from .env
# raise that limit significantly.
_OPENSKY_BBOX = {
    "india":  (8.0, 68.0, 37.0, 97.0),   # lamin, lomin, lamax, lomax
    "world":  None,
    "europe": (36.0, -15.0, 72.0, 40.0),
    "south_asia": (5.0, 60.0, 40.0, 95.0),
}

def _fetch_opensky_fallback(region: str = "india") -> pd.DataFrame:
    """
    Fetch live state vectors from OpenSky REST API.
    Returns a DataFrame with the same columns as AirLabs output so the
    rest of the app works without modification.
    """
    bbox = _OPENSKY_BBOX.get(region, _OPENSKY_BBOX["india"])
    url = "https://opensky-network.org/api/states/all"
    params: dict = {}
    if bbox:
        params = {"lamin": bbox[0], "lomin": bbox[1], "lamax": bbox[2], "lomax": bbox[3]}
    auth = None
    usr = os.getenv("OPENSKY_USERNAME")
    pwd = os.getenv("OPENSKY_PASSWORD")
    if usr and pwd:
        auth = (usr, pwd)
    try:
        resp = requests.get(url, params=params, auth=auth, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        states = data.get("states") or []
        if not states:
            return pd.DataFrame(columns=KEEP_COLUMNS)
        rows = []
        for s in states:
            # OpenSky state vector fields (indices):
            # 0=icao24, 1=callsign, 2=origin_country, 3=time_position,
            # 4=last_contact, 5=longitude, 6=latitude, 7=baro_altitude,
            # 8=on_ground, 9=velocity, 10=true_track, 11=vertical_rate,
            # 12=sensors, 13=geo_altitude, 14=squawk, 15=spi, 16=position_source
            try:
                callsign = (s[1] or "").strip()
                if not callsign:
                    continue
                rows.append({
                    "icao24":       s[0] or "",
                    "flight_iata":  callsign,
                    "flight_icao":  callsign,
                    "flight":       callsign,
                    "airline_iata": callsign[:2] if len(callsign) >= 2 else "",
                    "airline_icao": callsign[:3] if len(callsign) >= 3 else "",
                    "airline_code": callsign[:2] if len(callsign) >= 2 else "",
                    "airline":      callsign[:2] if len(callsign) >= 2 else "Unknown",
                    "lat":          float(s[6]) if s[6] is not None else None,
                    "lng":          float(s[5]) if s[5] is not None else None,
                    "altitude_ft":  round(float(s[7]) * 3.28084) if s[7] else None,
                    "speed_kts":    round(float(s[9]) * 1.94384) if s[9] else None,
                    "heading":      float(s[10]) if s[10] is not None else None,
                    "vertical_speed": float(s[11]) if s[11] is not None else None,
                    "on_ground":    bool(s[8]),
                    "dep":          "",
                    "arr":          "",
                    "dep_iata":     "",
                    "arr_iata":     "",
                    "aircraft_icao": "",
                    "aircraft":     "Unknown",
                    "reg":          "",
                    "status":       "en-route",
                    "delayed":      0,
                    "delayed_min":  0,
                    "eta":          None,
                    "family":       "narrow",
                    "airline_name": callsign[:2] if len(callsign) >= 2 else "Unknown",
                })
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(columns=KEEP_COLUMNS)
        df = pd.DataFrame(rows)
        # Drop aircraft on ground
        df = df[df["on_ground"] == False].copy()  # noqa: E712
        df.drop(columns=["on_ground"], inplace=True, errors="ignore")
        logger.info("OpenSky fallback: %d airborne flights", len(df))
        return df
    except Exception as exc:
        logger.error("OpenSky fallback failed: %s", exc)
        return pd.DataFrame(columns=KEEP_COLUMNS)


def get_airport_schedules(
    airport_iata: str,
    direction: str = "arrival",
    max_retries: int = 2,
) -> pd.DataFrame:
    """
    Fetch airport schedules for a single airport and direction.

    Parameters
    ----------
    airport_iata : str
        Airport IATA code.
    direction : str
        Either "arrival" or "departure".
    """
    airport_iata = str(airport_iata or "").upper().strip()
    if not airport_iata or airport_iata == "N/A":
        return pd.DataFrame()

    direction = "arrival" if direction != "departure" else "departure"
    cache_key = (airport_iata, direction)
    now = time.time()
    cached = _SCHEDULE_CACHE.get(cache_key)
    if cached and now - cached[0] < SCHEDULE_CACHE_TTL:
        return cached[1].copy()

    api_key = os.getenv("AIRLABS_API_KEY")
    if not api_key:
        logger.error("AIRLABS_API_KEY not found for schedule request.")
        return pd.DataFrame()

    params = {
        "api_key": api_key,
        "limit": 200,
    }
    if direction == "arrival":
        params["arr_iata"] = airport_iata
    else:
        params["dep_iata"] = airport_iata

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(f"{BASE_URL}/schedules", params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("response", [])
            df = pd.DataFrame(data)
            if not df.empty:
                for column in ["dep_time_ts", "dep_estimated_ts", "arr_time_ts", "arr_estimated_ts", "delayed"]:
                    if column in df.columns:
                        df[column] = pd.to_numeric(df[column], errors="coerce")
            _SCHEDULE_CACHE[cache_key] = (time.time(), df.copy())
            # Cap cache size to prevent unbounded growth
            if len(_SCHEDULE_CACHE) > _MAX_CACHE_ENTRIES:
                oldest_key = min(_SCHEDULE_CACHE, key=lambda k: _SCHEDULE_CACHE[k][0])
                _SCHEDULE_CACHE.pop(oldest_key, None)
            return df
        except requests.exceptions.HTTPError as exc:
            logger.warning("Schedule request failed for %s (%s): %s", airport_iata, direction, exc)
            status = exc.response.status_code if exc.response is not None else 0
            if status in {401, 403, 404}:
                break
        except Exception as exc:
            logger.warning("Unexpected schedule error for %s (%s): %s", airport_iata, direction, exc)
        if attempt < max_retries:
            time.sleep(min(2 ** attempt * 2, 30))

    return pd.DataFrame()


# ── Internal parser ────────────────────────────────────────────────────────────
def _parse_flights(flights: list) -> pd.DataFrame:
    """
    Convert raw AirLabs flight list into a clean, typed DataFrame.

    Parameters
    ----------
    flights : list
        Raw list of flight dicts from AirLabs API response.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame ready for mapping and analysis.
    """

    df = pd.DataFrame(flights)

    # Keep only defined columns, add missing ones as NaN
    df = df.reindex(columns=KEEP_COLUMNS)

    # Drop rows with no position — can't place on map
    df.dropna(subset=["lat", "lng"], inplace=True)

    # ── Rename for clarity ─────────────────────────────────────────────────────
    df.rename(columns={
        "lat":   "latitude",
        "lng":   "longitude",
        "alt":   "altitude_m",
        "dir":   "heading",
        "speed": "speed_kmh",
    }, inplace=True)

    # ── Unit conversions ───────────────────────────────────────────────────────
    # Altitude: metres → feet (1m = 3.28084 ft)
    df["altitude_ft"] = (df["altitude_m"] * 3.28084).round(0)

    # Speed: km/h → knots (1 km/h = 0.539957 knots)
    df["speed_kts"] = (df["speed_kmh"] * 0.539957).round(1)

    # ── Fill missing text fields ───────────────────────────────────────────────
    df["flight_iata"]   = df["flight_iata"].fillna("N/A")
    df["airline_iata"]  = df["airline_iata"].fillna("N/A")
    df["dep_iata"]      = df["dep_iata"].fillna("N/A")
    df["arr_iata"]      = df["arr_iata"].fillna("N/A")
    df["status"]        = df["status"].fillna("unknown")
    df["aircraft_icao"] = df["aircraft_icao"].fillna("N/A")

    df.reset_index(drop=True, inplace=True)

    return df


# ── Known airline IATA codes ───────────────────────────────────────────────────
KNOWN_AIRLINES = {
    # ── India ───────────────────────────────────────────────────────────────
    "AI": "Air India",
    "IX": "Air India Express",
    "6E": "IndiGo",
    "SG": "SpiceJet",
    "UK": "Vistara",
    "G8": "Go First",
    "QP": "Akasa Air",
    "I5": "AirAsia India",
    "S5": "Star Air",
    "9I": "Alliance Air",
    "2T": "TruJet",
    "CD": "IndiGo (charter)",
    # ── Gulf / Middle East ────────────────────────────────────────────
    "EK": "Emirates",
    "EY": "Etihad Airways",
    "QR": "Qatar Airways",
    "FZ": "flydubai",
    "G9": "Air Arabia",
    "WY": "Oman Air",
    "GF": "Gulf Air",
    "SV": "Saudia",
    "XY": "flynas",
    "UL": "SriLankan Airlines",
    "RJ": "Royal Jordanian",
    "MS": "EgyptAir",
    "IY": "Yemenia",
    "IR": "Iran Air",
    # ── Europe ────────────────────────────────────────────────────────
    "BA": "British Airways",
    "LH": "Lufthansa",
    "AF": "Air France",
    "KL": "KLM",
    "TK": "Turkish Airlines",
    "AZ": "ITA Airways",
    "IB": "Iberia",
    "SK": "Scandinavian Airlines",
    "AY": "Finnair",
    "OS": "Austrian Airlines",
    "LX": "Swiss International Air Lines",
    "SN": "Brussels Airlines",
    "U2": "easyJet",
    "FR": "Ryanair",
    "VY": "Vueling",
    "HV": "Transavia",
    # ── Asia / Pacific ─────────────────────────────────────────────────
    "SQ": "Singapore Airlines",
    "MI": "SilkAir",
    "TR": "Scoot",
    "MH": "Malaysia Airlines",
    "AK": "AirAsia",
    "TG": "Thai Airways",
    "FD": "Thai AirAsia",
    "CX": "Cathay Pacific",
    "KA": "Cathay Dragon",
    "NH": "All Nippon Airways",
    "JL": "Japan Airlines",
    "OZ": "Asiana Airlines",
    "KE": "Korean Air",
    "CI": "China Airlines",
    "CZ": "China Southern",
    "CA": "Air China",
    "MU": "China Eastern",
    "VN": "Vietnam Airlines",
    "BX": "Air Busan",
    "GA": "Garuda Indonesia",
    "QZ": "Indonesia AirAsia",
    # ── North America ──────────────────────────────────────────────────
    "AA": "American Airlines",
    "UA": "United Airlines",
    "DL": "Delta Air Lines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "F9": "Frontier Airlines",
    "NK": "Spirit Airlines",
    "AC": "Air Canada",
    "WS": "WestJet",
    "AM": "Aeroméxico",
    # ── Africa / Other ────────────────────────────────────────────────
    "ET": "Ethiopian Airlines",
    "KQ": "Kenya Airways",
    "SA": "South African Airways",
    "PK": "Pakistan International Airlines",
    "BG": "Biman Bangladesh Airlines",
}

def get_airline_name(iata_code: str) -> str:
    """Return full airline name from IATA code, or the code itself if unknown."""
    return KNOWN_AIRLINES.get(iata_code.upper(), iata_code)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nTesting AirLabs API connection...")
    df = get_flight_data(region="india")

    if not df.empty:
        print(f"\nTotal flights: {len(df)}")
        print(f"\nSample flights:")
        print(df[[
            "flight_iata", "airline_iata", "dep_iata",
            "arr_iata", "altitude_ft", "speed_kts", "status"
        ]].head(5).to_string(index=False))
    else:
        print("No data returned. Check your API key in .env file.")
