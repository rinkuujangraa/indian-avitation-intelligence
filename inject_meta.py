"""
inject_meta.py
--------------
Patches Streamlit's bundled index.html to inject proper OG/Twitter/SEO
meta tags BEFORE the closing </head> tag, so LinkedIn, Twitter, and
Google crawlers see them in the static HTML — not just after JS runs.

Run this as a build step (see railway.toml) so it applies once at deploy
time, not on every request.
"""
import os
import sys
import streamlit

INDEX_PATH = os.path.join(
    os.path.dirname(streamlit.__file__), "static", "index.html"
)

META_BLOCK = """
    <!-- ── SEO / Social meta ── injected by inject_meta.py at build time ── -->
    <title>Aviation Intelligence · India Live Flight Tracker</title>
    <meta name="description" content="Real-time India flight tracking, ML delay prediction, airport congestion analytics and anomaly detection. Built with Python, Streamlit, XGBoost and Mapbox." />

    <!-- Open Graph -->
    <meta property="og:type"        content="website" />
    <meta property="og:site_name"   content="Aviation Intelligence Platform" />
    <meta property="og:title"       content="Aviation Intelligence · India Live Flight Tracker" />
    <meta property="og:description" content="Real-time India flight tracking, ML delay prediction, airport congestion analytics and anomaly detection." />
    <meta property="og:image"       content="https://web-production-39c38.up.railway.app/app/static/og.png" />
    <meta property="og:image:width"  content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:image:alt"   content="Aviation Intelligence Platform — Live Airspace Analytics" />
    <meta property="og:url"         content="https://web-production-39c38.up.railway.app" />

    <!-- Twitter / X card -->
    <meta name="twitter:card"        content="summary_large_image" />
    <meta name="twitter:title"       content="Aviation Intelligence · India Live Flight Tracker" />
    <meta name="twitter:description" content="Real-time India flight tracking, ML delay prediction and airport analytics." />
    <meta name="twitter:image"       content="https://web-production-39c38.up.railway.app/app/static/og.png" />

    <!-- Favicon — inline SVG data URI (gold A-mark) -->
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg width='32' height='32' viewBox='24 30 300 300' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3ClinearGradient id='fa' x1='32' y1='40' x2='260' y2='300' gradientUnits='userSpaceOnUse'%3E%3Cstop stop-color='%23FFC14D'/%3E%3Cstop offset='1' stop-color='%23FF9F1A'/%3E%3C/linearGradient%3E%3ClinearGradient id='fd' x1='0' y1='0' x2='320' y2='320' gradientUnits='userSpaceOnUse'%3E%3Cstop stop-color='%23121A26'/%3E%3Cstop offset='1' stop-color='%230A1018'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect x='24' y='30' width='300' height='300' rx='72' fill='url(%23fd)'/%3E%3Cpath d='M112 246L176 88H228L292 246H239L224 207H180L166 246H112ZM194 163H210L202 140L194 163Z' fill='url(%23fa)'/%3E%3Cpath d='M94 212C120 167 163 136 219 120C243 113 267 110 291 110' stroke='%235CD2FF' stroke-width='10' stroke-linecap='round' stroke-dasharray='2 18'/%3E%3Cpath d='M274 98L309 111L283 139' stroke='%235CD2FF' stroke-width='10' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E" />
    <!-- ── end SEO block ── -->
"""

def patch():
    if not os.path.exists(INDEX_PATH):
        print(f"[inject_meta] ERROR: index.html not found at {INDEX_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    if "og:title" in html:
        print("[inject_meta] index.html already patched — skipping.")
        return

    # Replace the bare <title>Streamlit</title> and inject full meta block
    patched = html.replace(
        "<title>Streamlit</title>",
        META_BLOCK.strip(),
    )

    if patched == html:
        # Fallback: inject just before </head>
        patched = html.replace("</head>", META_BLOCK + "</head>", 1)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"[inject_meta] Patched {INDEX_PATH}")


if __name__ == "__main__":
    patch()
