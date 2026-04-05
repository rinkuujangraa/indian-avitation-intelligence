"""
aircraft_icons.py
-----------------
Realistic top-down aircraft silhouettes using SVG bezier paths.
Style matches FlightRadar24 — solid filled shapes with proper:
  - Curved fuselage taper (nose to tail)
  - Swept wing geometry with chord variation
  - Correctly sized and positioned engine nacelles
  - Proper horizontal tailplane shapes
  - Winglets where the type has them

All silhouettes:
  - 100×100 viewBox, nose pointing UP (toward y=0)
  - Single fill colour passed as parameter
  - Rotated at render time by aircraft heading
"""


# ── ICAO type code → icon family ───────────────────────────────────────────────
ICAO_TO_FAMILY = {
    # 4-engine heavies
    "B741":"b747","B742":"b747","B743":"b747","B744":"b747","B748":"b747","BLCF":"b747",
    "A388":"a380","A389":"a380",
    "A342":"a340","A343":"a340","A345":"a340","A346":"a340",
    "C17" :"military","C130":"military","A124":"military","A225":"military","C5":"military",

    # Wide-body twins
    "B772":"b777","B773":"b777","B77L":"b777","B77W":"b777","B778":"b777","B779":"b777",
    "A35K":"a350","A358":"a350","A359":"a350",
    "A332":"a330","A333":"a330","A338":"a330","A339":"a330",
    "B787":"b787","B788":"b787","B789":"b787","B78X":"b787",
    "B762":"b767","B763":"b767","B764":"b767","B76X":"b767",

    # Narrow-body twins
    "B752":"b757","B753":"b757",
    "A318":"a320","A319":"a320","A320":"a320","A321":"a320",
    "A20N":"a320","A21N":"a320","A19N":"a320",
    "B731":"b737","B732":"b737","B733":"b737","B734":"b737",
    "B735":"b737","B736":"b737","B737":"b737","B738":"b737",
    "B739":"b737","B37M":"b737","B38M":"b737","B39M":"b737","B3XM":"b737",

    # Regional jets
    "E170":"e190","E175":"e190","E190":"e190","E195":"e190",
    "E75L":"e190","E75S":"e190","E290":"e190","E295":"e190",
    "CRJ1":"crj","CRJ2":"crj","CRJ7":"crj","CRJ9":"crj","CRJX":"crj",

    # Turboprops
    "AT43":"atr","AT44":"atr","AT45":"atr","AT72":"atr",
    "AT73":"atr","AT75":"atr","AT76":"atr",
    "DH8A":"atr","DH8B":"atr","DH8C":"atr","DH8D":"atr",

    # Special
    "CONC":"concorde",
}


def get_family(aircraft_icao: str) -> str:
    if not aircraft_icao or aircraft_icao in ("N/A", ""):
        return "default"
    return ICAO_TO_FAMILY.get(str(aircraft_icao).strip().upper(), "default")


def get_aircraft_icon_html(
    aircraft_icao: str,
    heading: float,
    color: str,
    size: int = 36
) -> str:
    """
    Return complete HTML for a Folium DivIcon —
    a rotated, type-accurate SVG silhouette.
    """
    try:
        rotation = float(heading) if heading and heading == heading else 0
    except Exception:
        rotation = 0

    family = get_family(aircraft_icao)
    svg_body = _get_svg(family, color)

    return f"""<div style="
        width:{size}px;height:{size}px;
        transform:rotate({rotation}deg);
        transform-origin:center center;
        filter:drop-shadow(0 1px 3px rgba(0,0,0,0.45));
    ">
    <svg xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 100 100" width="{size}" height="{size}">
        {svg_body}
    </svg></div>"""


def get_aircraft_svg(aircraft_icao: str, color: str, size: int = 36) -> str:
    family = get_family(aircraft_icao)
    svg_body = _get_svg(family, color)
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 100 100" width="{size}" height="{size}">
        {svg_body}
    </svg>"""


def _get_svg(family: str, c: str) -> str:
    fns = {
        "b747":    _b747,
        "a380":    _a380,
        "a340":    _a340,
        "b777":    _b777,
        "a350":    _a350,
        "a330":    _a330,
        "b787":    _b787,
        "b767":    _b767,
        "b757":    _b757,
        "a320":    _a320,
        "b737":    _b737,
        "e190":    _e190,
        "crj":     _crj,
        "atr":     _atr,
        "concorde":_concorde,
        "military":_military,
        "default": _default,
    }
    return fns.get(family, _default)(c)


# ═══════════════════════════════════════════════════════════════════════════════
#  SILHOUETTE DEFINITIONS  — FR24-style professional top-down silhouettes
#  ViewBox 0 0 100 100. Nose = top (y≈4). Tail = bottom (y≈94). Centre x=50.
#
#  Design language:
#    · stroke="#00121C" stroke-width="2" paint-order="stroke fill"
#      → dark navy outline on every shape for crisp definition on dark maps
#    · Proper swept+tapered wing planform: LE swept more than TE
#    · Nacelles positioned at ~40% chord, slightly forward of wing centre
#    · Fuselage centreline accent: subtle highlight for perceived 3-D depth
#    · Each family has a clearly distinct plan-view silhouette
# ═══════════════════════════════════════════════════════════════════════════════

_SK = 'stroke="#00121C" stroke-width="2" stroke-linejoin="round" paint-order="stroke fill"'


def _cl(c, y1=6, y2=88):
    """Fuselage centreline highlight — subtle lighter streak for depth."""
    return (f'<line x1="50" y1="{y1}" x2="50" y2="{y2}"'
            f' stroke="{c}" stroke-opacity="0.22" stroke-width="1.5"'
            f' stroke-linecap="round"/>')


def _b747(c):
    """Boeing 747 — wide fuselage, 4 engines, very deep swept wings."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C54,4 58,9 58,16 C58,26 57,32 56,40 L56,72 C56,82 53,91 50,95 C47,91 44,82 44,72 L44,40 C43,32 42,26 42,16 C42,9 46,4 50,4 Z"/>
      <path d="M44,42 L44,60 L5,72 L7,60 Z"/>
      <path d="M56,42 L56,60 L95,72 L93,60 Z"/>
      <path d="M44,79 L44,85 L26,92 L27,87 Z"/>
      <path d="M56,79 L56,85 L74,92 L73,87 Z"/>
      <rect x="48.5" y="77" width="3" height="14" rx="1.5" stroke="none"/>
      <ellipse cx="28" cy="56" rx="4"   ry="8"/>
      <ellipse cx="16" cy="63" rx="3.5" ry="7"/>
      <ellipse cx="72" cy="56" rx="4"   ry="8"/>
      <ellipse cx="84" cy="63" rx="3.5" ry="7"/>
    </g>
    {_cl(c)}"""


def _a380(c):
    """Airbus A380 — widest fuselage (8 px), 4 engines, massive span."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C55,4 60,9 60,16 L60,72 C60,82 56,91 50,95 C44,91 40,82 40,72 L40,16 C40,9 45,4 50,4 Z"/>
      <path d="M40,41 L40,60 L4,73 L6,61 Z"/>
      <path d="M60,41 L60,60 L96,73 L94,61 Z"/>
      <path d="M40,79 L40,86 L22,93 L23,87 Z"/>
      <path d="M60,79 L60,86 L78,93 L77,87 Z"/>
      <rect x="48" y="77" width="4" height="14" rx="2" stroke="none"/>
      <ellipse cx="27" cy="57" rx="4.5" ry="8.5"/>
      <ellipse cx="14" cy="65" rx="3.5" ry="7"/>
      <ellipse cx="73" cy="57" rx="4.5" ry="8.5"/>
      <ellipse cx="86" cy="65" rx="3.5" ry="7"/>
    </g>
    {_cl(c)}"""


def _a340(c):
    """Airbus A340 — long slender fuselage, 4 high-aspect-ratio engines."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C52.5,4 55,8 55,14 L55,75 C55,84 52.5,91 50,96 C47.5,91 45,84 45,75 L45,14 C45,8 47.5,4 50,4 Z"/>
      <path d="M45,41 L45,57 L6,69 L8,59 Z"/>
      <path d="M55,41 L55,57 L94,69 L92,59 Z"/>
      <path d="M45,80 L45,86 L27,93 L28,88 Z"/>
      <path d="M55,80 L55,86 L73,93 L72,88 Z"/>
      <rect x="48.5" y="78" width="3" height="13" rx="1.5" stroke="none"/>
      <ellipse cx="29" cy="54" rx="3.5" ry="7"/>
      <ellipse cx="18" cy="61" rx="3"   ry="6"/>
      <ellipse cx="71" cy="54" rx="3.5" ry="7"/>
      <ellipse cx="82" cy="61" rx="3"   ry="6"/>
    </g>
    {_cl(c)}"""


def _b777(c):
    """Boeing 777 — wide body, raked wingtips, GE90 mega-nacelles."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C54,4 56,9 56,15 L56,72 C56,82 53,91 50,95 C47,91 44,82 44,72 L44,15 C44,9 46,4 50,4 Z"/>
      <path d="M44,41 L44,58 L5,70 L7,59 Z"/>
      <path d="M56,41 L56,58 L95,70 L93,59 Z"/>
      <path d="M5,70 L3,61 L7,59 Z"/>
      <path d="M95,70 L97,61 L93,59 Z"/>
      <path d="M44,79 L44,85 L27,92 L28,87 Z"/>
      <path d="M56,79 L56,85 L73,92 L72,87 Z"/>
      <rect x="48" y="77" width="4" height="13" rx="2" stroke="none"/>
      <ellipse cx="25" cy="55" rx="6"   ry="10"/>
      <ellipse cx="75" cy="55" rx="6"   ry="10"/>
    </g>
    {_cl(c)}"""


def _a350(c):
    """Airbus A350 — curved raked wingtips, Trent XWB nacelles."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C53.5,4 56,9 56,15 L56,72 C56,82 53,91 50,95 C47,91 44,82 44,72 L44,15 C44,9 46.5,4 50,4 Z"/>
      <path d="M44,41 L44,58 L7,69 L9,58 Z"/>
      <path d="M56,41 L56,58 L93,69 L91,58 Z"/>
      <path d="M7,69 L4,60 L9,58 Z"/>
      <path d="M93,69 L96,60 L91,58 Z"/>
      <path d="M44,79 L44,85 L27,92 L28,87 Z"/>
      <path d="M56,79 L56,85 L73,92 L72,87 Z"/>
      <rect x="48" y="77" width="4" height="13" rx="2" stroke="none"/>
      <ellipse cx="26" cy="54" rx="5.5" ry="9.5"/>
      <ellipse cx="74" cy="54" rx="5.5" ry="9.5"/>
    </g>
    {_cl(c)}"""


def _a330(c):
    """Airbus A330 — widebody twin, blunt wingtip or small winglet."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C53.5,4 56,9 56,15 L56,72 C56,82 53,91 50,95 C47,91 44,82 44,72 L44,15 C44,9 46.5,4 50,4 Z"/>
      <path d="M44,41 L44,58 L6,70 L8,59 Z"/>
      <path d="M56,41 L56,58 L94,70 L92,59 Z"/>
      <path d="M44,79 L44,85 L26,93 L27,88 Z"/>
      <path d="M56,79 L56,85 L74,93 L73,88 Z"/>
      <rect x="48" y="77" width="4" height="13" rx="2" stroke="none"/>
      <ellipse cx="26" cy="55" rx="5.5" ry="9.5"/>
      <ellipse cx="74" cy="55" rx="5.5" ry="9.5"/>
    </g>
    {_cl(c)}"""


def _b787(c):
    """Boeing 787 Dreamliner — dramatic raked wingtips, GEnx nacelles."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C53.5,4 56,9 56,15 L56,72 C56,82 53,91 50,95 C47,91 44,82 44,72 L44,15 C44,9 46.5,4 50,4 Z"/>
      <path d="M44,41 L44,57 L8,67 L10,56 Z"/>
      <path d="M56,41 L56,57 L92,67 L90,56 Z"/>
      <path d="M8,67 L4,57 L10,56 Z"/>
      <path d="M92,67 L96,57 L90,56 Z"/>
      <path d="M44,79 L44,85 L27,92 L28,87 Z"/>
      <path d="M56,79 L56,85 L73,92 L72,87 Z"/>
      <rect x="48" y="77" width="4" height="13" rx="2" stroke="none"/>
      <ellipse cx="25" cy="52" rx="5"   ry="9"/>
      <ellipse cx="75" cy="52" rx="5"   ry="9"/>
    </g>
    {_cl(c)}"""


def _b767(c):
    """Boeing 767 — medium widebody, slightly narrower than B777."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C53,4 55,9 55,15 L55,72 C55,82 52.5,91 50,95 C47.5,91 45,82 45,72 L45,15 C45,9 47,4 50,4 Z"/>
      <path d="M45,41 L45,57 L7,68 L9,58 Z"/>
      <path d="M55,41 L55,57 L93,68 L91,58 Z"/>
      <path d="M45,79 L45,85 L27,93 L28,88 Z"/>
      <path d="M55,79 L55,85 L73,93 L72,88 Z"/>
      <rect x="48" y="77" width="4" height="13" rx="2" stroke="none"/>
      <ellipse cx="26" cy="53" rx="5"   ry="9"/>
      <ellipse cx="74" cy="53" rx="5"   ry="9"/>
    </g>
    {_cl(c)}"""


def _b757(c):
    """Boeing 757 — very long narrow fuselage, tall distinctive V-fin."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,3 C52,3 53.5,7 53.5,13 L53.5,75 C53.5,84 52,91 50,96 C48,91 46.5,84 46.5,75 L46.5,13 C46.5,7 48,3 50,3 Z"/>
      <path d="M46.5,42 L46.5,56 L8,65 L10,55 Z"/>
      <path d="M53.5,42 L53.5,56 L92,65 L90,55 Z"/>
      <path d="M46.5,80 L46.5,86 L29,93 L30,88 Z"/>
      <path d="M53.5,80 L53.5,86 L71,93 L70,88 Z"/>
      <rect x="48.5" y="77" width="3" height="16" rx="1.5" stroke="none"/>
      <ellipse cx="27" cy="52" rx="4"   ry="7.5"/>
      <ellipse cx="73" cy="52" rx="4"   ry="7.5"/>
    </g>
    {_cl(c)}"""


def _a320(c):
    """Airbus A320 family — sharklets, CFM / IAE engines."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C52,4 54,8 54,14 L54,74 C54,83 52,91 50,95 C48,91 46,83 46,74 L46,14 C46,8 48,4 50,4 Z"/>
      <path d="M46,43 L46,57 L9,67 L11,57 Z"/>
      <path d="M54,43 L54,57 L91,67 L89,57 Z"/>
      <path d="M11,57 L7,56 L9,67 Z"/>
      <path d="M89,57 L93,56 L91,67 Z"/>
      <path d="M46,79 L46,85 L29,92 L30,87 Z"/>
      <path d="M54,79 L54,85 L71,92 L70,87 Z"/>
      <rect x="48.5" y="77" width="3" height="13" rx="1.5" stroke="none"/>
      <ellipse cx="27" cy="52" rx="4"   ry="7.5"/>
      <ellipse cx="73" cy="52" rx="4"   ry="7.5"/>
    </g>
    {_cl(c)}"""


def _b737(c):
    """Boeing 737 — split-scimitar winglets, underslung flat-bottom engines."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C52,4 54,8 54,14 L54,74 C54,83 52,91 50,95 C48,91 46,83 46,74 L46,14 C46,8 48,4 50,4 Z"/>
      <path d="M46,43 L46,57 L11,66 L13,57 Z"/>
      <path d="M54,43 L54,57 L89,66 L87,57 Z"/>
      <path d="M13,57 L8,55 L11,66 Z"/>
      <path d="M87,57 L92,55 L89,66 Z"/>
      <path d="M46,79 L46,85 L30,92 L31,87 Z"/>
      <path d="M54,79 L54,85 L70,92 L69,87 Z"/>
      <rect x="48.5" y="77" width="3" height="13" rx="1.5" stroke="none"/>
      <ellipse cx="28" cy="52" rx="4"   ry="7"/>
      <ellipse cx="72" cy="52" rx="4"   ry="7"/>
    </g>
    {_cl(c)}"""


def _e190(c):
    """Embraer E190/E195 — smaller regional jet, winglets."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,5 C51.5,5 53,8 53,14 L53,73 C53,82 51.5,90 50,94 C48.5,90 47,82 47,73 L47,14 C47,8 48.5,5 50,5 Z"/>
      <path d="M47,43 L47,56 L14,65 L16,56 Z"/>
      <path d="M53,43 L53,56 L86,65 L84,56 Z"/>
      <path d="M16,56 L13,55 L14,65 Z"/>
      <path d="M84,56 L87,55 L86,65 Z"/>
      <path d="M47,78 L47,83 L33,89 L34,85 Z"/>
      <path d="M53,78 L53,83 L67,89 L66,85 Z"/>
      <rect x="48.5" y="76" width="3" height="12" rx="1.5" stroke="none"/>
      <ellipse cx="30" cy="52" rx="3.5" ry="6.5"/>
      <ellipse cx="70" cy="52" rx="3.5" ry="6.5"/>
    </g>
    {_cl(c, 7, 84)}"""


def _crj(c):
    """Bombardier CRJ — rear-mount engines, prominent T-tail, clean swept wings."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,5 C51.5,5 53,8 53,14 L53,75 C53,83 51.5,90 50,94 C48.5,90 47,83 47,75 L47,14 C47,8 48.5,5 50,5 Z"/>
      <path d="M47,42 L47,53 L17,61 L19,53 Z"/>
      <path d="M53,42 L53,53 L83,61 L81,53 Z"/>
      <path d="M47,79 L47,84 L34,89 L35,85 Z"/>
      <path d="M53,79 L53,84 L66,89 L65,85 Z"/>
      <rect x="48.5" y="74" width="3" height="16" rx="1.5" stroke="none"/>
      <ellipse cx="43" cy="73" rx="3.5" ry="6.5"/>
      <ellipse cx="57" cy="73" rx="3.5" ry="6.5"/>
    </g>
    {_cl(c, 7, 82)}"""


def _atr(c):
    """ATR / Dash-8 turboprop — high straight wing, T-tail, prop discs."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,6 C51.5,6 53,9 53,15 L53,71 C53,79 51.5,87 50,91 C48.5,87 47,79 47,71 L47,15 C47,9 48.5,6 50,6 Z"/>
      <path d="M47,36 L13,40 L13,47 L47,45 Z"/>
      <path d="M53,36 L87,40 L87,47 L53,45 Z"/>
      <path d="M47,76 L47,81 L33,87 L34,83 Z"/>
      <path d="M53,76 L53,81 L67,87 L66,83 Z"/>
      <rect x="48.5" y="72" width="3" height="13" rx="1.5" stroke="none"/>
      <ellipse cx="27" cy="41" rx="3"   ry="6"/>
      <ellipse cx="73" cy="41" rx="3"   ry="6"/>
      <ellipse cx="27" cy="33" rx="12"  ry="3" fill-opacity="0.28" stroke="none"/>
      <ellipse cx="73" cy="33" rx="12"  ry="3" fill-opacity="0.28" stroke="none"/>
    </g>
    {_cl(c, 8, 84)}"""


def _concorde(c):
    """Concorde — ogival delta wing, needle nose, 4 underslung engines."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,3 C51,5 53,12 54,20 L68,75 L74,89 C68,87 61,84 56,83 L54,87 L50,94 L46,87 L44,83 C39,84 32,87 26,89 L32,75 L46,20 C47,12 49,5 50,3 Z"/>
      <ellipse cx="35" cy="64" rx="3"   ry="7.5"/>
      <ellipse cx="28" cy="70" rx="2.5" ry="6.5"/>
      <ellipse cx="65" cy="64" rx="3"   ry="7.5"/>
      <ellipse cx="72" cy="70" rx="2.5" ry="6.5"/>
    </g>
    {_cl(c, 5, 88)}"""


def _military(c):
    """Military cargo — high straight wing, 4 engines, boxy fuselage."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,6 C56,6 60,11 60,17 L60,74 C60,83 56,90 50,94 C44,90 40,83 40,74 L40,17 C40,11 44,6 50,6 Z"/>
      <path d="M40,36 L5,42 L5,50 L40,46 Z"/>
      <path d="M60,36 L95,42 L95,50 L60,46 Z"/>
      <path d="M40,78 L40,85 L22,91 L23,87 Z"/>
      <path d="M60,78 L60,85 L78,91 L77,87 Z"/>
      <rect x="47.5" y="74" width="5" height="15" rx="2.5" stroke="none"/>
      <ellipse cx="20" cy="44" rx="3.5" ry="7"/>
      <ellipse cx="34" cy="42" rx="3.5" ry="7"/>
      <ellipse cx="66" cy="42" rx="3.5" ry="7"/>
      <ellipse cx="80" cy="44" rx="3.5" ry="7"/>
    </g>
    {_cl(c, 8, 86)}"""


def _default(c):
    """Generic narrowbody twin — fallback for unrecognised ICAO types."""
    return f"""
    <g fill="{c}" {_SK}>
      <path d="M50,4 C52,4 54,8 54,14 L54,74 C54,83 52,91 50,95 C48,91 46,83 46,74 L46,14 C46,8 48,4 50,4 Z"/>
      <path d="M46,43 L46,57 L9,67 L11,57 Z"/>
      <path d="M54,43 L54,57 L91,67 L89,57 Z"/>
      <path d="M46,79 L46,85 L29,92 L30,87 Z"/>
      <path d="M54,79 L54,85 L71,92 L70,87 Z"/>
      <rect x="48.5" y="77" width="3" height="13" rx="1.5" stroke="none"/>
      <ellipse cx="27" cy="52" rx="4"   ry="7.5"/>
      <ellipse cx="73" cy="52" rx="4"   ry="7.5"/>
    </g>
    {_cl(c)}"""


