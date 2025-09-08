# app.py
# ------------------------------------------------------------
# Streamlit — Redessiner un SVG (effet "dessiné à la main")
# SANS déformer la géométrie (anatomie préservée)
#
# - Conserve strictement les coordonnées des chemins (aucun jitter / warp)
# - Décale légèrement les couleurs (HSL) pour réduire la ressemblance
# - Redessine les traits : 2–3 passes en pointillés arrondis
# - Variation subtile d'épaisseur / opacité + "grain" doux (aucun déplacement)
# - Option : hachures internes via clipPath (bordures intactes)
#
# Déploiement Streamlit Cloud :
# 1) requirements.txt : streamlit>=1.28
# 2) streamlit run app.py (ou via share.streamlit.io)
# ------------------------------------------------------------

import streamlit as st
from xml.etree import ElementTree as ET
from xml.dom import minidom
import re
import random

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# --------------------- Utils: parsing & styles ---------------------

def pretty(elem):
    """Beautify XML output."""
    return minidom.parseString(ET.tostring(elem, encoding="utf-8")).toprettyxml(indent="  ")

def parse_style_attr(style_str):
    """Parse inline style='a:b;c:d' -> dict."""
    out = {}
    if not style_str:
        return out
    for part in style_str.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k:
                out[k] = v
    return out

def merge_inline_styles(el):
    """
    Return a dict of 'computed' attributes for fill/stroke/stroke-width/opacity/etc.
    Priority: explicit attributes > inline style.
    """
    computed = dict(el.attrib)  # copy
    styles = parse_style_attr(el.get("style", ""))

    # Map common CSS style keys to attributes if not already present
    mapping = {
        "fill": "fill",
        "stroke": "stroke",
        "stroke-width": "stroke-width",
        "stroke-linecap": "stroke-linecap",
        "stroke-linejoin": "stroke-linejoin",
        "opacity": "opacity",
        "fill-opacity": "fill-opacity",
        "stroke-opacity": "stroke-opacity",
    }
    for sk, ak in mapping.items():
        if ak not in computed and sk in styles:
            computed[ak] = styles[sk]

    return computed

def duplicate_with_computed(el):
    """Duplicate element and 'normalize' inline styles into attributes on the copy."""
    comp = merge_inline_styles(el)
    dup = ET.Element(el.tag, comp)
    return dup

# --------------------- Color conversions (HSL) ---------------------

def clamp01(x):
    return max(0.0, min(1.0, x))

def parse_color(cstr):
    """Return (r,g,b,a) or None if unsupported (named colors left unchanged)."""
    if not cstr or cstr in ("none", "transparent"):
        return None
    cstr = cstr.strip()
    if cstr.startswith("#"):
        # #rgb or #rrggbb
        if len(cstr) == 4:
            r = int(cstr[1]*2, 16); g = int(cstr[2]*2, 16); b = int(cstr[3]*2, 16)
            return (r, g, b, 1.0)
        if len(cstr) == 7:
            r = int(cstr[1:3], 16); g = int(cstr[3:5], 16); b = int(cstr[5:7], 16)
            return (r, g, b, 1.0)
    m = re.match(r"rgba?\(([^)]+)\)", cstr, re.I)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            r = int(float(parts[0])); g = int(float(parts[1])); b = int(float(parts[2]))
            a = float(parts[3]) if len(parts) == 4 else 1.0
            return (r, g, b, a)
    return None  # named color -> unchanged

def css_rgba(rgba):
    r, g, b, a = rgba
    if a >= 0.999:
        return f"rgb({r},{g},{b})"
    return f"rgba({r},{g},{b},{a:.3f})"

def rgb_to_hsl(r, g, b):
    r /= 255.0; g /= 255.0; b /= 255.0
    mx = max(r, g, b); mn = min(r, g, b)
    l = (mx + mn) / 2.0
    if mx == mn:
        return (0.0, 0.0, l)
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    h /= 6.0
    return (h * 360.0, s, l)

def hsl_to_rgb(h, s, l):
    h = (h % 360.0) / 360.0

    def f(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p

    if s == 0:
        r = g = b = l
    else:
        q = l + s - l * s if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = f(p, q, h + 1/3)
        g = f(p, q, h)
        b = f(p, q, h - 1/3)

    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

def shift_color(cstr, dh=8.0, ds=0.05, dl=0.02):
    """Shift color in HSL; leave unsupported/named colors unchanged."""
    rgba = parse_color(cstr)
    if not rgba:
        return cstr
    r, g, b, a = rgba
    h, s, l = rgb_to_hsl(r, g, b)
    h = (h + dh) % 360.0
    s = clamp01(s * (1.0 + ds))
    l = clamp01(l * (1.0 + dl))
    r2, g2, b2 = hsl_to_rgb(h, s, l)
    return css_rgba((r2, g2, b2, a))

# --------------------- Filters & effects ---------------------

def ensure_defs(root):
    for c in list(root):
        if c.tag.endswith("defs"):
            return c
    return ET.SubElement(root, f"{{{SVG_NS}}}defs")

def add_grain_filter(defs, fid="grain", freq=0.8, seed=1, intensity=0.07):
    """
    'Grain' doux : pas de feDisplacementMap (donc aucun déplacement).
    On module légèrement l'alpha visuelle pour un rendu papier.
    """
    f = ET.SubElement(defs, f"{{{SVG_NS}}}filter", {
        "id": fid, "x": "-3%", "y": "-3%", "width": "106%", "height": "106%",
        "color-interpolation-filters": "sRGB"
    })
    ET.SubElement(f, f"{{{SVG_NS}}}feTurbulence", {
        "type": "fractalNoise", "baseFrequency": str(freq),
        "numOctaves": "2", "seed": str(seed), "result": "n"
    })
    ET.SubElement(f, f"{{{SVG_NS}}}feColorMatrix", {
        "in": "n", "type": "saturate", "values": "0", "result": "n2"
    })
    ET.SubElement(f, f"{{{SVG_NS}}}feComposite", {
        "in": "SourceGraphic", "in2": "n2", "operator": "arithmetic",
        "k1": "0", "k2": "1", "k3": f"{float(intensity):.3f}", "k4": "0"
    })
    return f

def make_sketch_strokes(base_el, passes=2, width=1.2, seed=42, stroke_color=None):
    """
    Crée 'passes' duplicates du même élément avec :
    - fill none
    - stroke dasharray aléatoire
    - caps/joins arrondis
    - légère variation d'épaisseur/opacity
    - filtre grain (pas de déplacement)
    """
    rnd = random.Random(seed)
    overlays = []
    for _ in range(passes):
        d = duplicate_with_computed(base_el)
        d.set("fill", "none")
        if stroke_color:
            d.set("stroke", stroke_color)
        # width variation
        w = width * (0.92 + 0.16 * rnd.random())
        d.set("stroke-width", f"{w:.3f}")
        d.set("stroke-linecap", d.get("stroke-linecap", "round"))
        d.set("stroke-linejoin", d.get("stroke-linejoin", "round"))
        # dash pattern
        base = 6.0 * (0.9 + 0.2 * rnd.random())
        segs = [max(1.0, base * (0.6 + 0.8 * rnd.random())) for _ in range(10)]
        d.set("stroke-dasharray", " ".join(f"{s:.2f}" for s in segs))
        d.set("stroke-dashoffset", f"{base * rnd.random():.2f}")
        d.set("opacity", f"{0.78 + 0.18 * rnd.random():.3f}")
        d.set("filter", "url(#grain)")
        overlays.append(d)
    return overlays

# --------------------- Hatching (optionnel) ---------------------

SAFE_SHAPES_FOR_HATCH = {"path", "polygon", "rect", "circle", "ellipse"}

def add_hatching(defs, parent_group, el, spacing=8.0, angle=45.0, opacity=0.18):
    """
    Hachures internes : on clippe un groupe de lignes à la forme.
    Note : si l'élément a des 'transform' complexes, l'alignement peut varier.
    """
    tag_local = el.tag.split("}")[-1]
    if tag_local not in SAFE_SHAPES_FOR_HATCH:
        return
    # Créer clipPath unique
    clip_id = f"hclip_{id(el)}"
    cp = ET.SubElement(defs, f"{{{SVG_NS}}}clipPath", {"id": clip_id})
    cp.append(duplicate_with_computed(el))
    # Groupe hachures
    g = ET.SubElement(parent_group, f"{{{SVG_NS}}}g", {
        "clip-path": f"url(#{clip_id})",
        "opacity": str(opacity),
        "transform": f"rotate({angle})"
    })
    # Lignes infinies (grosses marges), on compte sur le clip
    x = -5000.0
    while x <= 5000.0:
        line = ET.SubElement(g, f"{{{SVG_NS}}}line", {
            "x1": str(x), "y1": "-5000", "x2": str(x), "y2": "5000",
            "stroke": el.get("fill", "#000"),
            "stroke-width": "1.0",
            "stroke-linecap": "round"
        })
        x += float(spacing)

# --------------------- Core processing ---------------------

SAFE_LEAF = {"path", "line", "polyline", "polygon", "rect", "circle", "ellipse", "text"}

def process_svg(svg_bytes,
                hue_shift=8.0, sat_gain=0.05, light_gain=0.02,
                passes=2, stroke_gain=1.2,
                replace_original_strokes=True,
                seed=42,
                enable_hatching=False, hatch_spacing=8.0, hatch_angle=45.0, hatch_opacity=0.18,
                grain_freq=0.8, grain_intensity=0.07):
    """
    - Ne modifie pas la géométrie
    - Recolore légèrement (HSL)
    - Redessine traits en passes pointillées
    """
    # Parse
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        txt = svg_bytes.decode("utf-8", errors="ignore")
        start = txt.find("<svg")
        if start != -1:
            root = ET.fromstring(txt[start:].encode("utf-8"))
        else:
            raise

    # Nouveau root avec mêmes attributs (viewBox/width/height)
    out = ET.Element(root.tag, root.attrib)
    defs = ensure_defs(out)
    add_grain_filter(defs, seed=int(seed), freq=float(grain_freq), intensity=float(grain_intensity))
    wrapper = ET.SubElement(out, f"{{{SVG_NS}}}g", {"id": "handdrawn"})

    def walk(node, parent):
        for child in list(node):
            local = child.tag.split("}")[-1]

            if local == "defs":
                # Reprendre les defs originales
                parent.append(child)
                continue

            # Si group, on recrée le groupe et on descend
            if len(list(child)) > 0 and local not in SAFE_LEAF:
                g = ET.SubElement(parent, child.tag, child.attrib)
                walk(child, g)
                continue

            if local in SAFE_LEAF:
                # Base = duplication avec styles inline normalisés
                base = duplicate_with_computed(child)

                # Recoloration légère (HSL) des fills/strokes
                # (conserve couleurs nommées inchangées)
                fill = base.get("fill")
                if fill and fill not in ("none", "transparent"):
                    base.set("fill", shift_color(fill, dh=hue_shift, ds=sat_gain, dl=light_gain))

                stroke = base.get("stroke")
                had_stroke = (stroke and stroke not in ("none", "transparent"))
                if had_stroke:
                    # Augmenter un peu l'épaisseur pour que les passes croquis ressortent
                    try:
                        w = float(re.sub(r"[^0-9.+-eE]", "", base.get("stroke-width", "1") or "1"))
                    except Exception:
                        w = 1.0
                    base.set("stroke-width", f"{w * stroke_gain:.3f}")
                    base.set("stroke-linecap", base.get("stroke-linecap", "round"))
                    base.set("stroke-linejoin", base.get("stroke-linejoin", "round"))
                    base.set("stroke", shift_color(stroke, dh=hue_shift * 0.6, ds=sat_gain, dl=light_gain * 0.5))

                # Appliquer un grain doux sur la base (aucune déformation)
                base.set("filter", "url(#grain)")

                # Option : remplacer complètement le trait original par les passes "croquis"
                if replace_original_strokes and had_stroke:
                    base.set("stroke", "none")

                # Ajouter la base
                parent.append(base)

                # Overlays croquis (multi-passes) si l'élément a un trait
                # ou si c'est typiquement un contour (line/polyline/path)
                overlay_needed = had_stroke or local in ("line", "polyline", "path")
                if overlay_needed:
                    # Choisir couleur d'overlay : stroke si présent, sinon fill
                    overlay_color = stroke if had_stroke else base.get("fill")
                    if not overlay_color or overlay_color in ("none", "transparent"):
                        overlay_color = "#000"
                    # Calcul largeur d'après base (si on a un stroke-width)
                    try:
                        w = float(re.sub(r"[^0-9.+-eE]", "", (child.get("stroke-width") or base.get("stroke-width") or "1")))
                    except Exception:
                        w = 1.0
                    for ov in make_sketch_strokes(child, passes=int(passes), width=w * stroke_gain, seed=int(seed), stroke_color=overlay_color):
                        parent.append(ov)

                # Hachures facultatives (strictement à l'intérieur de la forme)
                if enable_hatching and base.get("fill") and base.get("fill") != "none":
                    if local in SAFE_SHAPES_FOR_HATCH:
                        add_hatching(defs, parent, child, spacing=hatch_spacing, angle=hatch_angle, opacity=hatch_opacity)

            else:
                # Élément feuille inconnu : copie brute
                parent.append(child)

    walk(root, wrapper)
    return pretty(out).encode("utf-8")

# --------------------- Streamlit UI ---------------------

st.set_page_config(page_title="SVG main-levée (anatomie safe)", page_icon="✏️", layout="wide")
st.title("✏️ Redessiner un SVG — **style dessiné à la main** sans déformer les structures")

uploaded = st.file_uploader("Dépose ton fichier SVG", type=["svg"])

c1, c2, c3 = st.columns(3)
with c1:
    hue = st.slider("Décalage teinte (°)", -24, 24, 8, 1)
    sat = st.slider("Gain saturation", -0.3, 0.3, 0.05, 0.01)
with c2:
    light = st.slider("Gain luminosité", -0.3, 0.3, 0.02, 0.01)
    passes = st.slider("Passes de trait (croquis)", 1, 3, 2, 1)
with c3:
    stroke_gain = st.slider("Épaisseur relative du trait", 0.5, 2.5, 1.2, 0.05)
    seed = st.number_input("Graine aléatoire", 0, 9999, 42, 1)

replace = st.checkbox("Remplacer le trait original (recommandé)", True,
                      help="Supprime le stroke de base et ajoute 2–3 passes en pointillés par-dessus.")
st.markdown("—")

with st.expander("Options avancées"):
    enable_hatching = st.checkbox("Hachures internes (optionnel)", False)
    hatch_spacing = st.slider("Espacement des hachures (px)", 4.0, 20.0, 8.0, 0.5)
    hatch_angle = st.slider("Angle des hachures (°)", 0.0, 180.0, 45.0, 1.0)
    hatch_opacity = st.slider("Opacité des hachures", 0.05, 0.5, 0.18, 0.01)
    grain_freq = st.slider("Grain – fréquence bruit", 0.2, 1.5, 0.8, 0.05)
    grain_intensity = st.slider("Grain – intensité", 0.00, 0.20, 0.07, 0.01)

if uploaded:
    raw = uploaded.read()
    out = process_svg(
        raw,
        hue_shift=float(hue),
        sat_gain=float(sat),
        light_gain=float(light),
        passes=int(passes),
        stroke_gain=float(stroke_gain),
        replace_original_strokes=bool(replace),
        seed=int(seed),
        enable_hatching=bool(enable_hatching),
        hatch_spacing=float(hatch_spacing),
        hatch_angle=float(hatch_angle),
        hatch_opacity=float(hatch_opacity),
        grain_freq=float(grain_freq),
        grain_intensity=float(grain_intensity),
    )

    st.download_button("⬇️ Télécharger le SVG redessiné (safe)", data=out,
                       file_name="handdrawn_safe.svg", mime="image/svg+xml")

    colA, colB = st.columns(2)
    with colA:
        st.caption("Original")
        st.markdown(
            f"<div style='border:1px solid #ddd;padding:6px;max-height:72vh;overflow:auto'>{raw.decode('utf-8', errors='ignore')}</div>",
            unsafe_allow_html=True,
        )
    with colB:
        st.caption("Redessiné (géométrie inchangée)")
        st.markdown(
            f"<div style='border:1px solid #ddd;padding:6px;max-height:72vh;overflow:auto'>{out.decode('utf-8')}</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("Charge un SVG. L’app décale légèrement les couleurs et superpose des traits en pointillés arrondis "
            "pour un rendu 'dessiné à la main' **sans déplacer un seul point** du schéma.")
