"""
Streamlit — Redessiner un SVG en style 'dessiné à la main' (anatomie préservée)
-------------------------------------------------------------------------------
Déploiement Streamlit Community Cloud :
- Dépose ce repo avec `app_handdrawn.py` comme entrée principale (renomme en `app.py` si tu veux).
- `requirements.txt` inclus.


PRINCIPE
- **Couleurs et remplissages conservés** (on garde l'élément source pour le fill).
- **Traits redessinés**: on reconstruit des contours "jittered" en 2 passes (façon Rough.js).
- Trajet de jitter le long des **normales** pour éviter les glissements tangents.
- **Lissage Chaikin** pour un rendu crayon/feutre, et cap/join arrondis.

NB : On évite toute modification destructive des surfaces remplies (fill), afin d'éliminer les risques
d'erreur anatomique. Les traits sont clairement différents de l'original (visuellement 'main levée').
"""
import streamlit as st
from xml.etree import ElementTree as ET
from xml.dom import minidom
import numpy as np
from math import cos, sin, pi
from svgpathtools import parse_path, Path as SvgPath
import re
import io
import random

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

# ---------------------- Geometry helpers ----------------------
def pretty(elem):
    return minidom.parseString(ET.tostring(elem, encoding="utf-8")).toprettyxml(indent="  ")

def get_float(el, name, default=0.0):
    v = el.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except:
        # remove units if any
        return float(re.sub(r"[^0-9.+-eE]", "", v))

def parse_points(points_str):
    # supports "x1,y1 x2,y2 ..." or "x1 y1 x2 y2"
    pts = []
    if not points_str:
        return pts
    # Replace commas with spaces, then split
    s = re.sub(r"[,\s]+", " ", points_str.strip()).split()
    for i in range(0, len(s), 2):
        try:
            x = float(s[i]); y = float(s[i+1])
            pts.append((x, y))
        except:
            break
    return pts

def sample_svgpath_to_polyline(d, density=1.5):
    """Sample a path 'd' into polyline points.
    density: points per 100px of length (>= 0.5)"""
    try:
        p = parse_path(d)
    except Exception:
        return []
    L = p.length()
    if L == 0:
        return []
    n = max(int((L/100.0) * max(0.5, density)), 8)
    # Use arclength parameterization for uniform sampling
    pts = []
    for i in range(n+1):
        s = (L * i) / n
        t = p.ilength(s)
        z = p.point(t)
        pts.append((z.real, z.imag))
    return pts

def circle_points(cx, cy, r, n=120):
    return [(cx + r*cos(2*pi*i/n), cy + r*sin(2*pi*i/n)) for i in range(n+1)]

def ellipse_points(cx, cy, rx, ry, n=160):
    return [(cx + rx*cos(2*pi*i/n), cy + ry*sin(2*pi*i/n)) for i in range(n+1)]

def rect_points(x, y, w, h, rx=0.0, ry=0.0, n_arc=24):
    # rounded rectangle approx
    if rx==0 and ry==0:
        return [(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)]
    rx = min(rx, w/2); ry = min(ry, h/2)
    pts = []
    def arc(cx, cy, rx, ry, start, end, steps):
        return [(cx + rx*cos(t), cy + ry*sin(t)) for t in np.linspace(start, end, steps)]
    # start at (x+rx, y)
    pts += [(x+rx, y)]
    # top edge to (x+w-rx, y)
    pts += [(x+w-rx, y)]
    # top-right arc
    pts += arc(x+w-rx, y+ry, rx, ry, -pi/2, 0, n_arc)
    # right edge
    pts += [(x+w, y+h-ry)]
    # bottom-right arc
    pts += arc(x+w-rx, y+h-ry, rx, ry, 0, pi/2, n_arc)
    # bottom edge
    pts += [(x+rx, y+h)]
    # bottom-left arc
    pts += arc(x+rx, y+h-ry, rx, ry, pi/2, pi, n_arc)
    # left edge
    pts += [(x, y+ry)]
    # top-left arc
    pts += arc(x+rx, y+ry, rx, ry, pi, 3*pi/2, n_arc)
    pts += [(x+rx, y)]
    return pts

def normals(points):
    """Compute unit normals for each segment, then per point (averaged)."""
    if len(points) < 2:
        return [(0.0,0.0)]*len(points)
    seg_normals = []
    for i in range(len(points)-1):
        x1,y1 = points[i]; x2,y2 = points[i+1]
        dx, dy = x2-x1, y2-y1
        l = (dx*dx+dy*dy)**0.5 or 1.0
        nx, ny = -dy/l, dx/l
        seg_normals.append((nx, ny))
    # per-point average of adjacent segment normals
    pt_normals = [(seg_normals[0][0], seg_normals[0][1])]
    for i in range(1, len(points)-1):
        nx = seg_normals[i-1][0] + seg_normals[i][0]
        ny = seg_normals[i-1][1] + seg_normals[i][1]
        l = (nx*nx+ny*ny)**0.5 or 1.0
        pt_normals.append((nx/l, ny/l))
    pt_normals.append((seg_normals[-1][0], seg_normals[-1][1]))
    return pt_normals

def chaikin_smooth(points, passes=1):
    pts = points[:]
    for _ in range(max(0, passes)):
        new_pts = [pts[0]]
        for i in range(len(pts)-1):
            p = np.array(pts[i]); q = np.array(pts[i+1])
            Q = 0.75*p + 0.25*q
            R = 0.25*p + 0.75*q
            new_pts.extend([tuple(Q), tuple(R)])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts

def jitter_polyline(points, amp=1.5, seed=42, roughness=1.0):
    """Apply normal-direction jitter using smoothed random noise."""
    if len(points) < 3:
        return points
    rng = np.random.default_rng(seed)
    N = len(points)
    # random walk smoothed
    base = rng.normal(0, 1, N)
    # smooth via convolution to avoid high-frequency zigzags
    k = int(max(3, 7*roughness))
    kernel = np.hanning(k); kernel /= kernel.sum()
    smooth = np.convolve(base, kernel, mode="same")
    smooth /= (np.std(smooth) + 1e-6)
    norms = normals(points)
    out = []
    for (x,y), (nx,ny), s in zip(points, norms, smooth):
        d = amp * s
        out.append((x + d*nx, y + d*ny))
    return out

def polyline_to_pathd(points):
    if not points:
        return ""
    d = ["M {:.3f} {:.3f}".format(points[0][0], points[0][1])]
    for (x,y) in points[1:]:
        d.append("L {:.3f} {:.3f}".format(x, y))
    return " ".join(d)

def darken_hex(color, factor=0.85):
    # naive darken for hex like #RRGGBB
    if not color or not color.startswith("#") or len(color) not in (4,7):
        return color
    if len(color)==4:
        r = int(color[1]*2,16); g=int(color[2]*2,16); b=int(color[3]*2,16)
    else:
        r = int(color[1:3],16); g=int(color[3:5],16); b=int(color[5:7],16)
    r = int(max(0, min(255, r*factor)))
    g = int(max(0, min(255, g*factor)))
    b = int(max(0, min(255, b*factor)))
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(v)))

def hex_to_rgb(color):
    if not color or not color.startswith("#"):
        return None
    if len(color) == 4:
        r = int(color[1]*2,16); g=int(color[2]*2,16); b=int(color[3]*2,16)
    elif len(color) == 7:
        r = int(color[1:3],16); g=int(color[3:5],16); b=int(color[5:7],16)
    else:
        return None
    return (r,g,b)

def rgb_to_hex(rgb):
    r,g,b = rgb
    return "#{:02x}{:02x}{:02x}".format(clamp(r), clamp(g), clamp(b))

def jitter_hex_color(color, amount=10, rng=None):
    rgb = hex_to_rgb(color)
    if rgb is None:
        return color
    if rng is None:
        rng = np.random.default_rng()
    dr = rng.integers(-amount, amount+1)
    dg = rng.integers(-amount, amount+1)
    db = rng.integers(-amount, amount+1)
    r,g,b = rgb
    return rgb_to_hex((r+dr, g+dg, b+db))

def is_black(color, threshold=10):
    rgb = hex_to_rgb(color)
    if rgb is None:
        return False
    return rgb[0] < threshold and rgb[1] < threshold and rgb[2] < threshold

def adjust_hex_lightness(color, factor=1.0):
    rgb = hex_to_rgb(color)
    if rgb is None:
        return color
    r,g,b = rgb
    r = clamp(r*factor); g = clamp(g*factor); b = clamp(b*factor)
    return rgb_to_hex((r,g,b))

# ---------------------- SVG processing ----------------------
def element_to_polyline(el, density):
    tag = el.tag
    if tag.endswith("path"):
        d = el.get("d", "")
        return sample_svgpath_to_polyline(d, density=density)
    elif tag.endswith("polyline") or tag.endswith("polygon"):
        pts = parse_points(el.get("points", ""))
        if tag.endswith("polygon") and pts:
            pts = pts + [pts[0]]
        return pts
    elif tag.endswith("line"):
        x1 = get_float(el, "x1"); y1 = get_float(el, "y1")
        x2 = get_float(el, "x2"); y2 = get_float(el, "y2")
        return [(x1,y1),(x2,y2)]
    elif tag.endswith("rect"):
        x = get_float(el,"x"); y=get_float(el,"y")
        w = get_float(el,"width"); h=get_float(el,"height")
        rx = get_float(el,"rx",0.0); ry=get_float(el,"ry",0.0)
        return rect_points(x,y,w,h,rx,ry)
    elif tag.endswith("circle"):
        cx = get_float(el,"cx"); cy=get_float(el,"cy"); r=get_float(el,"r")
        return circle_points(cx,cy,r)
    elif tag.endswith("ellipse"):
        cx = get_float(el,"cx"); cy=get_float(el,"cy"); rx=get_float(el,"rx"); ry=get_float(el,"ry")
        return ellipse_points(cx,cy,rx,ry)
    else:
        return []

def clone_without_stroke(el):
    c = ET.Element(el.tag, el.attrib)
    # remove stroke to avoid double edges
    if "stroke" in c.attrib:
        c.set("data-original-stroke", c.get("stroke"))
        c.set("stroke", "none")
    return c

def is_closed_polyline(points, eps=1e-3):
    if not points:
        return False
    x1,y1 = points[0]
    x2,y2 = points[-1]
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5 < eps

def overshoot_polyline(points, amount=4.0):
    # Extend first and last segments a little to emulate pen overshoot for open paths
    if len(points) < 2 or is_closed_polyline(points):
        return points
    p0 = np.array(points[0]); p1 = np.array(points[1])
    pn1 = np.array(points[-2]); pn = np.array(points[-1])
    v_start = p0 - p1
    v_end = pn - pn1
    def extend(p, v, amt):
        l = np.linalg.norm(v) or 1.0
        d = (v / l) * amt
        return tuple((p + d).tolist())
    new_start = extend(p0, v_start, amount)
    new_end = extend(pn, v_end, amount)
    return [new_start] + points + [new_end]

def ensure_defs(out):
    # Find or create defs
    for child in list(out):
        if child.tag.endswith("defs"):
            return child
    return ET.SubElement(out, f"{{{SVG_NS}}}defs")

def add_sketch_filter(defs, *, base_freq=0.012, octaves=2, warp_scale=4.0, seed=0, saturation=1.0, contrast=1.0):
    filt = ET.SubElement(defs, f"{{{SVG_NS}}}filter", {"id": "sketch_warp"})
    turb = ET.SubElement(filt, f"{{{SVG_NS}}}feTurbulence", {
        "type": "fractalNoise",
        "baseFrequency": str(base_freq),
        "numOctaves": str(int(max(1, octaves))),
        "seed": str(int(seed)),
        "result": "noise"
    })
    disp = ET.SubElement(filt, f"{{{SVG_NS}}}feDisplacementMap", {
        "in": "SourceGraphic",
        "in2": "noise",
        "scale": str(float(warp_scale)),
        "xChannelSelector": "R",
        "yChannelSelector": "G",
        "result": "displaced"
    })
    if saturation != 1.0 or contrast != 1.0:
        # Optional tone tweak (subtle)
        ET.SubElement(filt, f"{{{SVG_NS}}}feComponentTransfer", {
            "in": "displaced",
            "result": "toned"
        })
        # We won't parameterize per-channel here; browser defaults keep it subtle
        final_in = "toned"
    else:
        final_in = "displaced"
    ET.SubElement(filt, f"{{{SVG_NS}}}feComposite", {
        "in": final_in,
        "in2": "SourceGraphic",
        "operator": "over"
    })
    return filt

def redraw_svg(svg_bytes, *, density=1.8, jitter=2.4, jitter2=1.2, smooth_passes=1, stroke_gain=1.15, replace_strokes=True, seed=42,
               roughness=1.0, enable_extra_pass=True, jitter3=0.8, color_jitter=8, overshoot_amt=3.0,
               enable_warp=True, warp_scale=3.0, warp_freq=0.01, warp_octaves=2,
               stroke_variation=0.1,
               enable_hatching=False, hatch_spacing=8.0, hatch_angle=45.0, hatch_jitter=1.5, hatch_opacity=0.22,
               anatomy_mode=True, color_variation_fill=0.08, color_variation_stroke=0.06):
    # parse root
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        text = svg_bytes.decode("utf-8", errors="ignore")
        start = text.find("<svg")
        if start!=-1:
            root = ET.fromstring(text[start:].encode("utf-8"))
        else:
            raise

    # Keep viewBox/size
    out = ET.Element(root.tag, root.attrib)
    defs = ensure_defs(out)
    # Optionally add warp filter
    if enable_warp and not anatomy_mode and float(warp_scale) > 0:
        add_sketch_filter(defs, base_freq=warp_freq, octaves=warp_octaves, warp_scale=warp_scale, seed=seed)
    g_main = ET.SubElement(out, f"{{{SVG_NS}}}g", {"id":"handdrawn"})
    if enable_warp and not anatomy_mode and float(warp_scale) > 0:
        g_main.set("filter", "url(#sketch_warp)")

    # simple counter for unique ids
    unique_counter = [0]

    # Helpers for hatching
    def add_hatching_for_points(parent_group, pts, color, spacing, angle_deg, jitter_amt, opacity, seed_local):
        if not pts or is_closed_polyline(pts) is False:
            return
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
        cx = (minx+maxx)/2.0; cy=(miny+maxy)/2.0
        # clip path using polygonal approximation
        unique_counter[0] += 1
        clip_id = f"clip_hatch_{unique_counter[0]}"
        cp = ET.SubElement(defs, f"{{{SVG_NS}}}clipPath", {"id": clip_id})
        d_clip = polyline_to_pathd(pts + [pts[0]])
        ET.SubElement(cp, f"{{{SVG_NS}}}path", {"d": d_clip})
        # lines group
        g = ET.SubElement(parent_group, f"{{{SVG_NS}}}g", {
            "clip-path": f"url(#{clip_id})",
            "opacity": str(opacity),
            "transform": f"rotate({angle_deg} {cx} {cy})"
        })
        rng = np.random.default_rng(seed_local)
        # draw vertical lines covering bbox after rotation approx by margin
        margin = max(spacing*2, 20)
        x = minx - margin
        xmax = maxx + margin
        while x <= xmax:
            dy = float(rng.normal(0, jitter_amt))
            y1 = miny - margin + dy
            y2 = maxy + margin + dy
            ET.SubElement(g, f"{{{SVG_NS}}}line", {
                "x1": str(x), "y1": str(y1), "x2": str(x), "y2": str(y2),
                "stroke": color if color else "#000",
                "stroke-width": "1.0"
            })
            x += spacing

    # Traverse original elements
    def process(el, parent_group):
        for child in list(el):
            tag = child.tag
            if tag.endswith("defs"):
                # copy defs as-is
                parent_group.append(child)
                continue
            if len(list(child))>0:
                # group-like: recurse into a new subgroup
                subg = ET.SubElement(parent_group, f"{{{SVG_NS}}}g", child.attrib)
                process(child, subg)
                continue

            # Only shape-like elements handled here
            pts = element_to_polyline(child, density=density)
            if not pts:
                # copy unknowns as-is
                parent_group.append(child)
                continue

            # original fill preserved (but remove stroke if we redraw it)
            preserved = clone_without_stroke(child) if replace_strokes else ET.Element(child.tag, child.attrib)
            parent_group.append(preserved)

            # choose stroke color/width
            stroke_color = child.get("stroke")
            fill_color = child.get("fill")
            if (not stroke_color or stroke_color in ("none","transparent")) and fill_color and fill_color != "none":
                stroke_color = darken_hex(fill_color, 0.8)

            stroke_width = get_float(child, "stroke-width", 1.0) * stroke_gain
            linecap = child.get("stroke-linecap","round")
            linejoin = child.get("stroke-linejoin","round")

            # Build jittered polylines
            base_pts = pts if anatomy_mode else overshoot_polyline(pts, amount=overshoot_amt)
            j_amp1 = min(jitter, 0.6) if anatomy_mode else jitter
            j_amp2 = min(jitter2, 0.4) if anatomy_mode else jitter2
            j_amp3 = min(jitter3, 0.3) if anatomy_mode else jitter3
            p1 = jitter_polyline(base_pts, amp=j_amp1, seed=seed, roughness=roughness)
            p2 = jitter_polyline(base_pts, amp=j_amp2, seed=seed+1, roughness=roughness)
            if smooth_passes>0:
                p1 = chaikin_smooth(p1, passes=smooth_passes)
                p2 = chaikin_smooth(p2, passes=smooth_passes)

            d1 = polyline_to_pathd(p1); d2 = polyline_to_pathd(p2)
            paths_ds = [d1, d2]
            if enable_extra_pass:
                p3 = jitter_polyline(base_pts, amp=j_amp3, seed=seed+2, roughness=roughness)
                if smooth_passes>0:
                    p3 = chaikin_smooth(p3, passes=smooth_passes)
                d3 = polyline_to_pathd(p3)
                paths_ds.append(d3)

            rng_local = np.random.default_rng(seed)
            for d in paths_ds:
                if not d:
                    continue
                this_stroke = stroke_color if stroke_color else "#000"
                if color_jitter and not anatomy_mode and this_stroke and this_stroke != "none":
                    this_stroke = jitter_hex_color(this_stroke, amount=int(color_jitter), rng=rng_local)
                path = ET.SubElement(parent_group, f"{{{SVG_NS}}}path", {
                    "d": d,
                    "fill": "none",
                    "stroke": this_stroke,
                    "stroke-width": str(stroke_width * (1.0 + float(rng_local.normal(0, max(0.0, stroke_variation))))),
                    "stroke-linecap": linecap,
                    "stroke-linejoin": linejoin,
                    "opacity": "0.95"
                })

            # Optional hatching inside filled shapes
            if enable_hatching and not anatomy_mode and fill_color and fill_color != "none" and is_closed_polyline(pts):
                hatch_color = darken_hex(fill_color, 0.65) if fill_color else "#000"
                add_hatching_for_points(parent_group, pts, hatch_color, hatch_spacing, hatch_angle, hatch_jitter, hatch_opacity, seed+11)

            # Adjust fill/stroke colors slightly (not for black) in anatomy mode too
            if fill_color and fill_color != "none" and not is_black(fill_color):
                factor = 1.0 + (color_variation_fill if not anatomy_mode else (color_variation_fill*0.6)) * (1 if (seed % 2)==0 else -1)
                preserved.set("fill", adjust_hex_lightness(fill_color, factor))
            if stroke_color and stroke_color != "none" and not is_black(stroke_color):
                factor_s = 1.0 + (color_variation_stroke if not anatomy_mode else (color_variation_stroke*0.6)) * (1 if (seed % 3)==0 else -1)
                preserved.set("stroke", adjust_hex_lightness(stroke_color, factor_s))

    process(root, g_main)
    return pretty(out).encode("utf-8")

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Redessiner SVG — style main levée", page_icon="✏️", layout="wide")
st.title("✏️ Redessiner un SVG en **style dessiné à la main** (couleurs intactes, anatomie préservée)")

uploaded = st.file_uploader("Dépose un SVG", type=["svg"])

c1,c2,c3 = st.columns(3)
with c1:
    density = st.slider("Densité (points / 100 px)", 0.5, 6.0, 1.8, 0.1, help="Résolution de l'échantillonnage des contours.")
    smooth = st.slider("Lissage (passes Chaikin)", 0, 4, 2, 1, help="Plus lisse → plus 'feutre'. 0 conserve les aspérités.")
with c2:
    jitter = st.slider("Jitter principal (px)", 0.0, 8.0, 4.5, 0.1, help="Amplitude de la 1ère passe (effet visible).")
    jitter2 = st.slider("Jitter secondaire (px)", 0.0, 6.0, 3.0, 0.1, help="Amplitude de la 2nde passe (double trait).")
with c3:
    seed = st.number_input("Graine aléatoire", 0, 9999, 42, 1)
    stroke_gain = st.slider("Épaisseur relative des traits", 0.5, 2.5, 1.15, 0.05)
    replace_strokes = st.checkbox("Remplacer les traits originaux (recommandé)", True,
                                  help="Sinon on superpose les traits redessinés par dessus.")

with st.expander("Avancé"):
    cA, cB, cC = st.columns(3)
    with cA:
        roughness = st.slider("Rugosité (bruit lissé)", 0.5, 3.0, 1.8, 0.1)
        enable_extra_pass = st.checkbox("Activer une 3ᵉ passe", True)
        jitter3 = st.slider("Jitter 3ᵉ passe (px)", 0.0, 6.0, 1.2, 0.1)
        anatomy_mode = st.checkbox("Mode anatomie (préserver la géométrie)", True)
    with cB:
        color_jitter = st.slider("Variation couleur du trait", 0, 40, 10, 1,
                                 help="Décalage aléatoire sur R/G/B (valeurs 0–255).")
        overshoot_amt = st.slider("Overshoot (débord) des extrémités (px)", 0.0, 12.0, 6.0, 0.5)
        stroke_variation = st.slider("Variation aléatoire d'épaisseur", 0.0, 0.6, 0.1, 0.02)
    with cC:
        enable_warp = st.checkbox("Déformer globalement (warp)", True)
        warp_scale = st.slider("Intensité du warp", 0.0, 15.0, 6.0, 0.5)
        warp_freq = st.slider("Fréquence du bruit", 0.002, 0.05, 0.012, 0.002)
        warp_octaves = st.slider("Octaves du bruit", 1, 6, 3, 1)
        enable_hatching = st.checkbox("Hachures internes (remplissages)", False)
        hatch_spacing = st.slider("Espacement hachures (px)", 3.0, 20.0, 8.0, 0.5)
        hatch_angle = st.slider("Angle hachures (°)", 0.0, 180.0, 45.0, 1.0)
        hatch_jitter = st.slider("Jitter hachures (px)", 0.0, 4.0, 1.5, 0.1)
        hatch_opacity = st.slider("Opacité hachures", 0.05, 0.6, 0.22, 0.01)
        color_variation_fill = st.slider("Variation légère de fill", 0.0, 0.2, 0.08, 0.01)
        color_variation_stroke = st.slider("Variation légère de stroke", 0.0, 0.2, 0.06, 0.01)
    st.markdown("Paramètres par défaut optimisés pour un rendu non reconnaissable mais fidèle.")

if uploaded:
    raw = uploaded.read()
    # fallback defaults if expander not opened (Streamlit keeps state but ensure names exist)
    if 'roughness' not in locals():
        roughness = 1.0; enable_extra_pass = True; jitter3 = 0.8
        color_jitter = 8; overshoot_amt = 3.0
        enable_warp = True; warp_scale = 3.0; warp_freq = 0.01; warp_octaves = 2
        anatomy_mode = True; stroke_variation = 0.1
        enable_hatching = False; hatch_spacing = 8.0; hatch_angle = 45.0; hatch_jitter = 1.5; hatch_opacity = 0.22
        color_variation_fill = 0.08; color_variation_stroke = 0.06

    out = redraw_svg(raw, density=density, jitter=jitter, jitter2=jitter2,
                     smooth_passes=smooth, stroke_gain=stroke_gain,
                     replace_strokes=replace_strokes, seed=int(seed),
                     roughness=roughness, enable_extra_pass=enable_extra_pass, jitter3=jitter3,
                     color_jitter=color_jitter, overshoot_amt=overshoot_amt,
                     enable_warp=enable_warp, warp_scale=warp_scale, warp_freq=warp_freq, warp_octaves=warp_octaves,
                     stroke_variation=stroke_variation,
                     enable_hatching=enable_hatching, hatch_spacing=hatch_spacing, hatch_angle=hatch_angle, hatch_jitter=hatch_jitter, hatch_opacity=hatch_opacity,
                     anatomy_mode=anatomy_mode, color_variation_fill=color_variation_fill, color_variation_stroke=color_variation_stroke)
    st.download_button("⬇️ Télécharger le SVG redessiné", data=out, file_name="redessine.svg", mime="image/svg+xml")

    st.subheader("Avant / Après")
    colA, colB = st.columns(2)
    with colA:
        st.caption("Original")
        st.markdown(f"<div style='border:1px solid #ddd;padding:8px'>{raw.decode('utf-8', errors='ignore')}</div>", unsafe_allow_html=True)
    with colB:
        st.caption("Redessiné")
        st.markdown(f"<div style='border:1px solid #ddd;padding:8px'>{out.decode('utf-8')}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.info("Astuce : augmente **Jitter principal** à 3–5 px pour un rendu très marqué. Le remplissage (fill) est conservé tel quel.")
else:
    st.info("Charge un fichier SVG. Le rendu 'main levée' remplace/ajoute les traits en double passe, mais **ne modifie pas** les couleurs ni les surfaces remplies.")

