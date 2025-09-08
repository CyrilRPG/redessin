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
from coloraide import Color
import cairosvg
from PIL import Image

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

def average_segment_length(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points)-1):
        x1,y1 = points[i]; x2,y2 = points[i+1]
        total += ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    return total / max(1, (len(points)-1))

def make_smoothed_noise(N, roughness, rng):
    base = rng.normal(0, 1, N)
    k = int(max(3, 7*roughness))
    kernel = np.hanning(k); kernel /= kernel.sum()
    smooth = np.convolve(base, kernel, mode="same")
    std = np.std(smooth) or 1.0
    return smooth / std

def jitter_pair_polylines(points, amp=1.2, seed=42, roughness=1.0, stroke_width=1.0, rigor=0.75):
    """Return two polylines jittered symmetrically along normals (+d and -d),
    with displacement clamped to preserve geometry for anatomy diagrams.
    """
    if len(points) < 3:
        return points, points
    rng = np.random.default_rng(seed)
    N = len(points)
    noise = make_smoothed_noise(N, roughness, rng)
    norms = normals(points)
    # constraints
    Lavg = max(average_segment_length(points), 1e-6)
    max_abs = 0.6  # px hard cap
    max_by_len = 0.02 * Lavg  # 2% of avg segment length
    max_by_stroke = 0.6 * max(0.8, stroke_width)  # fraction of stroke width
    max_d = min(max_abs, max_by_len, max_by_stroke, amp)

    weights = compute_geometry_weights(points, protect_angles=True, end_taper_frac=0.18)
    plus = []
    minus = []
    for (x,y), (nx,ny), s, w in zip(points, norms, noise, weights):
        # rigor in [0,1]: closer to 1 → stronger damping
        damping = (rigor*0.85 + (1.0-rigor)*1.0)
        d = float(s) * max_d * w * damping
        plus.append((x + d*nx, y + d*ny))
        minus.append((x - d*nx, y - d*ny))
    return plus, minus

def compute_geometry_weights(points, protect_angles=True, end_taper_frac=0.15):
    """Return per-point weights in [0,1] reducing jitter near corners, tiny segments and endpoints."""
    N = len(points)
    if N == 0:
        return []
    if N == 1:
        return [1.0]
    # segment lengths
    seg_len = [0.0]*(N-1)
    for i in range(N-1):
        x1,y1 = points[i]; x2,y2 = points[i+1]
        seg_len[i] = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    Lavg = max(sum(seg_len)/max(1, len(seg_len)), 1e-6)
    # angle changes (internal points)
    ang = [0.0]*N
    if protect_angles and N > 2:
        for i in range(1, N-1):
            ax = points[i][0] - points[i-1][0]; ay = points[i][1] - points[i-1][1]
            bx = points[i+1][0] - points[i][0]; by = points[i+1][1] - points[i][1]
            la = (ax*ax+ay*ay)**0.5 or 1.0
            lb = (bx*bx+by*by)**0.5 or 1.0
            ax/=la; ay/=la; bx/=lb; by/=lb
            dot = max(-1.0, min(1.0, ax*bx + ay*by))
            ang[i] = np.arccos(dot)  # radians
    # build weights
    weights = [1.0]*N
    theta0 = np.deg2rad(25.0)
    for i in range(N):
        # length weight from adjacent segs
        l_here = 0.0
        if i>0:
            l_here += seg_len[i-1]
        if i < N-1:
            l_here += seg_len[i]
        l_here = l_here/ (2.0 if 0<i<N-1 else 1.0)
        w_len = max(0.35, min(1.0, l_here / (0.7*Lavg)))
        # angle weight (small near corners)
        w_ang = 1.0
        if protect_angles and 0<i<N-1:
            a = abs(ang[i])
            w_ang = 1.0 / (1.0 + (a/theta0)**2)
            w_ang = max(0.25, min(1.0, w_ang))
        weights[i] = min(1.0, w_len * w_ang)
    # endpoint taper for open polylines
    if N>2 and not is_closed_polyline(points):
        k = max(1, int(end_taper_frac * N))
        for i in range(min(k, N)):
            t = (i+1)/float(k+1)
            weights[i] *= t
            weights[N-1-i] *= t
    return weights

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

# -------- Perceptual color helpers (OKLCH) --------
def hex_oklch_shift(color, dL=0.0, dC=0.0, dH=0.0):
    try:
        col = Color(color)
        oklch = col.convert('oklch')
        L, C, H = oklch['l'], oklch['c'], oklch['h']
        L = max(0.0, min(1.0, L + dL))
        C = max(0.0, C + dC)
        H = (H + dH) % 360.0
        shifted = Color('oklch', [L, C, H]).convert('srgb')
        # clamp and output hex
        r = clamp(round(shifted['r']*255)); g = clamp(round(shifted['g']*255)); b = clamp(round(shifted['b']*255))
        return rgb_to_hex((r,g,b))
    except Exception:
        return color

def collect_svg_colors(root):
    colors = []
    for el in root.iter():
        for attr in ('fill','stroke'):
            v = el.get(attr)
            if v and v != 'none' and v.startswith('#'):
                colors.append(v)
    return colors

def compute_palette_shift(colors, seed=42, max_dL=0.02, max_dC=0.02, max_dH=3.0):
    if not colors:
        return (0.0,0.0,0.0)
    rng = np.random.default_rng(seed)
    dL = float(rng.uniform(-max_dL, max_dL))
    dC = float(rng.uniform(-max_dC, max_dC))
    dH = float(rng.uniform(-max_dH, max_dH))
    return (dL,dC,dH)

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

def redraw_svg(svg_bytes, *, density=1.8, jitter=1.2, jitter2=0.8, smooth_passes=2, stroke_gain=1.1, replace_strokes=True, seed=42,
               roughness=1.2, enable_extra_pass=True, jitter3=0.5, stroke_variation=0.08,
               color_variation_fill=0.06, color_variation_stroke=0.05):
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

    # Global palette shift (perceptual) for harmony
    palette_shift = compute_palette_shift(collect_svg_colors(root), seed=seed)
    # Keep viewBox/size
    out = ET.Element(root.tag, root.attrib)
    defs = ensure_defs(out)
    g_main = ET.SubElement(out, f"{{{SVG_NS}}}g", {"id":"handdrawn"})

    # (Hatching/warp removed for anatomy safety)

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

            # Build symmetric jittered polylines for geometry preservation
            p1, p2 = jitter_pair_polylines(pts, amp=jitter, seed=seed, roughness=roughness, stroke_width=stroke_width, rigor=0.85)
            if smooth_passes>0:
                p1 = chaikin_smooth(p1, passes=smooth_passes)
                p2 = chaikin_smooth(p2, passes=smooth_passes)

            d1 = polyline_to_pathd(p1); d2 = polyline_to_pathd(p2)
            paths_ds = [d1, d2]
            if enable_extra_pass:
                p3_plus, p3_minus = jitter_pair_polylines(pts, amp=max(0.25, min(jitter3, jitter*0.5)), seed=seed+2, roughness=roughness, stroke_width=stroke_width, rigor=0.9)
                p3 = p3_plus if (seed % 2)==0 else p3_minus
                if smooth_passes>0:
                    p3 = chaikin_smooth(p3, passes=smooth_passes)
                d3 = polyline_to_pathd(p3)
                paths_ds.append(d3)

            rng_local = np.random.default_rng(seed)
            for d in paths_ds:
                if not d:
                    continue
                this_stroke = stroke_color if stroke_color else "#000"
                # Random hue jitter removed; color changes handled below deterministically
                path = ET.SubElement(parent_group, f"{{{SVG_NS}}}path", {
                    "d": d,
                    "fill": "none",
                    "stroke": this_stroke,
                    "stroke-width": str(stroke_width * (1.0 + float(rng_local.normal(0, max(0.0, stroke_variation))))),
                    "stroke-linecap": linecap,
                    "stroke-linejoin": linejoin,
                    "opacity": "0.95"
                })

            # Adjust fill/stroke perceptually (OKLCH), skip black
            dL, dC, dH = palette_shift
            if fill_color and fill_color != "none" and not is_black(fill_color):
                preserved.set("fill", hex_oklch_shift(fill_color, dL=color_variation_fill*dL, dC=color_variation_fill*dC, dH=color_variation_fill*10*dH))
            if stroke_color and stroke_color != "none" and not is_black(stroke_color):
                preserved.set("stroke", hex_oklch_shift(stroke_color, dL=color_variation_stroke*dL, dC=color_variation_stroke*dC, dH=color_variation_stroke*10*dH))

    process(root, g_main)
    return pretty(out).encode("utf-8")

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Redessiner SVG — style main levée", page_icon="✏️", layout="wide")
st.title("✏️ Redessiner un SVG — Mode Anatomie (géométrie préservée)")

uploaded = st.file_uploader("Dépose un SVG", type=["svg"])

c1,c2,c3 = st.columns(3)
with c1:
    density = st.slider("Densité (points / 100 px)", 0.5, 6.0, 1.8, 0.1, help="Résolution de l'échantillonnage des contours.")
    smooth = st.slider("Lissage (passes Chaikin)", 0, 4, 2, 1, help="Plus lisse → plus 'feutre'. 0 conserve les aspérités.")
with c2:
    jitter = st.slider("Amplitude du jitter (px)", 0.0, 4.0, 1.2, 0.1, help="Amplitude maximale de déplacement perpendiculaire.")
    jitter2 = st.slider("Amplitude secondaire (px)", 0.0, 3.0, 0.8, 0.1, help="Amplitude secondaire pour richesse du trait.")
with c3:
    seed = st.number_input("Graine aléatoire", 0, 9999, 42, 1)
    stroke_gain = st.slider("Épaisseur relative des traits", 0.5, 2.5, 1.15, 0.05)
    replace_strokes = st.checkbox("Remplacer les traits originaux (recommandé)", True,
                                  help="Sinon on superpose les traits redessinés par dessus.")

with st.expander("Avancé"):
    cA, cB = st.columns(2)
    with cA:
        roughness = st.slider("Rugosité (bruit lissé)", 0.5, 3.0, 1.2, 0.1)
        enable_extra_pass = st.checkbox("Activer une 3ᵉ passe subtile", True)
        jitter3 = st.slider("Amplitude 3ᵉ passe (px)", 0.0, 2.0, 0.5, 0.1)
        stroke_variation = st.slider("Variation aléatoire d'épaisseur", 0.0, 0.4, 0.08, 0.02)
        rigor = st.slider("Rigueur anatomique (anti-déformation)", 0.0, 1.0, 0.85, 0.01)
    with cB:
        color_variation_fill = st.slider("Variation légère du fill", 0.0, 0.2, 0.06, 0.01)
        color_variation_stroke = st.slider("Variation légère du stroke", 0.0, 0.2, 0.05, 0.01)
    st.markdown("Mode Anatomie: jitter symétrique conscient de la géométrie; couleurs ajustées en OKLCH.")

if uploaded:
    raw = uploaded.read()
    # fallback defaults if expander not opened (Streamlit keeps state but ensure names exist)
    if 'roughness' not in locals():
        roughness = 1.2; enable_extra_pass = True; jitter3 = 0.5
        stroke_variation = 0.08
        color_variation_fill = 0.06; color_variation_stroke = 0.05

    out = redraw_svg(raw, density=density, jitter=jitter, jitter2=jitter2,
                     smooth_passes=smooth, stroke_gain=stroke_gain,
                     replace_strokes=replace_strokes, seed=int(seed),
                     roughness=roughness, enable_extra_pass=enable_extra_pass, jitter3=jitter3,
                     stroke_variation=stroke_variation,
                     color_variation_fill=color_variation_fill, color_variation_stroke=color_variation_stroke)
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

