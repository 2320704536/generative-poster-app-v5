import random
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from colorsys import rgb_to_hsv, hsv_to_rgb
from PIL import Image, ImageEnhance, ImageFilter

st.set_page_config(page_title="Generative Poster v5.2.1", layout="wide")
st.title("Generative Abstract Poster v5.2.1")
st.markdown("Stable Release — Cleaner, Faster, No Autoplay")

# ---------- Helpers: palette ----------
def clamp01(x): return max(0.0, min(1.0, x))

def hex_to_rgb01(hex_str):
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return r, g, b

def custom_palette_from_hex(hex_str, k=7):
    r, g, b = hex_to_rgb01(hex_str)
    h, s, v = rgb_to_hsv(r, g, b)
    values = np.linspace(0.35, 0.95, k)
    sats = np.linspace(min(0.25, s*0.8), min(0.6, max(0.3, s)), k)
    return [hsv_to_rgb(h, float(sats[i]), float(values[i])) for i in range(k)]

def pastel_palette(k=6):
    return [(random.uniform(0.65,0.95), random.uniform(0.65,0.95), random.uniform(0.65,0.95)) for _ in range(k)]

def vibrant_palette(k=6):
    anchors = [(0.95,0.30,0.30),(1.00,0.65,0.00),(0.20,0.70,0.30),(0.20,0.40,0.95),(0.65,0.25,0.90),(0.95,0.20,0.60)]
    cols = []
    for i in range(k):
        r,g,b = anchors[i % len(anchors)]
        cols.append((r,g,b))
    return cols

def mono_palette(k=6):
    h = random.random()
    base = np.linspace(0.35, 0.95, k)
    return [hsv_to_rgb(h, 0.4, v) for v in base]

def random_palette(k=6):
    return [(random.random(), random.random(), random.random()) for _ in range(k)]

def pink_palette(k=6):
    return [(random.uniform(0.9,1.0), random.uniform(0.4,0.75), random.uniform(0.6,0.9)) for _ in range(k)]

def blue_palette(k=6):
    return [(random.uniform(0.2,0.5), random.uniform(0.4,0.8), random.uniform(0.7,1.0)) for _ in range(k)]

def green_palette(k=6):
    return [(random.uniform(0.2,0.5), random.uniform(0.6,1.0), random.uniform(0.3,0.7)) for _ in range(k)]

def get_palette(kind, k=6):
    return {
        "Pastel": pastel_palette, "Vibrant": vibrant_palette, "Mono": mono_palette,
        "Random": random_palette, "Pink": pink_palette, "Blue": blue_palette, "Green": green_palette
    }.get(kind, pastel_palette)(k)

# ---------- Shapes ----------
def blob(center=(0.5,0.5), r=0.3, points=200, wobble=0.15):
    ang = np.linspace(0, 2*math.pi, points)
    rad = r * (1 + wobble * (np.random.rand(points) - 0.5))
    return center[0] + rad*np.cos(ang), center[1] + rad*np.sin(ang)

def polygon(center=(0.5,0.5), sides=6, r=0.3, wobble=0.1):
    ang = np.linspace(0, 2*math.pi, sides, endpoint=False)
    rad = r * (1 + wobble * (np.random.rand(sides) - 0.5))
    x = center[0] + rad*np.cos(ang); y = center[1] + rad*np.sin(ang)
    return np.append(x, x[0]), np.append(y, y[0])

def waves(center=(0.5,0.5), r=0.3, points=400, frequency=6, wobble=0.05):
    ang = np.linspace(0, 2*math.pi, points)
    rad = r * (1 + wobble * np.sin(frequency * ang))
    return center[0] + rad*np.cos(ang), center[1] + rad*np.sin(ang)

# ---------- Background ----------
def set_background(ax, mode):
    if mode == "Off-white":
        ax.set_facecolor((0.98, 0.98, 0.97)); return "dark"
    if mode == "Dark":
        ax.set_facecolor((0.08, 0.08, 0.08)); return "light"
    ax.set_facecolor((1,1,1)); return "dark"

# ---------- Draw Poster ----------
def draw_poster(shape="Blob", layers=8, wobble=0.15, palette_kind="Pastel", bg="Off-white",
                seed=None, palette_override=None):
    if seed not in (None, "", 0):
        try:
            seed = int(seed); random.seed(seed); np.random.seed(seed)
        except:
            pass
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.axis("off")
    text_mode = set_background(ax, bg)
    cols = palette_override if palette_override else get_palette(palette_kind, 7)
    for _ in range(layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)
        color = random.choice(cols)
        alpha = random.uniform(0.25, 0.6)
        if shape == "Blob":
            x, y = blob((cx, cy), rr, wobble=wobble)
            ax.fill(x, y, color=color, alpha=alpha)
        elif shape == "Polygon":
            x, y = polygon((cx, cy), sides=random.randint(3,8), r=rr, wobble=wobble)
            ax.fill(x, y, color=color, alpha=alpha)
        elif shape == "Waves":
            x, y = waves((cx, cy), rr, frequency=random.randint(4,8), wobble=wobble)
            ax.fill(x, y, color=color, alpha=alpha)
    txt_color = (0.95,0.95,0.95) if text_mode == "light" else (0.1,0.1,0.1)
    ax.text(0.05,0.95,"Generative Poster",fontsize=18,weight="bold",transform=ax.transAxes,color=txt_color)
    ax.text(0.05,0.91,"Interactive - Arts & Advanced Big Data",fontsize=11,transform=ax.transAxes,color=txt_color)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    return fig

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Custom Color (optional)")
    use_custom = st.checkbox("Use custom color palette", value=False)
    picked_hex = st.color_picker("Pick a base color", "#ff88aa")
    custom_cols = custom_palette_from_hex(picked_hex, k=7) if use_custom else None
    st.caption("Palette preview")
    preview_cols = custom_cols if use_custom else get_palette("Pastel", 7)
    st.pyplot(plt.figure(), use_container_width=True)
    st.header("Controls")
    shape = st.selectbox("Shape Type", ["Blob","Polygon","Waves"])
    layers = st.slider("Number of Layers",1,25,10,1)
    wobble = st.slider("Wobble Intensity",0.01,0.6,0.18,0.01)
    palette_kind = st.selectbox("Palette",["Pastel","Vibrant","Mono","Random","Pink","Blue","Green"]) if not use_custom else "Custom Color"
    bg_mode = st.selectbox("Background",["Off-white","Dark"])
    if "reroll" not in st.session_state: st.session_state.reroll=0
    if st.button("Randomize Now"): st.session_state.reroll+=1
    seed_in = st.text_input("Seed (optional, int)",value="")

# ---------- Render ----------
seed_val = None if seed_in.strip()=="" else int(seed_in)
fig = draw_poster(shape, layers, wobble, palette_kind, bg_mode, seed=seed_val, palette_override=custom_cols)
st.pyplot(fig, use_container_width=True)
