import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from colorsys import rgb_to_hsv, hsv_to_rgb
from PIL import Image, ImageEnhance, ImageFilter

st.set_page_config(page_title="Generative Poster v5.2", layout="wide")
st.title("Generative Abstract Poster v5.2")
st.markdown("Interactive - Arts & Advanced Big Data  \nNow with Autoplay, Hi-Res & SVG export, Film Grain, and cleaner outputs")

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
        cols.append((clamp01(r+random.uniform(-0.05,0.05)),
                     clamp01(g+random.uniform(-0.05,0.05)),
                     clamp01(b+random.uniform(-0.05,0.05))))
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

def rings(center=(0.5,0.5), base_r=0.3, count=4, wobble=0.1):
    return [blob(center, base_r*(0.5+i*0.4), 200, wobble) for i in range(count)]

def star(center=(0.5,0.5), points=5, r1=0.3, r2=0.15):
    ang = np.linspace(0, 2*math.pi, points*2, endpoint=False)
    rad = np.array([r1 if i%2==0 else r2 for i in range(points*2)])
    x = center[0] + rad*np.cos(ang); y = center[1] + rad*np.sin(ang)
    return np.append(x, x[0]), np.append(y, y[0])

def spiral(center=(0.5,0.5), turns=3, points=500, r=0.4):
    t = np.linspace(0, 2*math.pi*turns, points)
    rad = np.linspace(0.01, r, points)
    return center[0]+rad*np.cos(t), center[1]+rad*np.sin(t)

def cloud(center=(0.5,0.5), r=0.3, blobs=6):
    coords = []
    for i in range(blobs):
        ang = random.uniform(0, 2*math.pi); rr = r*random.uniform(0.6, 1.2)
        cx = center[0] + r*0.6*math.cos(ang); cy = center[1] + r*0.6*math.sin(ang)
        x, y = blob((cx,cy), rr*0.4, points=100, wobble=0.3)
        coords.append((x,y))
    return coords

# ---------- Background ----------
def set_background(ax, mode):
    if mode == "Off-white":
        ax.set_facecolor((0.98, 0.98, 0.97)); return "dark"
    if mode == "Light gray":
        ax.set_facecolor((0.92, 0.92, 0.92)); return "dark"
    if mode == "Dark":
        ax.set_facecolor((0.08, 0.08, 0.08)); return "light"
    if mode == "Gradient":
        grad = np.linspace(0.95, 0.75, 512).reshape(-1,1)
        _ = ax.imshow(np.dstack([grad,grad,grad]), extent=[0,1,0,1], origin="lower", zorder=-10)
        ax.set_facecolor((1,1,1,0)); return "dark"
    ax.set_facecolor((1,1,1)); return "dark"

# ---------- Draw Poster ----------
def draw_poster(shape="Blob", layers=8, wobble=0.15, palette_kind="Pastel", bg="Off-white",
                seed=None, alpha_min=0.25, alpha_max=0.6, r_min=0.15, r_max=0.45,
                palette_override=None, aspect="Portrait"):
    if seed not in (None, "", 0):
        try:
            seed = int(seed); random.seed(seed); np.random.seed(seed)
        except:
            pass
    if aspect == "Portrait":
        fig, ax = plt.subplots(figsize=(7, 10))
    elif aspect == "Landscape":
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.axis("off")
    text_mode = set_background(ax, bg)
    cols = palette_override if palette_override else get_palette(palette_kind, 7)

    for _ in range(layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(r_min, r_max)
        color = random.choice(cols)
        alpha = random.uniform(alpha_min, alpha_max)

        if shape == "Blob":
            x, y = blob((cx, cy), rr, wobble=wobble)
            _ = ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))
        elif shape == "Polygon":
            x, y = polygon((cx, cy), sides=random.randint(3,8), r=rr, wobble=wobble)
            _ = ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))
        elif shape == "Waves":
            x, y = waves((cx, cy), rr, frequency=random.randint(4,8), wobble=wobble)
            _ = ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))
        elif shape == "Rings":
            for x, y in rings((cx, cy), rr, count=random.randint(2,4), wobble=wobble):
                _ = ax.plot(x, y, color=color, alpha=alpha, lw=2)
        elif shape == "Star":
            x, y = star((cx, cy), points=random.randint(5,8), r1=rr, r2=rr*0.5)
            _ = ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))
        elif shape == "Spiral":
            x, y = spiral((cx, cy), turns=random.randint(2,4), r=rr)
            _ = ax.plot(x, y, color=color, alpha=alpha, lw=2)
        elif shape == "Cloud":
            for x, y in cloud((cx, cy), rr):
                _ = ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))

    txt_color = (0.95,0.95,0.95) if text_mode == "light" else (0.1,0.1,0.1)
    _ = ax.text(0.05, 0.95, "Generative Poster", fontsize=18, weight="bold", transform=ax.transAxes, color=txt_color)
    _ = ax.text(0.05, 0.91, "Interactive - Arts & Advanced Big Data", fontsize=11, transform=ax.transAxes, color=txt_color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig

def fig_to_pil(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img, buf

def render_palette_preview(colors):
    fig, ax = plt.subplots(figsize=(3.5, 0.5))
    ax.axis("off")
    n = len(colors)
    for i, c in enumerate(colors):
        _ = ax.add_patch(plt.Rectangle((i/n, 0), 1/n, 1, color=c, ec=(0,0,0,0)))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    return fig

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Custom Color (optional)")
    use_custom = st.checkbox("Use custom color palette", value=False)
    picked_hex = st.color_picker("Pick a base color", "#ff88aa")
    custom_cols = custom_palette_from_hex(picked_hex, k=7) if use_custom else None
    st.caption("If enabled, the app hides palette dropdown and uses your custom gradient palette.")

    st.caption("Palette preview")
    preview_cols = custom_cols if use_custom else get_palette("Pastel", 7)
    prev_fig = render_palette_preview(preview_cols)
    st.pyplot(prev_fig, use_container_width=True)

    st.header("Controls")
    shape = st.selectbox("Shape Type", ["Blob", "Polygon", "Waves", "Rings", "Star", "Spiral", "Cloud"])
    layers = st.slider("Number of Layers", 1, 25, 10, 1)
    wobble = st.slider("Wobble Intensity", 0.01, 0.6, 0.18, 0.01)

    if not use_custom:
        palette_kind = st.selectbox("Palette", ["Pastel","Vibrant","Mono","Random","Pink","Blue","Green"])
    else:
        palette_kind = "Custom Color"

    bg_mode = st.selectbox("Background", ["Off-white","Light gray","Dark","Gradient"])
    aspect = st.selectbox("Aspect Ratio", ["Portrait","Landscape","Square"])

    st.subheader("Visual Controls")
    alpha_min = st.slider("Alpha Min", 0.0, 1.0, 0.25, 0.01)
    alpha_max = st.slider("Alpha Max", 0.0, 1.0, 0.60, 0.01)
    r_min = st.slider("Shape Size Min", 0.05, 0.60, 0.15, 0.01)
    r_max = st.slider("Shape Size Max", 0.05, 0.60, 0.45, 0.01)

    st.subheader("Filters")
    brightness = st.slider("Brightness", 0.5, 1.5, 1.00, 0.01)
    contrast   = st.slider("Contrast",   0.5, 1.5, 1.00, 0.01)
    blur_amt   = st.slider("Blur",       0.0, 5.0, 0.00, 0.1)
    grain_amt  = st.slider("Film Grain", 0.0, 1.0, 0.00, 0.05)

    st.subheader("Autoplay")
    autoplay = st.checkbox("Enable autoplay", value=False)
    interval = st.slider("Interval (seconds)", 1, 20, 5, 1)
    if "last_tick" not in st.session_state: st.session_state.last_tick = time.time()
    if "reroll" not in st.session_state: st.session_state.reroll = 0
    if st.button("Randomize Now"): st.session_state.reroll += 1

    seed_in = st.text_input("Seed (optional, int)", value="")

# ---------- Autoplay logic ----------
if autoplay:
    now = time.time()
    if now - st.session_state.last_tick >= interval:
        st.session_state.reroll += 1
        st.session_state.last_tick = now
        # trigger a rerun
        st.experimental_rerun()

# ---------- Seed combine ----------
if seed_in.strip() == "":
    effective_seed = st.session_state.reroll if st.session_state.reroll > 0 else None
else:
    try:
        base = int(seed_in)
    except:
        base = None
    effective_seed = None if base is None else base + st.session_state.reroll

# ---------- Render and post-process filters ----------
fig = draw_poster(shape, layers, wobble, palette_kind, bg_mode, seed=effective_seed,
                  alpha_min=alpha_min, alpha_max=alpha_max, r_min=r_min, r_max=r_max,
                  palette_override=custom_cols, aspect=aspect)

# Export options: DPI slider before rasterization
st.subheader("Export Options")
dpi_val = st.slider("PNG DPI (resolution)", 72, 600, 300, 10)

# Convert figure to PIL at chosen DPI
pil_img, raw_buf = fig_to_pil(fig, dpi=dpi_val)

# Apply filters using PIL
if abs(brightness - 1.0) > 1e-6:
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
if abs(contrast - 1.0) > 1e-6:
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
if blur_amt > 0:
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_amt))
if grain_amt > 0:
    # film grain: add noise layer and blend
    w, h = pil_img.size
    noise = np.random.normal(0.5, 0.15, (h, w)).clip(0,1)
    noise_img = Image.fromarray((noise*255).astype("uint8"), mode="L").convert("RGBA")
    pil_img = Image.blend(pil_img, noise_img, alpha=grain_amt)

# Show final image
st.image(pil_img, use_container_width=True)

# Downloads
out_buf = io.BytesIO()
pil_img.save(out_buf, format="PNG")
st.download_button("Download Final PNG", data=out_buf.getvalue(), file_name="poster_v5.2.png", mime="image/png")

# SVG (raw Matplotlib, filters not applied)
svg_buf = io.BytesIO()
fig.savefig(svg_buf, format="svg", bbox_inches="tight")
st.download_button("Download Raw SVG (no filters)", data=svg_buf.getvalue(), file_name="poster_v5.2.svg", mime="image/svg+xml")

st.markdown("---")
st.caption(f"Palette: {('Custom ' + picked_hex) if use_custom else palette_kind} | Background: {bg_mode} | Shape: {shape} | Layers: {layers} | Wobble: {wobble:.2f} | Alpha: {alpha_min:.2f}-{alpha_max:.2f} | Size: {r_min:.2f}-{r_max:.2f} | Aspect: {aspect} | Brightness: {brightness:.2f} | Contrast: {contrast:.2f} | Blur: {blur_amt:.2f} | Grain: {grain_amt:.2f} | DPI: {dpi_val} | Reroll: {st.session_state.reroll}")
