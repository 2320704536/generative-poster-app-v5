# ============================================================
# Emotional Crystal × Designer Poster v6.0 (FULL INTEGRATED VERSION)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import time
import io

from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from colorsys import rgb_to_hsv, hsv_to_rgb

st.set_page_config(page_title="Emotional Crystal Suite v6.0", layout="wide")

# ============================================================
# 0. MODE SELECTOR
# ============================================================
st.title("🎨 Emotional Crystal Suite v6.0")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Emotional Crystal", "Designer Poster"],
    index=0
)

# ============================================================
# 1. Shared Helpers
# ============================================================

def clamp01(x):
    return max(0.0, min(1.0, x))

def hex_to_rgb01(hex_str):
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return r, g, b

def rgb01_to_rgb255(c):
    return (int(c[0]*255), int(c[1]*255), int(c[2]*255))

def rgb255_to_rgb01(c):
    return (c[0]/255, c[1]/255, c[2]/255)

def srgb_to_linear(x):
    x = np.clip(x,0,1)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

def linear_to_srgb(x):
    x = np.clip(x,0,1)
    return np.where(x < 0.0031308, x*12.92,
                    1.055*(x**(1/2.4))-0.055)

# ============================================================
# 2. CSV Palette Import / Export
# ============================================================

def import_palette_csv(file):
    try:
        dfc = pd.read_csv(file)
        required = {"emotion", "r", "g", "b"}
        low = {c.lower(): c for c in dfc.columns}
        if not required.issubset(low.keys()):
            st.error("CSV must contain emotion,r,g,b columns.")
            return None

        pal = {}
        for _,row in dfc.iterrows():
            try:
                emo = str(row[low["emotion"]]).strip()
                r = int(row[low["r"]])
                g = int(row[low["g"]])
                b = int(row[low["b"]])
                pal[emo]=(r,g,b)
            except:
                continue
        st.success(f"Imported {len(pal)} colors from CSV.")
        return pal
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def export_palette_csv(pal):
    buf = BytesIO()
    df = pd.DataFrame([
        {"emotion":k, "r":v[0], "g":v[1], "b":v[2]} for k,v in pal.items()
    ])
    df.to_csv(buf,index=False)
    buf.seek(0)
    return buf

# ============================================================
# 3. MODE 1: EMOTIONAL CRYSTAL (PART 1)
# ============================================================

if mode == "Emotional Crystal":
    st.header("💎 Emotional Crystal Mode")

    # ========= CSV Palette Upload =========
    st.subheader("Palette (CSV Only Mode Available)")

    if "crystal_palette" not in st.session_state:
        st.session_state["crystal_palette"] = {}

    use_csv_only = st.checkbox(
        "Use CSV palette only (ignore internal colors)",
        value=False
    )

    uploaded_csv = st.file_uploader("Upload CSV Palette", type=["csv"])
    if uploaded_csv:
        pal = import_palette_csv(uploaded_csv)
        if pal is not None:
            st.session_state["crystal_palette"] = pal

    # ========= Crystal Parameters =========
    st.subheader("Crystal Parameters")

    seed = st.slider("Seed", 0, 99999, 12345)
    layers = st.slider("Crystal Layers", 1, 30, 8)
    wobble = st.slider("Crystal Wobble", 0.0, 0.8, 0.25)
    shapes_per_emotion = st.slider("Shapes per Emotion", 1, 40, 10)
    min_size = st.slider("Min Crystal Size", 20, 300, 70)
    max_size = st.slider("Max Crystal Size", 40, 600, 220)

    bg_hex = st.color_picker("Background Color", "#000000")
    bg_rgb = tuple(int(bg_hex[i:i+2],16) for i in (1,3,5))

    st.markdown("---")

    # ========= Fake Emotion Data (for demo) =========
    st.subheader("Input Text (Demo)")

    user_text = st.text_area(
        "Enter lines of text (each line = one crystal emotion item)",
        "Sky was glowing.\nMarket anxiety remains.\nA moment of awe.\nHope rises again."
    )

    lines = [t.strip() for t in user_text.split("\n") if t.strip()]
    if len(lines) == 0:
        st.warning("Please enter at least one line.")
        st.stop()

    # Fake emotion mapping
    EMOS = []
    for i, txt in enumerate(lines):
        EMOS.append(f"emo_{i+1}")

    df = pd.DataFrame({"text": lines, "emotion": EMOS})

    # ========= Active Palette =========
    if use_csv_only:
        palette = dict(st.session_state["crystal_palette"])
        if len(palette) == 0:
            st.error("CSV-only mode requires an imported palette.")
            st.stop()

        # force df emotions to rotate through palette keys
        keys = list(palette.keys())
        df["emotion"] = [keys[i % len(keys)] for i in range(len(df))]

    else:
        # fallback — generate random pastel palette
        base = {}
        for emo in df["emotion"].unique():
            base[emo] = (
                random.randint(60,255),
                random.randint(60,255),
                random.randint(60,255)
            )
        palette = base

    # Store palette for export
    out_csv = export_palette_csv(palette)

    # Button to download palette
    st.download_button(
        "Download Current Palette CSV",
        data=out_csv.getvalue(),
        file_name="crystal_palette.csv",
        mime="text/csv"
    )

# ============================================================
# 3. MODE 1: EMOTIONAL CRYSTAL (PART 2 — Rendering Engine)
# ============================================================

    st.markdown("## Crystal Rendering Engine")

    # ---------- Crystal Shape ----------
    def crystal_shape(center=(0.5,0.5), r=150, wobble=0.25, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        cx, cy = center
        n_vertices = int(rng.integers(5, 11))
        angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
        rng.shuffle(angles)
        radii = r * (1 + rng.uniform(-wobble, wobble, size=n_vertices))
        pts = []
        for a, rr in zip(angles, radii):
            x = cx + rr * math.cos(a)
            y = cy + rr * math.sin(a)
            pts.append((float(x), float(y)))
        pts.append(pts[0])
        return pts

    # ---------- Soft Drawing ----------
    def draw_polygon_soft(canvas_rgba, pts, rgb01, alpha=200, blur_px=6, edge_width=0):
        W, H = canvas_rgba.size
        layer = Image.new("RGBA", (W,H), (0,0,0,0))
        d = ImageDraw.Draw(layer, "RGBA")

        col = (
            int(rgb01[0]*255),
            int(rgb01[1]*255),
            int(rgb01[2]*255),
            alpha
        )
        d.polygon(pts, fill=col)

        if edge_width > 0:
            edge = (255,255,255,max(80,alpha//2))
            d.line(pts, fill=edge, width=edge_width, joint="curve")

        if blur_px > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(blur_px))

        canvas_rgba.alpha_composite(layer)

    # ---------- Color Helpers ----------
    def jitter_color(rgb01, rng, amount=0.05):
        j = (rng.random(3)-0.5)*2*amount
        return tuple(np.clip(np.array(rgb01)+j,0,1))

    # ---------- Render Crystal Mix ----------
    def render_crystal(df, palette, width=1500, height=850,
                       seed=12345, layers=10, wobble=0.25,
                       shapes_per_emotion=10, min_size=60, max_size=220,
                       bg_color=(0,0,0)):

        rng = np.random.default_rng(seed)
        base = Image.new("RGBA", (width,height),
                         (bg_color[0],bg_color[1],bg_color[2],255))
        canvas = Image.new("RGBA", (width,height),(0,0,0,0))

        emos = df["emotion"].tolist()

        for _layer in range(layers):
            for emo in emos:
                if emo not in palette:
                    continue
                rgb = palette[emo]
                rgb01 = rgb255_to_rgb01(rgb)

                for _ in range(shapes_per_emotion):
                    cx = rng.uniform(0.05*width, 0.95*width)
                    cy = rng.uniform(0.05*height,0.95*height)
                    rr = int(rng.uniform(min_size, max_size))

                    pts = crystal_shape(
                        center=(cx,cy), r=rr,
                        wobble=wobble, rng=rng
                    )

                    col01 = jitter_color(rgb01, rng, amount=0.07)
                    local_alpha = int(rng.uniform(120,230))
                    local_blur = int(rng.uniform(3,10))
                    edge = 0 if rng.random()<0.6 else max(1,int(rr*0.02))

                    draw_polygon_soft(
                        canvas, pts, col01,
                        alpha=local_alpha,
                        blur_px=local_blur,
                        edge_width=edge
                    )

        base.alpha_composite(canvas)
        return base.convert("RGB")

    # ============================================================
    #  CINEMATIC COLOR PIPELINE
    # ============================================================

    def highlight_rolloff(lin, roll):
        threshold = 0.8
        mask = np.clip((lin-threshold)/(1e-6 + 1.0-threshold),0,1)
        out = lin*(1-mask) + (threshold + (lin-threshold)/(1.0+4*roll*mask))*mask
        return np.clip(out,0,1)

    def adjust_contrast(img,c):
        return np.clip((img-0.5)*c+0.5,0,1)

    def adjust_saturation(img,s):
        lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
        lum = lum[...,None]
        return np.clip(lum + (img-lum)*s,0,1)

    def gamma_correct(img,g):
        return np.clip(img**(1.0/g),0,1)

    def apply_bloom(img,radius,intensity):
        pil = Image.fromarray((img*255).astype(np.uint8))
        if radius>0:
            blurred = pil.filter(ImageFilter.GaussianBlur(radius))
            b = np.array(blurred)/255.0
            return np.clip(img*(1-intensity)+b*intensity,0,1)
        return img

    def apply_vignette(img,strength):
        h,w,_ = img.shape
        yy,xx=np.mgrid[0:h,0:w]
        xx=(xx-w/2)/(w/2); yy=(yy-h/2)/(h/2)
        r = np.sqrt(xx*xx+yy*yy)
        mask = np.clip(1-strength*(r**1.5),0,1)
        return img*mask[...,None]

    # ------- Color controls -------
    exp = st.slider("Exposure", -0.2, 1.5, 0.5)
    contrast_val = st.slider("Contrast", 0.6, 1.8, 1.2)
    saturation_val = st.slider("Saturation", 0.6, 1.8, 1.2)
    gamma_val = st.slider("Gamma", 0.7, 1.4, 1.0)
    bloom_radius = st.slider("Bloom Radius", 0.0, 20.0, 6.0)
    bloom_intensity = st.slider("Bloom Intensity", 0.0, 1.0, 0.5)
    vignette_strength = st.slider("Vignette", 0.0, 0.6, 0.25)

    st.markdown("### Generate Crystal Image")

    # ============================================================
    #  FINAL CRYSTAL GENERATION
    # ============================================================
    img = render_crystal(
        df=df, palette=palette,
        width=1500, height=900,
        seed=seed, layers=layers,
        wobble=wobble, shapes_per_emotion=shapes_per_emotion,
        min_size=min_size, max_size=max_size,
        bg_color=bg_rgb
    )

    arr = np.array(img).astype(np.float32)/255.0

    # exposure
    lin = srgb_to_linear(arr)
    lin *= (2**exp)
    lin = highlight_rolloff(lin,0.4)
    arr = linear_to_srgb(lin)
    arr = np.clip(arr,0,1)

    # contrast/sat/gamma
    arr = adjust_contrast(arr,contrast_val)
    arr = adjust_saturation(arr,saturation_val)
    arr = gamma_correct(arr,gamma_val)

    # bloom/vignette
    arr = apply_bloom(arr,bloom_radius,bloom_intensity)
    arr = apply_vignette(arr,vignette_strength)

    final_img = Image.fromarray((arr*255).astype(np.uint8))

    st.image(final_img, use_container_width=True)

    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    st.download_button(
        "💾 Download Crystal PNG",
        data=buf.getvalue(),
        file_name="emotional_crystal.png",
        mime="image/png"
    )
# ============================================================
# 4. MODE 2: DESIGNER POSTER v6.0 (FULL)
# ============================================================

if mode == "Designer Poster":
    st.header("🖼️ Designer Poster v6.0")

    # ----------------------------------------------------------
    #  CSV Palette Upload
    # ----------------------------------------------------------
    st.subheader("Palette Options (CSV or Built-in)")

    if "poster_palette" not in st.session_state:
        st.session_state.poster_palette = None

    use_custom_csv = st.checkbox("Use CSV palette for Poster", value=False)

    csv_file = st.file_uploader("Upload CSV Palette", type=["csv"])
    if csv_file:
        pal = import_palette_csv(csv_file)
        if pal is not None:
            st.session_state.poster_palette = {
                k: rgb255_to_rgb01(v) for k, v in pal.items()
            }
            st.success("Poster CSV palette loaded.")

    # ----------------------------------------------------------
    # Shape functions
    # ----------------------------------------------------------
    def blob(center=(0.5,0.5), r=0.3, points=200, wobble=0.15):
        ang=np.linspace(0,2*math.pi,points)
        rad=r*(1+wobble*(np.random.rand(points)-0.5))
        return center[0]+rad*np.cos(ang), center[1]+rad*np.sin(ang)

    def polygon(center=(0.5,0.5), sides=6, r=0.3, wobble=0.1):
        ang=np.linspace(0,2*math.pi,sides,endpoint=False)
        rad=r*(1+wobble*(np.random.rand(sides)-0.5))
        x=center[0]+rad*np.cos(ang)
        y=center[1]+rad*np.sin(ang)
        return np.append(x,x[0]), np.append(y,y[0])

    def waves(center=(0.5,0.5), r=0.3, points=400, frequency=6, wobble=0.05):
        ang=np.linspace(0,2*math.pi,points)
        rad=r*(1+wobble*np.sin(frequency*ang))
        return center[0]+rad*np.cos(ang), center[1]+rad*np.sin(ang)

    def rings(center=(0.5,0.5), base_r=0.3, count=4, wobble=0.1):
        return [blob(center, base_r*(0.5+i*0.4), 200, wobble) for i in range(count)]

    def star(center=(0.5,0.5), points=5, r1=0.3, r2=0.15):
        ang=np.linspace(0,2*math.pi,points*2,endpoint=False)
        rad=np.array([r1 if i%2==0 else r2 for i in range(points*2)])
        x=center[0]+rad*np.cos(ang)
        y=center[1]+rad*np.sin(ang)
        return np.append(x,x[0]), np.append(y,y[0])

    def spiral(center=(0.5,0.5),turns=3,points=500,r=0.4):
        t=np.linspace(0,2*math.pi*turns,points)
        rad=np.linspace(0.01,r,points)
        return center[0]+rad*np.cos(t), center[1]+rad*np.sin(t)

    def cloud(center=(0.5,0.5),r=0.3,blobs=6):
        coords=[]
        for i in range(blobs):
            ang=random.uniform(0,2*math.pi)
            rr=r*random.uniform(0.6,1.2)
            cx=center[0]+r*0.6*math.cos(ang)
            cy=center[1]+r*0.6*math.sin(ang)
            x,y=blob((cx,cy),rr*0.4,100,0.3)
            coords.append((x,y))
        return coords

    # ----------------------------------------------------------
    # Background
    # ----------------------------------------------------------
    def set_background(ax, bg_mode):
        if bg_mode=="Off-white":
            ax.set_facecolor((0.98,0.98,0.97))
            return "dark"
        if bg_mode=="Light gray":
            ax.set_facecolor((0.92,0.92,0.92))
            return "dark"
        if bg_mode=="Dark":
            ax.set_facecolor((0.08,0.08,0.08))
            return "light"
        if bg_mode=="Gradient":
            grad=np.linspace(0.95,0.75,512).reshape(-1,1)
            ax.imshow(np.dstack([grad,grad,grad]),
                      extent=[0,1,0,1],
                      origin="lower",zorder=-10)
            ax.set_facecolor((1,1,1,0))
            return "dark"
        ax.set_facecolor((1,1,1))
        return "dark"

    # ----------------------------------------------------------
    # Poster Palette (built-in fallback)
    # ----------------------------------------------------------
    def random_palette(k=6):
        return [(random.random(),random.random(),random.random()) for _ in range(k)]

    def pastel_palette(k=6):
        return [(random.uniform(0.65,0.95),
                 random.uniform(0.65,0.95),
                 random.uniform(0.65,0.95))
                for _ in range(k)]

    def vibrant_palette(k=6):
        anchors=[
            (0.95,0.30,0.30),(1.00,0.65,0.00),
            (0.20,0.70,0.30),(0.20,0.40,0.95),
            (0.65,0.25,0.90),(0.95,0.20,0.60)
        ]
        cols=[]
        for i in range(k):
            r,g,b=anchors[i%len(anchors)]
            cols.append((clamp01(r+random.uniform(-0.05,0.05)),
                         clamp01(g+random.uniform(-0.05,0.05)),
                         clamp01(b+random.uniform(-0.05,0.05))))
        return cols

    def mono_palette(k=6):
        h=random.random()
        vals=np.linspace(0.35,0.95,k)
        return [hsv_to_rgb(h,0.4,v) for v in vals]

    builtin_map={
        "Pastel":pastel_palette,
        "Vibrant":vibrant_palette,
        "Mono":mono_palette,
        "Random":random_palette
    }

    # ----------------------------------------------------------
    # POSTER UI
    # ----------------------------------------------------------
    st.subheader("Poster Controls")

    shape = st.selectbox("Shape Type",[
        "Blob","Polygon","Waves","Rings","Star","Spiral","Cloud"
    ])

    layers = st.slider("Layers",1,25,10)
    wobble = st.slider("Wobble",0.01,0.6,0.18)
    bg_mode = st.selectbox("Background",["Off-white","Light gray","Dark","Gradient"])
    aspect = st.selectbox("Aspect Ratio",["Portrait","Landscape","Square"])

    if use_custom_csv and st.session_state.poster_palette:
        palette = list(st.session_state.poster_palette.values())
    else:
        palette_kind = st.selectbox("Built-in Palette", list(builtin_map.keys()))
        palette = builtin_map

    # Preview palette
    st.write("Palette Preview:")
    fig_prev, ax_prev = plt.subplots(figsize=(3,0.5))
    ax_prev.axis("off")
    n=len(palette)
    for i,c in enumerate(palette):
        ax_prev.add_patch(plt.Rectangle((i/n,0),1/n,1,color=c,ec=(0,0,0,0)))
    st.pyplot(fig_prev)

    # ----------------------------------------------------------
    # Poster Rendering
    # ----------------------------------------------------------
    def draw_poster(shape, layers, wobble, palette, bg_mode, aspect):
        if aspect=="Portrait":
            fig,ax=plt.subplots(figsize=(7,10))
        elif aspect=="Landscape":
            fig,ax=plt.subplots(figsize=(10,7))
        else:
            fig,ax=plt.subplots(figsize=(8,8))

        ax.axis("off")
        text_mode=set_background(ax,bg_mode)

        for _ in range(layers):
            cx,cy=random.random(),random.random()
            rr=random.uniform(0.15,0.45)
            col=random.choice(palette)
            alpha=random.uniform(0.25,0.6)

            if shape=="Blob":
                x,y=blob((cx,cy),rr,wobble=wobble)
                ax.fill(x,y,color=col,alpha=alpha,edgecolor=(0,0,0,0))
            elif shape=="Polygon":
                x,y=polygon((cx,cy),random.randint(3,8),rr,wobble)
                ax.fill(x,y,color=col,alpha=alpha,edgecolor=(0,0,0,0))
            elif shape=="Waves":
                x,y=waves((cx,cy),rr,frequency=random.randint(4,8),wobble=wobble)
                ax.fill(x,y,color=col,alpha=alpha,edgecolor=(0,0,0,0))
            elif shape=="Rings":
                for x,y in rings((cx,cy),rr,count=random.randint(2,4),wobble=wobble):
                    ax.plot(x,y,color=col,alpha=alpha,lw=2)
            elif shape=="Star":
                x,y=star((cx,cy),points=random.randint(5,8),r1=rr,r2=rr*0.5)
                ax.fill(x,y,color=col,alpha=alpha,edgecolor=(0,0,0,0))
            elif shape=="Spiral":
                x,y=spiral((cx,cy),turns=random.randint(2,4),r=rr)
                ax.plot(x,y,color=col,alpha=alpha,lw=2)
            elif shape=="Cloud":
                for x,y in cloud((cx,cy),rr):
                    ax.fill(x,y,color=col,alpha=alpha,edgecolor=(0,0,0,0))

        txt_color=(0.95,0.95,0.95) if text_mode=="light" else (0.1,0.1,0.1)
        ax.text(0.05,0.95,"Generative Poster",fontsize=18,weight="bold",
                transform=ax.transAxes,color=txt_color)
        ax.text(0.05,0.91,"Interactive Poster Engine",fontsize=11,
                transform=ax.transAxes,color=txt_color)
        ax.set_xlim(0,1); ax.set_ylim(0,1)

        return fig

    st.subheader("Render Poster")
    fig = draw_poster(shape, layers, wobble, palette, bg_mode, aspect)

    st.pyplot(fig)

    # ----------------------------------------------------------
    # Export Options
    # ----------------------------------------------------------
    st.subheader("Export")

    dpi = st.slider("PNG DPI",72,600,300)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=dpi, bbox_inches="tight")
    st.download_button(
        "Download PNG",
        data=buf_png.getvalue(),
        file_name="poster_v6.png",
        mime="image/png"
    )

    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches="tight")
    st.download_button(
        "Download SVG",
        data=buf_svg.getvalue(),
        file_name="poster_v6.svg",
        mime="image/svg+xml"
    )
