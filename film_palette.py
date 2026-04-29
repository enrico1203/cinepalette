#!/usr/bin/env python3
"""
film_palette.py — Pipeline completa di analisi colori per un film.

• V1: estrae 1 frame ogni 15s → 4 palette (15s/30s/45s/60s) come strisce verticali
• V2: estrae 20 frame equidistanti scegliendo il più nitido (Tenengrad su 61 frame
      consecutivi) → 20 card con frame + 5 colori dominanti (hex + %)

Uso:
    python3 film_palette.py
    python3 film_palette.py --film "/path/to/film.mkv" --output "/path/to/output"

Requisiti:
    pip install Pillow scikit-learn opencv-python-headless
    (ffmpeg e ffprobe devono essere nel PATH)
"""

import argparse
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

# ─── Costanti configurabili ───────────────────────────────────────────────────

# V1
V1_INTERVAL   = 15     # secondi tra un frame e l'altro
V1_WORKERS    = 8      # thread paralleli per ffmpeg
V1_STRIPE_W   = 2      # px per striscia di colore
V1_HEIGHT     = 400    # altezza palette px

# V2
V2_N_FRAMES          = 20     # frame equidistanti da estrarre
V2_N_COLORS          = 5      # colori per card
V2_MIN_COLOR_DIST    = 40     # distanza minima tra colori selezionati
V2_FRAMES_EACH_SIDE  = 30     # frame reali prima/dopo per la selezione nitidezza
V2_SWATCH_H          = 220
V2_PADDING           = 50
V2_TEXT_AREA         = 70
V2_FONT_SIZE_HEX     = 28
V2_FONT_SIZE_PCT     = 22
V2_BG_COLOR          = (255, 255, 255)


# ─── Rilevamento metadati film ────────────────────────────────────────────────

def probe_video(film_path: Path) -> dict:
    """Restituisce duration, width, height, fps usando ffprobe."""
    # Durata dal container
    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1",
        str(film_path),
    ], capture_output=True, text=True)
    duration = float(r.stdout.strip().split("=")[-1])

    # Dimensioni e fps dal primo stream video
    r2 = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1",
        str(film_path),
    ], capture_output=True, text=True)

    info = {}
    for line in r2.stdout.strip().splitlines():
        k, v = line.split("=", 1)
        info[k] = v

    width  = int(info["width"])
    height = int(info["height"])

    # r_frame_rate è una frazione tipo "24000/1001"
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    # cropdetect: rimuove bande nere
    r3 = subprocess.run([
        "ffmpeg", "-ss", "600", "-i", str(film_path),
        "-t", "5", "-vf", "cropdetect=24:16:0", "-f", "null", "-",
        "-hide_banner",
    ], capture_output=True, text=True)
    crop_h = height
    for line in r3.stderr.splitlines():
        if "crop=" in line:
            crop_part = line.split("crop=")[-1].split()[0]
            parts = crop_part.split(":")
            if len(parts) >= 2:
                crop_h = int(parts[1])

    return {
        "duration": int(duration),
        "width":    width,
        "height":   crop_h,   # altezza senza bande nere
        "fps":      fps,
    }


# ─── Utilità comuni ───────────────────────────────────────────────────────────

def try_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rgb_to_hex(r, g, b) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


# ─── V1 ───────────────────────────────────────────────────────────────────────

def v1_extract(args):
    i, t, film_path, frames_dir = args
    out = frames_dir / f"frame_{i+1:04d}.jpg"
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(t), "-i", str(film_path),
        "-vframes", "1", "-vf", "scale=480:-2", "-q:v", "4",
        str(out), "-hide_banner", "-loglevel", "error",
    ], capture_output=True)
    return i, t


def v1_dominant_color(img_path: Path) -> tuple:
    with Image.open(img_path) as img:
        img = img.resize((200, 112), Image.LANCZOS).convert("RGB")
        q   = img.quantize(colors=5, method=Image.Quantize.FASTOCTREE)
        pal = q.getpalette()
        idx = max(range(5), key=lambda i: q.histogram()[i])
        return (pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2])


def v1_build_palette(frames, path: Path, label: str):
    colors = [v1_dominant_color(f) for f in frames]
    img = Image.new("RGB", (len(colors) * V1_STRIPE_W, V1_HEIGHT))
    px  = img.load()
    for i, c in enumerate(colors):
        for x in range(i * V1_STRIPE_W, i * V1_STRIPE_W + V1_STRIPE_W):
            for y in range(V1_HEIGHT):
                px[x, y] = c
    img.save(path, "PNG")
    print(f"    ✓ {path.name}  →  {len(colors)} strisce  ({len(colors)*V1_STRIPE_W}×{V1_HEIGHT}px)")


def run_v1(film_path: Path, output_dir: Path, meta: dict):
    print("\n─── V1: palette strisce ─────────────────────────────────────────")
    frames_dir = output_dir / "V1" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration   = meta["duration"]
    timestamps = list(range(0, duration, V1_INTERVAL))
    total      = len(timestamps)
    print(f"  Frame da estrarre: {total}  (1 ogni {V1_INTERVAL}s)")

    start = time.time()
    done  = 0
    args_list = [(i, t, film_path, frames_dir) for i, t in enumerate(timestamps)]

    with ThreadPoolExecutor(max_workers=V1_WORKERS) as ex:
        futs = {ex.submit(v1_extract, a): a[0] for a in args_list}
        for fut in as_completed(futs):
            done += 1
            elapsed = time.time() - start
            eta = (total - done) / (done / elapsed) if done > 0 else 0
            print(f"  {done}/{total}  ETA {eta:.0f}s", end="\r", flush=True)

    print(f"\n  Estrazione completata in {time.time()-start:.0f}s — genero palette...")

    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    out = output_dir / "V1"
    v1_build_palette(all_frames[::1], out / "palette_15s.png", "15s")
    v1_build_palette(all_frames[::2], out / "palette_30s.png", "30s")
    v1_build_palette(all_frames[::3], out / "palette_45s.png", "45s")
    v1_build_palette(all_frames[::4], out / "palette_60s.png", "60s")
    print(f"  V1 completato → {out}")


# ─── V2 ───────────────────────────────────────────────────────────────────────

def tenengrad(img_path: Path) -> float:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    score = float(np.mean(gx**2 + gy**2))
    brightness = float(img.mean())
    if brightness < 20:
        score *= (brightness / 20)
    return score


def v2_extract_frame(idx: int, timestamp: int, film_path: Path,
                     frames_dir: Path, fps: float, frame_w: int):
    tmp_dir = frames_dir / f"_tmp_{idx:02d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    clip_duration = (V2_FRAMES_EACH_SIDE * 2 + 1) / fps
    clip_start    = max(0.0, timestamp - V2_FRAMES_EACH_SIDE / fps)

    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.4f}", "-i", str(film_path),
        "-t", f"{clip_duration:.4f}",
        "-vf", "scale=640:-2",
        "-q:v", "4",
        str(tmp_dir / "f_%04d.jpg"),
        "-hide_banner", "-loglevel", "error",
    ], capture_output=True)

    candidates = sorted(tmp_dir.glob("f_*.jpg"))
    best_score = -1.0
    best_idx   = len(candidates) // 2

    for i, c in enumerate(candidates):
        score = tenengrad(c)
        if score > best_score:
            best_score = score
            best_idx   = i

    best_t = clip_start + best_idx / fps

    out = frames_dir / f"frame_{idx:02d}.jpg"
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{best_t:.4f}", "-i", str(film_path),
        "-vframes", "1",
        "-vf", f"scale={frame_w}:-2",
        "-q:v", "1",
        str(out),
        "-hide_banner", "-loglevel", "error",
    ], capture_output=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out, best_t, best_score


def color_distance(c1, c2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def get_dominant_colors(img_path: Path) -> list:
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize((400, 225), Image.LANCZOS)
        pixels = np.array(img).reshape(-1, 3).astype(float)

    km = KMeans(n_clusters=V2_N_COLORS * 3, n_init=8, random_state=42)
    km.fit(pixels)

    centers = km.cluster_centers_
    counts  = np.bincount(km.labels_, minlength=len(centers))
    order   = np.argsort(-counts)
    centers, counts = centers[order], counts[order]

    sel_c, sel_n = [], []
    for c, cnt in zip(centers, counts):
        if all(color_distance(c, sc) >= V2_MIN_COLOR_DIST for sc in sel_c):
            sel_c.append(c)
            sel_n.append(cnt)
        if len(sel_c) == V2_N_COLORS:
            break

    if len(sel_c) < V2_N_COLORS:
        for c, cnt in zip(centers, counts):
            if not any(np.array_equal(c, sc) for sc in sel_c):
                sel_c.append(c)
                sel_n.append(cnt)
            if len(sel_c) == V2_N_COLORS:
                break

    total = sum(sel_n)
    return [(int(round(c[0])), int(round(c[1])), int(round(c[2])), cnt / total * 100)
            for c, cnt in zip(sel_c, sel_n)]


def build_card(frame_path: Path, colors: list, out_path: Path,
               frame_w: int, frame_h: int):
    font_hex = try_font(V2_FONT_SIZE_HEX)
    font_pct = try_font(V2_FONT_SIZE_PCT)

    total_w = frame_w + V2_PADDING * 2
    total_h = V2_PADDING + frame_h + V2_PADDING + V2_SWATCH_H + V2_TEXT_AREA + V2_PADDING
    card    = Image.new("RGB", (total_w, total_h), V2_BG_COLOR)
    draw    = ImageDraw.Draw(card)

    with Image.open(frame_path) as frm:
        frm = frm.resize((frame_w, frame_h), Image.LANCZOS)
        card.paste(frm, (V2_PADDING, V2_PADDING))

    swatch_w = frame_w // V2_N_COLORS
    swatch_y = V2_PADDING + frame_h + V2_PADDING
    text_y   = swatch_y + V2_SWATCH_H + 10

    for i, (r, g, b, pct) in enumerate(colors):
        x0 = V2_PADDING + i * swatch_w
        x1 = x0 + swatch_w - 3
        draw.rectangle([x0, swatch_y, x1, swatch_y + V2_SWATCH_H], fill=(r, g, b))
        cx = x0 + swatch_w // 2
        draw.text((cx, text_y), rgb_to_hex(r, g, b),
                  font=font_hex, fill=(50, 50, 50), anchor="mt")
        draw.text((cx, text_y + V2_FONT_SIZE_HEX + 6), f"{pct:.1f}%",
                  font=font_pct, fill=(110, 110, 110), anchor="mt")

    card.save(out_path, "PNG")


def run_v2(film_path: Path, output_dir: Path, meta: dict):
    print("\n─── V2: card frame + colori ─────────────────────────────────────")
    frames_dir = output_dir / "V2" / "frames"
    cards_dir  = output_dir / "V2" / "output"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    duration  = meta["duration"]
    fps       = meta["fps"]
    frame_w   = meta["width"]
    frame_h   = meta["height"]

    step       = duration / (V2_N_FRAMES + 1)
    timestamps = [int(step * (i + 1)) for i in range(V2_N_FRAMES)]

    print(f"  FPS rilevati: {fps:.3f}   Risoluzione: {frame_w}×{frame_h}")
    print(f"  Candidati per frame: {V2_FRAMES_EACH_SIDE*2+1} frame reali (±{V2_FRAMES_EACH_SIDE} @ {fps:.3f}fps)")
    print(f"  Timestamp target: {[f'{t//60}:{t%60:02d}' for t in timestamps]}\n")

    print(f"  Estrazione {V2_N_FRAMES} frame (selezione nitidezza Tenengrad)...")
    frame_paths = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(v2_extract_frame, i + 1, t, film_path, frames_dir, fps, frame_w): (i, t)
            for i, t in enumerate(timestamps)
        }
        for fut in as_completed(futs):
            i, t = futs[fut]
            path, best_t, score = fut.result()
            frame_paths[i] = path
            shift = best_t - t
            bt    = int(best_t)
            print(f"    ✓ Frame {i+1:02d}  "
                  f"target={t//60}:{t%60:02d}  "
                  f"best={bt//60}:{bt%60:02d}  "
                  f"shift={shift:+.2f}s  "
                  f"score={score:.0f}")

    print(f"\n  Generazione {V2_N_FRAMES} card...")
    for i in range(V2_N_FRAMES):
        fp     = frame_paths[i]
        colors = get_dominant_colors(fp)
        out_p  = cards_dir / f"palette_{i+1:02d}.png"
        build_card(fp, colors, out_p, frame_w, frame_h)
        print(f"    ✓ palette_{i+1:02d}.png  →  {[rgb_to_hex(*c[:3]) for c in colors]}")

    print(f"  V2 completato → {cards_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analisi palette colori di un film (V1 + V2)"
    )
    parser.add_argument("--film",   type=str, help="Percorso del file video")
    parser.add_argument("--output", type=str, help="Cartella di output per i risultati")
    return parser.parse_args()


def ask(prompt: str, default: str = "") -> str:
    if default:
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val else default
    while True:
        val = input(f"{prompt}: ").strip()
        if val:
            return val
        print("  ⚠ Inserisci un valore.")


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║         film_palette.py  —  V1 + V2 pipeline        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Percorso film
    if args.film:
        film_path = Path(args.film)
    else:
        raw = ask("Percorso del film (mkv/mp4/...)")
        film_path = Path(raw.strip('"').strip("'"))

    if not film_path.exists():
        print(f"  ✗ File non trovato: {film_path}")
        sys.exit(1)

    # ── Cartella output
    if args.output:
        output_dir = Path(args.output)
    else:
        default_out = film_path.parent / film_path.stem
        raw = ask("Cartella di output", str(default_out))
        output_dir = Path(raw.strip('"').strip("'"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Metadati automatici
    print(f"\n  Film: {film_path.name}")
    print("  Rilevamento metadati con ffprobe...")
    try:
        meta = probe_video(film_path)
    except Exception as e:
        print(f"  ✗ ffprobe fallito: {e}")
        sys.exit(1)

    d = meta["duration"]
    print(f"  ✓ Durata:      {d//3600}h {(d%3600)//60}m {d%60}s  ({d}s)")
    print(f"  ✓ Risoluzione: {meta['width']}×{meta['height']}px")
    print(f"  ✓ FPS:         {meta['fps']:.3f}")
    print(f"  ✓ Output:      {output_dir}\n")

    t0 = time.time()

    run_v1(film_path, output_dir, meta)
    run_v2(film_path, output_dir, meta)

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline completata in {elapsed/60:.1f} min")
    print(f"  Risultati in: {output_dir}")
    print(f"    V1/  →  palette_15s.png  palette_30s.png  palette_45s.png  palette_60s.png")
    print(f"    V2/output/  →  palette_01.png … palette_{V2_N_FRAMES:02d}.png")


if __name__ == "__main__":
    main()
