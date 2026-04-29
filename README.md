# cinepalette

**Extract dominant color palettes and frame cards from any film — automatically.**

`cinepalette` uses FFmpeg + K-Means clustering to analyse the color identity of a movie. Drop in a video file, run one script, get two outputs:

- **V1 — Color timeline:** a horizontal strip image where every 2px column is the dominant color of a frame sampled every 15, 30, 45, or 60 seconds. Great for seeing a film's overall palette at a glance.
- **V2 — Frame cards:** 20 high-quality frames, each paired with its 5 most distinct dominant colors (hex code + percentage). Each frame is selected for sharpness from 61 consecutive real video frames.

---

## Examples

**V1 — Color palette strip (How to Train Your Dragon)**
Each vertical stripe = dominant color of one frame sampled every 15 seconds.

**V2 — Frame card**
Full-resolution frame on top, 5 color swatches below with hex codes and coverage percentage.

---

## Requirements

```
Python 3.10+
ffmpeg + ffprobe  (must be in PATH)
pip install Pillow scikit-learn opencv-python-headless
```

Install Python dependencies:
```bash
pip install Pillow scikit-learn opencv-python-headless
```

Install FFmpeg (if not already installed):
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

---

## Usage

### Interactive (recommended)

```bash
python film_palette.py
```

The script will ask for:
1. Path to the video file (any format ffmpeg supports: `.mkv`, `.mp4`, `.avi`, etc.)
2. Output folder (defaults to a folder named after the film)

### Command-line arguments

```bash
python film_palette.py --film "/path/to/film.mkv" --output "/path/to/output"
```

### Output structure

```
output/
├── V1/
│   ├── palette_15s.png    ← 1 stripe per frame @ every 15s
│   ├── palette_30s.png    ← 1 stripe per frame @ every 30s
│   ├── palette_45s.png    ← 1 stripe per frame @ every 45s
│   ├── palette_60s.png    ← 1 stripe per frame @ every 60s
│   └── frames/            ← extracted source frames (480p)
└── V2/
    ├── output/
    │   ├── palette_01.png  ← frame card 1 (full resolution)
    │   ├── palette_02.png
    │   └── ...
    └── frames/            ← extracted source frames (full resolution)
```

---

## How it works

### V1 — Color timeline

1. Extracts one frame every 15 seconds at 480p using parallel FFmpeg workers (8 threads).
2. For each frame, quantizes the image to 5 colors and picks the most frequent one.
3. Builds 4 palette images by subsampling the frame list at different intervals (every 1st, 2nd, 3rd, 4th frame = 15s / 30s / 45s / 60s).

### V2 — Frame cards

1. Divides the film into 20 equidistant timestamps.
2. For each timestamp, extracts a ~2.5 second clip (61 consecutive real frames at native fps).
3. Scores every candidate frame with **Tenengrad sharpness** (sum of squared Sobel gradients) and a dark-frame penalty, then picks the sharpest one.
4. Re-extracts the winning frame at full native resolution.
5. Runs **K-Means clustering** (15 clusters) on the frame and greedily selects 5 visually distinct colors (minimum Euclidean distance of 40 in RGB space), ordered by coverage.
6. Renders a card: frame on top, color swatches with hex codes and percentages below.

### Automatic metadata detection

The script probes the video with `ffprobe` to detect:
- Duration (no need to specify it manually)
- Native resolution (width × height, cropping black bars with `cropdetect`)
- Frame rate (exact rational fps, e.g. 23.976)

---

## Configuration

All parameters are constants at the top of `film_palette.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `V1_INTERVAL` | `15` | Seconds between frames for V1 |
| `V1_WORKERS` | `8` | Parallel FFmpeg threads |
| `V1_STRIPE_W` | `2` | Width in px of each color stripe |
| `V1_HEIGHT` | `400` | Height in px of palette image |
| `V2_N_FRAMES` | `20` | Number of frame cards to generate |
| `V2_N_COLORS` | `5` | Colors per card |
| `V2_MIN_COLOR_DIST` | `40` | Minimum RGB distance between selected colors |
| `V2_FRAMES_EACH_SIDE` | `30` | Candidate frames before/after each timestamp |

---

## Notes

- **HDR / Dolby Vision files:** colors will appear desaturated grey. Use an SDR version of the film (the script does not perform tone-mapping).
- **Animated films** tend to score lower on Tenengrad since cel-shaded edges are softer than live-action; results are still good.
- V1 frame extraction runs in ~90s for a typical 2h film on 8 threads. V2 adds ~3–5 minutes.

---

## License

MIT
