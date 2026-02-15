import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

METRICS = Path("/workspace/benchmark_results/benchmark_metrics.json")
OUT = Path("/workspace/benchmark_results/vllm_load_benchmark.png")
LEGACY_OUT = Path("/workspace/benchmark_results/vllm_load_benchmark_simple.png")


def load_font(size=16, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_card(draw, box, fill="#ffffff", border="#d1d5db"):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=border, width=2)


def ffmt(value, digits=2):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def draw_grouped_bars(draw, box, base, runai):
    x0, y0, x1, y1 = box
    draw_card(draw, box)

    title_font = load_font(26, bold=True)
    label_font = load_font(16)
    value_font = load_font(14, bold=True)
    axis_font = load_font(13)

    draw.text((x0 + 20, y0 + 14), "Startup Timing Breakdown (seconds)", fill="#111827", font=title_font)

    metrics = [
        ("Ready", "ready_seconds"),
        ("Weights", "weights_loading_seconds"),
        ("Model", "model_loading_seconds"),
        ("Engine", "init_engine_seconds"),
        ("Compile", "torch_compile_seconds"),
        ("Graph", "graph_capture_seconds"),
    ]

    pad_left = 70
    pad_right = 30
    pad_top = 70
    pad_bottom = 72

    cx0 = x0 + pad_left
    cy0 = y0 + pad_top
    cx1 = x1 - pad_right
    cy1 = y1 - pad_bottom
    chart_h = cy1 - cy0
    chart_w = cx1 - cx0

    vals = []
    for _, key in metrics:
        b = base.get(key)
        r = runai.get(key)
        if isinstance(b, (int, float)):
            vals.append(b)
        if isinstance(r, (int, float)):
            vals.append(r)
    max_v = max(vals) if vals else 1.0
    max_v = max(max_v, 1.0)

    for t in range(6):
        value = max_v * t / 5
        y = cy1 - chart_h * t / 5
        draw.line([(cx0, y), (cx1, y)], fill="#e5e7eb", width=1)
        text = f"{value:.0f}s"
        tw = draw.textlength(text, font=axis_font)
        draw.text((cx0 - tw - 8, y - 8), text, fill="#6b7280", font=axis_font)

    draw.line([(cx0, cy0), (cx0, cy1)], fill="#374151", width=2)
    draw.line([(cx0, cy1), (cx1, cy1)], fill="#374151", width=2)

    n = len(metrics)
    group_w = chart_w / n
    bar_w = max(14, int(group_w * 0.24))
    gap = max(6, int(group_w * 0.10))
    base_color = "#3b82f6"
    runai_color = "#f59e0b"

    for i, (label, key) in enumerate(metrics):
        gx = cx0 + i * group_w + (group_w - (2 * bar_w + gap)) / 2
        b = base.get(key)
        r = runai.get(key)
        if isinstance(b, (int, float)):
            bh = int((b / max_v) * chart_h)
            bx0 = gx
            by0 = cy1 - bh
            bx1 = gx + bar_w
            draw.rounded_rectangle([bx0, by0, bx1, cy1], radius=6, fill=base_color, outline=base_color)
            val = f"{b:.1f}"
            tw = draw.textlength(val, font=value_font)
            draw.text((bx0 + (bar_w - tw) / 2, by0 - 20), val, fill="#1f2937", font=value_font)
        if isinstance(r, (int, float)):
            rh = int((r / max_v) * chart_h)
            rx0 = gx + bar_w + gap
            ry0 = cy1 - rh
            rx1 = rx0 + bar_w
            draw.rounded_rectangle([rx0, ry0, rx1, cy1], radius=6, fill=runai_color, outline=runai_color)
            val = f"{r:.1f}"
            tw = draw.textlength(val, font=value_font)
            draw.text((rx0 + (bar_w - tw) / 2, ry0 - 20), val, fill="#1f2937", font=value_font)

        lw = draw.textlength(label, font=label_font)
        draw.text((gx + (2 * bar_w + gap - lw) / 2, cy1 + 12), label, fill="#111827", font=label_font)

    lx = x1 - 240
    ly = y0 + 18
    draw.rounded_rectangle([lx, ly, lx + 18, ly + 18], radius=4, fill=base_color, outline=base_color)
    draw.text((lx + 26, ly - 1), "Base", fill="#111827", font=label_font)
    draw.rounded_rectangle([lx, ly + 28, lx + 18, ly + 46], radius=4, fill=runai_color, outline=runai_color)
    draw.text((lx + 26, ly + 27), "RunAI sharded", fill="#111827", font=label_font)


def draw_speedup_panel(draw, box, base, runai):
    x0, y0, x1, y1 = box
    draw_card(draw, box)

    title_font = load_font(24, bold=True)
    text_font = load_font(16)
    big_font = load_font(26, bold=True)
    draw.text((x0 + 20, y0 + 14), "Key Speedups (Base / RunAI)", fill="#111827", font=title_font)

    items = [
        ("Ready", "ready_seconds"),
        ("Weights", "weights_loading_seconds"),
        ("Model", "model_loading_seconds"),
    ]
    base_y = y0 + 70
    row_h = 66
    max_sp = 1.0
    speedups = []
    for _, key in items:
        b = base.get(key)
        r = runai.get(key)
        sp = b / r if isinstance(b, (int, float)) and isinstance(r, (int, float)) and r > 0 else None
        speedups.append(sp)
        if sp is not None:
            max_sp = max(max_sp, sp)

    for idx, (label, _) in enumerate(items):
        y = base_y + idx * row_h
        sp = speedups[idx]
        draw.text((x0 + 24, y + 18), label, fill="#111827", font=text_font)
        if sp is None:
            draw.text((x0 + 120, y + 14), "N/A", fill="#6b7280", font=text_font)
            continue
        ratio = sp / max_sp
        bar_x0 = x0 + 120
        bar_x1 = x1 - 110
        bw = int((bar_x1 - bar_x0) * ratio)
        draw.rounded_rectangle([bar_x0, y + 18, bar_x0 + bw, y + 42], radius=8, fill="#22c55e")
        draw.text((x1 - 90, y + 10), f"{sp:.2f}x", fill="#065f46", font=big_font)


def draw_info_panel(draw, box, runai):
    x0, y0, x1, y1 = box
    draw_card(draw, box)

    title_font = load_font(24, bold=True)
    text_font = load_font(18)
    draw.text((x0 + 20, y0 + 14), "RunAI Streamer Metrics", fill="#111827", font=title_font)

    lines = [
        f"Streamed size: {ffmt(runai.get('runai_stream_total_gib'))} GiB",
        f"Best stream time: {ffmt(runai.get('runai_stream_time_seconds_best'))} s",
        f"Best throughput: {ffmt(runai.get('runai_stream_bw_gib_s_best'))} GiB/s",
    ]
    for i, line in enumerate(lines):
        draw.text((x0 + 24, y0 + 62 + i * 34), line, fill="#1f2937", font=text_font)


def main():
    data = json.loads(METRICS.read_text())
    base = data["base"]
    runai = data["runai_sharded"]

    w, h = 1700, 760
    img = Image.new("RGB", (w, h), "#f3f4f6")
    draw = ImageDraw.Draw(img)

    title_font = load_font(42, bold=True)
    sub_font = load_font(20)
    draw.text((42, 24), "vLLM Load Benchmark", fill="#0f172a", font=title_font)
    draw.text((44, 80), "Base model vs RunAI sharded streaming load", fill="#334155", font=sub_font)

    # Only keep the top timing chart panel.
    draw_grouped_bars(draw, (34, 130, 1666, 700), base, runai)

    img.save(OUT)
    if LEGACY_OUT.exists():
        LEGACY_OUT.unlink()
    print(f"saved {OUT}")
    print(f"removed legacy image: {LEGACY_OUT}")


if __name__ == "__main__":
    main()
