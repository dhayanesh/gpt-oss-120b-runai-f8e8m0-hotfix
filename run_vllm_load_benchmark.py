import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("/workspace/benchmark_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        "id": "base",
        "label": "Base (no RunAI streamer)",
        "script": "/workspace/benchmark_load_base.py",
    },
    {
        "id": "runai_sharded",
        "label": "RunAI sharded (runai_streamer_sharded)",
        "script": "/workspace/benchmark_load_runai_sharded.py",
    },
]

RE_READY = re.compile(r"BENCHMARK_READY_SECONDS=([0-9.]+)")
RE_WEIGHTS = re.compile(r"Loading weights took ([0-9.]+) seconds")
RE_MODEL_LOADING = re.compile(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds")
RE_INIT_ENGINE = re.compile(r"init engine .* took ([0-9.]+) seconds")
RE_TORCH_COMPILE = re.compile(r"torch\.compile takes ([0-9.]+) s in total")
RE_GRAPH = re.compile(r"Graph capturing finished in ([0-9.]+) secs, took ([0-9.]+) GiB")
RE_RUNAI_STREAM = re.compile(
    r"Overall time to stream ([0-9.]+) GiB of all files to cpu: ([0-9.]+)s, ([0-9.]+) GiB/s"
)

LOG_KEYS = [
    "Loading weights took",
    "Model loading took",
    "init engine",
    "torch.compile takes",
    "Graph capturing finished",
    "RunAI Streamer",
    "BENCHMARK_READY_SECONDS",
]


def pick_last_float(regex: re.Pattern[str], text: str):
    m = regex.findall(text)
    if not m:
        return None
    x = m[-1]
    if isinstance(x, tuple):
        return tuple(float(v) for v in x)
    return float(x)


def parse_metrics(log_text: str):
    metrics = {}

    ready = pick_last_float(RE_READY, log_text)
    if ready is not None:
        metrics["ready_seconds"] = ready

    weights = RE_WEIGHTS.findall(log_text)
    if weights:
        metrics["weights_loading_seconds"] = float(weights[-1])

    model_loading = pick_last_float(RE_MODEL_LOADING, log_text)
    if model_loading is not None:
        model_mem, model_seconds = model_loading
        metrics["model_loading_memory_gib"] = model_mem
        metrics["model_loading_seconds"] = model_seconds

    init_engine = pick_last_float(RE_INIT_ENGINE, log_text)
    if init_engine is not None:
        metrics["init_engine_seconds"] = init_engine

    compile_s = pick_last_float(RE_TORCH_COMPILE, log_text)
    if compile_s is not None:
        metrics["torch_compile_seconds"] = compile_s

    graph = pick_last_float(RE_GRAPH, log_text)
    if graph is not None:
        graph_s, graph_mem = graph
        metrics["graph_capture_seconds"] = graph_s
        metrics["graph_capture_memory_gib"] = graph_mem

    runai_stream = RE_RUNAI_STREAM.findall(log_text)
    if runai_stream:
        triples = [(float(a), float(b), float(c)) for a, b, c in runai_stream]
        total_gib = triples[0][0]
        best_time = min(t[1] for t in triples)
        best_bw = max(t[2] for t in triples)
        metrics["runai_stream_total_gib"] = total_gib
        metrics["runai_stream_time_seconds_best"] = best_time
        metrics["runai_stream_bw_gib_s_best"] = best_bw

    return metrics


def run_case(case):
    cmd = [sys.executable, case["script"]]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/workspace",
    )
    wall = time.perf_counter() - t0
    log_text = proc.stdout

    log_path = OUT_DIR / f"{case['id']}.log"
    log_path.write_text(log_text)

    metrics = parse_metrics(log_text)
    metrics["case_id"] = case["id"]
    metrics["case_label"] = case["label"]
    metrics["exit_code"] = proc.returncode
    metrics["wall_seconds"] = wall
    metrics["command"] = " ".join(cmd)
    metrics["log_file"] = str(log_path)

    excerpts = [line.strip() for line in log_text.splitlines() if any(k in line for k in LOG_KEYS)]
    excerpt_path = OUT_DIR / f"{case['id']}_excerpts.log"
    excerpt_path.write_text("\n".join(excerpts) + "\n")
    metrics["excerpt_file"] = str(excerpt_path)

    return metrics


def ffmt(v, n=2):
    if v is None:
        return "N/A"
    return f"{v:.{n}f}"


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_rounded_bar(draw, x0, y0, x1, y1, color, radius=8):
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=color)


def make_chart(base, runai):
    W, H = 1800, 1000
    img = Image.new("RGB", (W, H), "#0f172a")
    draw = ImageDraw.Draw(img)

    for y in range(H):
        c = int(15 + 25 * (y / H))
        draw.line([(0, y), (W, y)], fill=(c, c + 10, c + 30))

    title_f = load_font(44, bold=True)
    sub_f = load_font(24)
    axis_f = load_font(20)
    small_f = load_font(18)
    num_f = load_font(22, bold=True)

    draw.text((50, 28), "vLLM Startup Benchmark: Base vs RunAI Sharded", fill="#e2e8f0", font=title_f)
    draw.text((50, 88), "Model: gpt-oss-120b, TP=2, max_model_len=4096, gpu_memory_utilization=0.92", fill="#cbd5e1", font=sub_f)

    panel1 = (40, 150, 1120, 930)
    panel2 = (1140, 150, 1760, 930)
    draw.rounded_rectangle(panel1, radius=22, fill="#111827", outline="#334155", width=2)
    draw.rounded_rectangle(panel2, radius=22, fill="#111827", outline="#334155", width=2)

    draw.text((70, 175), "Initialization Time Breakdown", fill="#f8fafc", font=load_font(30, bold=True))

    metrics = [
        ("Ready (wall)", "ready_seconds"),
        ("Model loading", "model_loading_seconds"),
        ("Weights loading", "weights_loading_seconds"),
        ("Engine init", "init_engine_seconds"),
        ("torch.compile", "torch_compile_seconds"),
        ("Graph capture", "graph_capture_seconds"),
    ]

    vals_base = [base.get(k) for _, k in metrics]
    vals_runai = [runai.get(k) for _, k in metrics]
    max_v = max([v for v in vals_base + vals_runai if isinstance(v, (int, float))] or [1.0])

    chart_x0, chart_y0 = 85, 245
    chart_x1, chart_y1 = 1080, 885
    draw.line([(chart_x0, chart_y1), (chart_x1, chart_y1)], fill="#64748b", width=2)
    draw.line([(chart_x0, chart_y0), (chart_x0, chart_y1)], fill="#64748b", width=2)

    for i in range(6):
        y = chart_y1 - (chart_y1 - chart_y0) * (i / 5)
        label = f"{max_v * (i/5):.0f}s"
        draw.line([(chart_x0, y), (chart_x1, y)], fill="#1f2937", width=1)
        draw.text((chart_x0 - 62, y - 12), label, fill="#94a3b8", font=small_f)

    group_w = (chart_x1 - chart_x0) / len(metrics)
    bar_w = group_w * 0.32
    base_color = "#38bdf8"
    runai_color = "#f59e0b"

    for i, (name, key) in enumerate(metrics):
        gx = chart_x0 + i * group_w + group_w * 0.18

        b = base.get(key)
        r = runai.get(key)

        if isinstance(b, (int, float)):
            bh = (b / max_v) * (chart_y1 - chart_y0)
            draw_rounded_bar(draw, gx, chart_y1 - bh, gx + bar_w, chart_y1, base_color, radius=6)
            draw.text((gx - 2, chart_y1 - bh - 30), f"{b:.1f}", fill="#e2e8f0", font=small_f)

        if isinstance(r, (int, float)):
            rx = gx + bar_w + group_w * 0.1
            rh = (r / max_v) * (chart_y1 - chart_y0)
            draw_rounded_bar(draw, rx, chart_y1 - rh, rx + bar_w, chart_y1, runai_color, radius=6)
            draw.text((rx - 2, chart_y1 - rh - 30), f"{r:.1f}", fill="#e2e8f0", font=small_f)

        draw.text((gx - 12, chart_y1 + 18), name, fill="#cbd5e1", font=axis_f)

    draw_rounded_bar(draw, 760, 185, 785, 210, base_color, radius=4)
    draw.text((795, 182), "Base", fill="#e2e8f0", font=axis_f)
    draw_rounded_bar(draw, 890, 185, 915, 210, runai_color, radius=4)
    draw.text((925, 182), "RunAI sharded", fill="#e2e8f0", font=axis_f)

    draw.text((1168, 175), "Speedup & Supporting Metrics", fill="#f8fafc", font=load_font(30, bold=True))

    speed_metrics = [
        ("Ready (wall)", "ready_seconds"),
        ("Model loading", "model_loading_seconds"),
        ("Weights loading", "weights_loading_seconds"),
    ]

    y = 255
    for label, key in speed_metrics:
        b = base.get(key)
        r = runai.get(key)
        if isinstance(b, (int, float)) and isinstance(r, (int, float)) and r > 0:
            sp = b / r
            draw.text((1180, y), label, fill="#cbd5e1", font=axis_f)
            draw.text((1560, y), f"{sp:.2f}x", fill="#22d3ee", font=num_f)
            bw = int(min(320, 320 * (sp / max(1.0, sp))))
            draw_rounded_bar(draw, 1320, y + 8, 1320 + bw, y + 32, "#22d3ee", radius=8)
            y += 90

    y += 10
    draw.text((1180, y), "Additional initialization metrics", fill="#f8fafc", font=load_font(24, bold=True))
    y += 44

    items = [
        ("Base model loading memory", f"{ffmt(base.get('model_loading_memory_gib'))} GiB"),
        ("RunAI model loading memory", f"{ffmt(runai.get('model_loading_memory_gib'))} GiB"),
        ("Base graph capture memory", f"{ffmt(base.get('graph_capture_memory_gib'))} GiB"),
        ("RunAI graph capture memory", f"{ffmt(runai.get('graph_capture_memory_gib'))} GiB"),
        (
            "RunAI streamer throughput",
            f"{ffmt(runai.get('runai_stream_bw_gib_s_best'))} GiB/s (best), "
            f"{ffmt(runai.get('runai_stream_total_gib'))} GiB total",
        ),
    ]
    for k, v in items:
        draw.text((1180, y), f"- {k}: {v}", fill="#cbd5e1", font=small_f)
        y += 38

    draw.text((50, 950), "Source: vLLM initialization logs (core.py, default_loader.py, gpu_model_runner.py, runai streamer).", fill="#94a3b8", font=small_f)

    out_path = OUT_DIR / "vllm_load_benchmark.png"
    img.save(out_path)
    return out_path


def write_markdown(base, runai, chart_path):
    def metric_row(title, key):
        return f"| {title} | {ffmt(base.get(key))} | {ffmt(runai.get(key))} |"

    lines = []
    lines.append("# vLLM Runtime Benchmark: Base vs RunAI Sharded")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Compare vLLM initialization/runtime loading behavior for:")
    lines.append("- Base model: `/workspace/models/gpt-oss-120b` (default load path)")
    lines.append("- RunAI sharded model: `/workspace/models/gpt-oss-120b-runai-sharded-tp2` with `load_format=runai_streamer_sharded`")
    lines.append("- Both runs use `tensor_parallel_size=2`, `dtype=auto`, `gpu_memory_utilization=0.92`, `max_model_len=4096`.")
    lines.append("")
    lines.append("## Visualization")
    lines.append(f"![vLLM load benchmark](./{chart_path.name})")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("| Metric | Base | RunAI sharded |")
    lines.append("|---|---:|---:|")
    lines.append(metric_row("Wall time to model ready (s)", "ready_seconds"))
    lines.append(metric_row("Weights loading time (s)", "weights_loading_seconds"))
    lines.append(metric_row("Model loading time (s)", "model_loading_seconds"))
    lines.append(metric_row("Engine init time (s)", "init_engine_seconds"))
    lines.append(metric_row("torch.compile total (s)", "torch_compile_seconds"))
    lines.append(metric_row("Graph capture time (s)", "graph_capture_seconds"))
    lines.append(metric_row("Model loading memory (GiB)", "model_loading_memory_gib"))
    lines.append(metric_row("Graph capture memory (GiB)", "graph_capture_memory_gib"))
    lines.append("")

    def speedup(k):
        b = base.get(k)
        r = runai.get(k)
        if isinstance(b, (int, float)) and isinstance(r, (int, float)) and r > 0:
            return b / r
        return None

    lines.append("## Speedups (Base / RunAI)")
    lines.append("| Metric | Speedup |")
    lines.append("|---|---:|")
    for label, key in [
        ("Wall time to ready", "ready_seconds"),
        ("Weights loading", "weights_loading_seconds"),
        ("Model loading", "model_loading_seconds"),
    ]:
        s = speedup(key)
        lines.append(f"| {label} | {ffmt(s)}x |" if s is not None else f"| {label} | N/A |")

    lines.append("")
    lines.append("## RunAI Streamer Metrics")
    lines.append(f"- Streamed size: {ffmt(runai.get('runai_stream_total_gib'))} GiB")
    lines.append(f"- Best observed stream time: {ffmt(runai.get('runai_stream_time_seconds_best'))} s")
    lines.append(f"- Best observed throughput: {ffmt(runai.get('runai_stream_bw_gib_s_best'))} GiB/s")
    lines.append("")

    lines.append("## Relevant vLLM Initialization Log Excerpts")
    lines.append("")
    lines.append("### Base")
    lines.append("```text")
    lines.extend(Path(base["excerpt_file"]).read_text().splitlines()[:40])
    lines.append("```")
    lines.append("")
    lines.append("### RunAI sharded")
    lines.append("```text")
    lines.extend(Path(runai["excerpt_file"]).read_text().splitlines()[:60])
    lines.append("```")
    lines.append("")

    lines.append("## Raw Artifacts")
    lines.append(f"- Base log: `{base['log_file']}`")
    lines.append(f"- RunAI log: `{runai['log_file']}`")
    lines.append(f"- Base excerpts: `{base['excerpt_file']}`")
    lines.append(f"- RunAI excerpts: `{runai['excerpt_file']}`")
    lines.append(f"- Metrics JSON: `{OUT_DIR / 'benchmark_metrics.json'}`")

    report_path = OUT_DIR / "VLLM_LOAD_BENCHMARK.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    results = {}
    for case in CASES:
        print(f"[benchmark] Running {case['label']} ...", flush=True)
        m = run_case(case)
        if m["exit_code"] != 0:
            raise RuntimeError(f"Case failed: {case['label']} (exit={m['exit_code']})")
        results[case["id"]] = m
        print(
            f"[benchmark] Completed {case['label']}: wall={m.get('wall_seconds', float('nan')):.2f}s, "
            f"ready={m.get('ready_seconds', float('nan')):.2f}s",
            flush=True,
        )

    metrics_path = OUT_DIR / "benchmark_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    chart_path = make_chart(results["base"], results["runai_sharded"])
    report_path = write_markdown(results["base"], results["runai_sharded"], chart_path)

    print(f"[benchmark] Metrics: {metrics_path}")
    print(f"[benchmark] Chart: {chart_path}")
    print(f"[benchmark] Report: {report_path}")


if __name__ == "__main__":
    main()
