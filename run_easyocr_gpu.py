import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import easyocr
import pandas as pd
import torch
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "test"
DEFAULT_OUTPUT_DIR = ROOT / "result" / "easyocr_gpu"


def read_gt(path: Path) -> str:
    tokens = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 2 and fields[0].strip():
                tokens.append(fields[0].strip())
    return " ".join(tokens)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def metrics(reference: str, prediction: str) -> dict:
    ref, hyp = clean(reference), clean(prediction)
    if not ref:
        raise ValueError("Empty GT transcript")
    a, b = Counter(ref.split()), Counter(hyp.split())
    matched = sum((a & b).values())
    precision = matched / sum(b.values()) if b else 0.0
    recall = matched / sum(a.values()) if a else 0.0
    return {
        "cer": float(cer(ref, hyp)),
        "wer": float(wer(ref, hyp)),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "gt_words": len(ref.split()),
        "pred_words": len(hyp.split()),
    }


def collect_pairs(category_filter: str = "all") -> list[dict]:
    categories = (("text", 6000), ("nontext", 3000))
    if category_filter != "all":
        categories = tuple(item for item in categories if item[0] == category_filter)
    pairs = []
    for category, expected in categories:
        image_dir, gt_dir = DATA_ROOT / category / "img", DATA_ROOT / category / "gt"
        images = {p.stem: p for p in image_dir.glob("*.jpg")}
        gts = {p.stem: p for p in gt_dir.glob("*.txt")}
        common = sorted(images.keys() & gts.keys())
        if len(common) != expected:
            raise RuntimeError(f"Expected {expected} paired {category} samples, found {len(common)}")
        print(f"{category}: {len(common):,} paired images", flush=True)
        for stem in common:
            parts = stem.split("__", 2)
            pairs.append({
                "category": category,
                "sample_id": stem,
                "image_path": str(images[stem]),
                "gt_path": str(gts[stem]),
                "occlusion_type": parts[0],
                "occlusion_density": parts[1] if len(parts) > 1 else "unknown",
            })
    return pairs


def save_csv_atomic(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def load_resume_state(results_csv: Path, pairs: list[dict]) -> tuple[list[dict], set[str], int]:
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return [], set(), 0
    previous = pd.read_csv(results_csv).to_dict("records")

    # Keep all valid successful rows across categories (deduplicated, preserving order)
    rows: list[dict] = []
    finished: set[str] = set()
    for r in previous:
        sid = r.get("sample_id")
        if isinstance(sid, str) and sid and r.get("status") == "ok" and sid not in finished:
            rows.append(r)
            finished.add(sid)

    start_idx = next((idx for idx, p in enumerate(pairs) if p["sample_id"] not in finished), len(pairs))
    return rows, finished, start_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EasyOCR evaluation on OPRB test set")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for EasyOCR recognition (default: 64)")
    parser.add_argument("--resume", action="store_true", help="Resume from the last completed sample in per_image_results.csv")
    parser.add_argument("--category", choices=("all", "text", "nontext"), default="all", help="Dataset category to run (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on total samples for testing")
    parser.add_argument("--save-every", type=int, default=25, help="Checkpoint interval in samples (default: 25)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "per_image_results.csv"
    status_path = output_dir / "run_status.json"

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    model_id = f"EasyOCR {easyocr.__version__} (en, {'GPU' if use_gpu else 'CPU'}, batch_size={args.batch_size})"
    print(f"Engine: {model_id} | torch={torch.__version__} | device={device_name}", flush=True)

    pairs = collect_pairs(args.category)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    total = len(pairs)

    if args.resume:
        rows, finished, start_idx = load_resume_state(results_csv, pairs)
        last_sid = rows[-1]["sample_id"] if rows else None
        print(
            f"Resume: {len(finished):,}/{total:,} completed; last_sample={last_sid}; "
            f"starting at index {start_idx} (remaining {max(0, total - start_idx):,})",
            flush=True,
        )
    else:
        rows, finished, start_idx = [], set(), 0
        print(f"Fresh run: {total:,} samples -> {output_dir}", flush=True)

    def write_status(phase: str, success: int, **extra) -> None:
        payload = {
            "phase": phase,
            "total": total,
            "success": success,
            "updated": time.time(),
            "output_dir": str(output_dir),
            "batch_size": args.batch_size,
            "device": device_name,
            **extra,
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    write_status("running", len(finished))

    t0, done = time.perf_counter(), 0
    remaining = pairs[start_idx:]

    for sample in remaining:
        if sample["sample_id"] in finished:
            continue
        row = {**sample, "model_id": model_id}
        try:
            start = time.perf_counter()
            lines = reader.readtext(sample["image_path"], detail=0, paragraph=False, batch_size=args.batch_size)
            latency = time.perf_counter() - start
            prediction = " ".join(str(t).strip() for t in lines if str(t).strip()).strip()
            reference = read_gt(Path(sample["gt_path"]))
            row.update({
                "status": "ok",
                "prediction": prediction,
                "ground_truth": reference,
                "latency_sec": latency,
                "generated_tokens": 0,
                "possibly_truncated": False,
                **metrics(reference, prediction),
            })
            finished.add(sample["sample_id"])
        except Exception as exc:
            row.update({"status": "error", "error": repr(exc)})

        rows.append(row)
        done += 1
        write_status("running", len(finished), last_sample=sample["sample_id"], last_status=row["status"])

        checkpointed = ""
        if done % args.save_every == 0:
            save_csv_atomic(rows, results_csv)
            checkpointed = " [checkpoint]"

        if use_gpu and done % 25 == 0:
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed else 0.0
        eta_min = (len(remaining) - done) / rate / 60 if rate else 0.0
        print(
            f"[{len(rows):,}/{total:,}] ok={len(finished):,}/{total:,} | new={done:,} | "
            f"rate={rate:.2f} img/s | ETA={eta_min:.1f}m | {sample['sample_id']} ({row['status']}){checkpointed}",
            end="\n\n",
            flush=True,
        )

    save_csv_atomic(rows, results_csv)
    results = pd.DataFrame(rows)
    ok = results.loc[results["status"] == "ok"] if len(results) else results
    summary = {
        "model": model_id,
        "dataset": "kpurkayastha/OPRB test",
        "total_expected": total,
        "n_success": int(len(ok)),
        "n_errors": int(len(results) - len(ok)),
        "mean_latency_sec": float(ok["latency_sec"].mean()) if len(ok) else None,
        "mean_cer": float(ok["cer"].mean()) if len(ok) else None,
        "mean_wer": float(ok["wer"].mean()) if len(ok) else None,
        "mean_token_f1": float(ok["token_f1"].mean()) if len(ok) else None,
        "batch_size": args.batch_size,
        "device": device_name,
        "runtime_sec_this_run": time.perf_counter() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {results_csv}", flush=True)
    write_status("completed" if summary["n_errors"] == 0 else "errors", len(ok), errors=summary["n_errors"])
    if summary["n_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
