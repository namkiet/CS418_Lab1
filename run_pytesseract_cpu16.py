import os

os.environ["OMP_THREAD_LIMIT"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OMP_DYNAMIC"] = "FALSE"
TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
os.environ["PATH"] = TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")

import argparse
import json
import re
import shutil
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import pytesseract
from jiwer import cer, wer
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_DIR, "tesseract.exe")
DATA_ROOT = ROOT / "test"
OUTPUT_DIR = ROOT / "result" / "tesseract_pytesseract_cpu16"
RESULTS_CSV = OUTPUT_DIR / "per_image_results.csv"
SAVE_EVERY = 50
MODEL_ID = "Tesseract 5.5.3.20260724 (pytesseract, eng, OEM 3, PSM 3)"


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


def collect_pairs() -> list[dict]:
    pairs = []
    for category, expected in (("text", 6000), ("nontext", 3000)):
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


def main() -> None:
    global OUTPUT_DIR, RESULTS_CSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    RESULTS_CSV = OUTPUT_DIR / "per_image_results.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = shutil.which("tesseract")
    if not engine:
        raise RuntimeError("Tesseract binary not available on PATH")
    version = subprocess.run([engine, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    print(f"Engine: {version}; PATH: {engine}; OMP_THREAD_LIMIT={os.environ['OMP_THREAD_LIMIT']}", flush=True)
    pairs = collect_pairs()
    if len(pairs) != 9000:
        raise RuntimeError(f"Expected 9,000 test pairs, found {len(pairs)}")

    if args.limit:
        pairs = pairs[:args.limit]
    total = len(pairs)
    status_path = ROOT / "result" / "run_status.json"
    status_path.parent.mkdir(exist_ok=True)
    def status(phase, success, **extra):
        status_path.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), total=total, success=success, updated=time.time(), output_dir=str(OUTPUT_DIR), **extra), indent=2))
    status("running", 0)
    previous = pd.read_csv(RESULTS_CSV).to_dict("records") if RESULTS_CSV.exists() else []
    ids = {p["sample_id"] for p in pairs}
    rows = [r for r in previous if r.get("sample_id") in ids and r.get("status") == "ok"]
    finished = {r["sample_id"] for r in rows if r.get("status") == "ok"}
    print(f"Resume: {len(finished):,} successful; remaining {len(pairs)-len(finished):,}", flush=True)
    t0, done = time.perf_counter(), 0

    for sample in tqdm(pairs, desc="Tesseract OPRB CPU (16-thread limit)"):
        if sample["sample_id"] in finished:
            continue
        row = {**sample, "model_id": MODEL_ID}
        try:
            start = time.perf_counter()
            with Image.open(sample["image_path"]) as image:
                prediction = pytesseract.image_to_string(image, lang="eng", config="--oem 3 --psm 3")
            latency = time.perf_counter() - start
            reference = read_gt(Path(sample["gt_path"]))
            row.update({
                "status": "ok",
                "prediction": prediction.strip(),
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
        status("running", len(finished), last_sample=sample["sample_id"], last_status=row["status"])
        if done % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed else 0
            print(
                f"\ncheckpoint: success={len(finished):,}/9,000; new={done:,}; "
                f"rate={rate:.2f} img/s; ETA={(9000-len(finished))/rate/3600:.2f} h",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    results = pd.DataFrame(rows)
    ok = results.loc[results["status"] == "ok"]
    summary = {
        "model": MODEL_ID,
        "dataset": "kpurkayastha/OPRB test",
        "total_expected": total,
        "n_success": int(len(ok)),
        "n_errors": int(len(results) - len(ok)),
        "mean_latency_sec": float(ok["latency_sec"].mean()) if len(ok) else None,
        "mean_cer": float(ok["cer"].mean()) if len(ok) else None,
        "mean_wer": float(ok["wer"].mean()) if len(ok) else None,
        "mean_token_f1": float(ok["token_f1"].mean()) if len(ok) else None,
        "thread_limit": 16,
        "runtime_sec_this_run": time.perf_counter() - t0,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {RESULTS_CSV}", flush=True)
    status("completed" if summary["n_errors"] == 0 else "errors", len(ok), errors=summary["n_errors"])
    if summary["n_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
