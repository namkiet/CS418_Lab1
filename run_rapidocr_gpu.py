import argparse
import gc
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path


def _register_nvidia_dlls() -> None:
    nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_dir.exists():
        return
    bin_dirs = [str(p) for p in sorted(nvidia_dir.glob("*/bin")) if p.is_dir()]
    if not bin_dirs:
        return
    for d in bin_dirs:
        os.add_dll_directory(d)
    os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


_register_nvidia_dlls()

import onnxruntime as ort
import pandas as pd
from jiwer import cer, wer
from rapidocr import RapidOCR
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "test"
DEFAULT_OUTPUT_DIR = ROOT / "result" / "rapidocr_gpu"


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


def save_csv_atomic(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def load_resume_state(results_csv: Path, pairs: list[dict]) -> tuple[list[dict], set[str], int]:
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return [], set(), 0
    previous = pd.read_csv(results_csv).to_dict("records")
    id_to_idx = {p["sample_id"]: idx for idx, p in enumerate(pairs)}

    rows: list[dict] = []
    finished: set[str] = set()
    last_idx = -1
    for r in previous:
        sid = r.get("sample_id")
        if sid in id_to_idx and r.get("status") == "ok" and sid not in finished:
            r.pop("error", None)
            rows.append(r)
            finished.add(sid)
            last_idx = max(last_idx, id_to_idx[sid])

    start_idx = last_idx + 1 if last_idx >= 0 else 0
    return rows, finished, start_idx


def query_gpu_stats() -> dict:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [p.strip() for p in res.stdout.splitlines()[0].split(",")]
        return {
            "gpu_name": parts[0],
            "gpu_memory_used_mb": int(parts[1]),
            "gpu_memory_total_mb": int(parts[2]),
            "gpu_util_pct": int(parts[3]),
        }
    except Exception:
        return {}


def verify_cuda_providers(ocr: RapidOCR) -> dict[str, list[str]]:
    providers = {}
    for name, module in (("det", ocr.text_det), ("cls", ocr.text_cls), ("rec", ocr.text_rec)):
        if module is not None:
            eps = module.session.session.get_providers()
            providers[name] = eps
            if not eps or eps[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"RapidOCR {name} is not using CUDAExecutionProvider: {eps}")
    return providers


def create_ocr(rec_batch_num: int) -> RapidOCR:
    return RapidOCR(
        params={
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.use_cuda": True,
            "EngineConfig.onnxruntime.use_dml": False,
            "EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy": "kSameAsRequested",
            "EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit": 4 * 1024**3,
            "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
            "Rec.rec_batch_num": rec_batch_num,
        }
    )


def create_cpu_ocr(threads: int = 8) -> RapidOCR:
    return RapidOCR(
        params={
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_dml": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RapidOCR GPU evaluation on OPRB test set")
    parser.add_argument("--rec-batch-num", type=int, default=8, help="Recognition batch size for RapidOCR (default: 8)")
    parser.add_argument("--resume", action="store_true", help="Resume and retry any non-ok samples in per_image_results.csv")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on total samples for testing")
    parser.add_argument("--save-every", type=int, default=25, help="Checkpoint interval in samples (default: 25)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    if args.rec_batch_num < 1:
        raise ValueError("--rec-batch-num must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {ort.get_available_providers()}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "per_image_results.csv"
    status_path = output_dir / "run_status.json"

    rapidocr_ver = importlib.metadata.version("rapidocr")
    model_id = f"RapidOCR {rapidocr_ver} (PP-OCRv6 onnxruntime CUDA, rec_batch_num={args.rec_batch_num})"
    print(f"Engine: {model_id} | onnxruntime-gpu={ort.__version__}", flush=True)

    pairs = collect_pairs()
    if args.limit is not None:
        pairs = pairs[: args.limit]
    total = len(pairs)

    if args.resume:
        rows, finished, _ = load_resume_state(results_csv, pairs)
        remaining = [p for p in pairs if p["sample_id"] not in finished]
        last_sid = rows[-1]["sample_id"] if rows else None
        print(
            f"Resume: {len(finished):,}/{total:,} completed; last_sample={last_sid}; "
            f"remaining {len(remaining):,} (including any earlier non-ok retries)",
            flush=True,
        )
    else:
        rows, finished, remaining = [], set(), pairs
        print(f"Fresh run: {total:,} samples -> {output_dir}", flush=True)

    def write_status(phase: str, success: int, **extra) -> None:
        payload = {
            "phase": phase,
            "total": total,
            "success": success,
            "updated": time.time(),
            "output_dir": str(output_dir),
            "rec_batch_num": args.rec_batch_num,
            "device": "CUDA",
            **extra,
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ocr = create_ocr(args.rec_batch_num)
    cpu_ocr: RapidOCR | None = None
    cpu_fallbacks = 0
    write_status("running", len(finished))

    t0, done, verified_providers, need_verify = time.perf_counter(), 0, None, True
    pbar = tqdm(remaining, initial=len(finished), total=total, desc=f"RapidOCR (GPU bs={args.rec_batch_num})")

    for sample in pbar:
        if sample["sample_id"] in finished:
            continue
        row = {**sample, "model_id": model_id}
        try:
            used_cpu = False
            start = time.perf_counter()
            try:
                out = ocr(sample["image_path"])
            except Exception:
                # Reset GPU ORT session and retry once before CPU fallback
                del ocr
                gc.collect()
                ocr = create_ocr(args.rec_batch_num)
                need_verify = True
                start = time.perf_counter()
                try:
                    out = ocr(sample["image_path"])
                except Exception:
                    if cpu_ocr is None:
                        cpu_ocr = create_cpu_ocr(8)
                    start = time.perf_counter()
                    out = cpu_ocr(sample["image_path"])
                    used_cpu = True
                    cpu_fallbacks += 1
            latency = time.perf_counter() - start
            if not used_cpu and need_verify:
                verified_providers = verify_cuda_providers(ocr)
                need_verify = False
            lines = out.txts or ()
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
            pbar.set_postfix(
                ok=len(finished),
                lat=f"{latency:.2f}s",
                cer=f"{row['cer']:.3f}",
                refresh=False,
            )
        except Exception as exc:
            row.update({"status": "error", "error": repr(exc)})
            pbar.set_postfix(ok=len(finished), err=len(rows) + 1 - len(finished), refresh=False)

        rows.append(row)
        done += 1

        if done % 10 == 0:
            write_status("running", len(finished), last_sample=sample["sample_id"], last_status=row["status"], cpu_fallbacks=cpu_fallbacks)

        if done % args.save_every == 0:
            save_csv_atomic(rows, results_csv)
            gc.collect()

        # Reset ORT CUDA allocator cache periodically or when VRAM exceeds 4.8 GB
        if done % 250 == 0 or (done % 100 == 0 and query_gpu_stats().get("gpu_memory_used_mb", 0) >= 4800):
            del ocr
            gc.collect()
            ocr = create_ocr(args.rec_batch_num)
            need_verify = True

    save_csv_atomic(rows, results_csv)
    gpu_stats = query_gpu_stats()
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
        "rec_batch_num": args.rec_batch_num,
        "cpu_fallbacks": cpu_fallbacks,
        "device": "CUDA",
        "providers": verified_providers,
        **gpu_stats,
        "runtime_sec_this_run": time.perf_counter() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {results_csv}", flush=True)
    write_status("completed" if summary["n_errors"] == 0 else "errors", len(ok), errors=summary["n_errors"], cpu_fallbacks=cpu_fallbacks)
    if summary["n_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
