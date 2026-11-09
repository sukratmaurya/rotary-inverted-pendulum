"""Read-only numerical equivalence check for a distilled policy deployment.

This tool never exports or rewrites weights. It reads an optional PyTorch
student checkpoint and an existing PROGMEM header, parses the header's float32
literals, and compares Python, NumPy and a host-compiled scalar C++ forward
pass. The C++ harness and executable live only in a temporary directory.

Example:
    python validate_deployment.py \
        --header ../firmware/RLControl/policy_weights_rig1_tmc2209_32.h \
        --student runs/<run>/distill_h16_dagger_dev/student.pt \
        --dataset runs/<run>/distill_h16_dagger_dev/dataset.npz

Without --student, the Python reference is constructed from the parsed header
weights. That validates header parsing and forward-pass parity, but cannot
prove that the header came from a particular checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ARRAY_NAMES = ("POLICY_W1", "POLICY_B1", "POLICY_W2", "POLICY_B2",
               "POLICY_W3", "POLICY_B3")
FLOAT_RE = re.compile(r"[+-]?(?:(?:\d+\.?(?:\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")


def parse_header(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int], str | None]:
    text = path.read_text(encoding="utf-8")
    dims: dict[str, int] = {}
    for macro in ("POLICY_OBS_DIM", "POLICY_HIDDEN_DIM", "POLICY_OUT_DIM"):
        match = re.search(rf"^\s*#define\s+{macro}\s+(\d+)\s*$", text, re.M)
        if match:
            dims[macro] = int(match.group(1))
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        match = re.search(
            rf"const\s+float\s+{name}\s*\[[^]]+\](?:\s*\[[^]]+\])?\s*"
            rf"PROGMEM\s*=\s*\{{(.*?)\}}\s*;",
            text, re.S,
        )
        if not match:
            raise ValueError(f"missing or unsupported array declaration: {name}")
        body = re.sub(r"//[^\n]*|/\*.*?\*/", "", match.group(1), flags=re.S)
        values = [np.float32(float(x)) for x in FLOAT_RE.findall(body)]
        arrays[name] = np.asarray(values, dtype=np.float32)
    source = re.search(r"^\s*//\s*source:\s*(.*?)\s*$", text, re.M)
    return arrays, dims, source.group(1).strip() if source else None


def checkpoint_weights(path: Path, device: str) -> tuple[dict[str, np.ndarray], dict]:
    try:
        import torch
        from distill import StudentMLP
    except ImportError as exc:
        raise RuntimeError("PyTorch is required when --student is supplied") from exc
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = StudentMLP(hidden=int(ckpt["hidden"]), obs_dim=int(ckpt["obs_dim"]),
                       act_dim=int(ckpt["act_dim"])).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    sd = model.state_dict()
    weights = {
        "POLICY_W1": sd["fc1.weight"].detach().cpu().numpy().astype(np.float32).ravel(),
        "POLICY_B1": sd["fc1.bias"].detach().cpu().numpy().astype(np.float32).ravel(),
        "POLICY_W2": sd["fc2.weight"].detach().cpu().numpy().astype(np.float32).ravel(),
        "POLICY_B2": sd["fc2.bias"].detach().cpu().numpy().astype(np.float32).ravel(),
        "POLICY_W3": sd["fc3.weight"].detach().cpu().numpy().astype(np.float32).ravel(),
        "POLICY_B3": sd["fc3.bias"].detach().cpu().numpy().astype(np.float32).ravel(),
    }
    return weights, {"hidden": int(ckpt["hidden"]), "obs_dim": int(ckpt["obs_dim"]),
                     "act_dim": int(ckpt["act_dim"]),
                     "val_mse": float(ckpt.get("val_mse", float("nan"))),
                     "model": model, "torch": torch}


def numpy_forward(weights: dict[str, np.ndarray], obs: np.ndarray,
                  obs_dim: int, hidden: int) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32).reshape(-1, obs_dim)
    w1 = weights["POLICY_W1"].reshape(hidden, obs_dim)
    b1 = weights["POLICY_B1"].reshape(hidden)
    w2 = weights["POLICY_W2"].reshape(hidden, hidden)
    b2 = weights["POLICY_B2"].reshape(hidden)
    w3 = weights["POLICY_W3"].reshape(1, hidden)
    b3 = weights["POLICY_B3"].reshape(1)
    h1 = np.maximum(np.float32(0), x @ w1.T + b1)
    h2 = np.maximum(np.float32(0), h1 @ w2.T + b2)
    return np.tanh(h2 @ w3.T + b3).astype(np.float32)


def edge_corpus(obs_dim: int, *, seed: int, random_count: int) -> np.ndarray:
    if obs_dim != 24:
        raise ValueError(f"expected the deployed 24-feature observation, got {obs_dim}")
    frames = 4
    frame_dim = 6
    rows: list[np.ndarray] = []
    # Typical reset and upright observations, replicated through all frames.
    for motor in (0.0, -2.30383, 2.30383):
        for sin_theta, cos_theta in ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)):
            f = np.asarray([motor, sin_theta, cos_theta, 0, 0, 0], np.float32)
            rows.append(np.tile(f, frames))
    # Per-frame feature impulses exercise ordering and every input column.
    for frame in range(frames):
        for feature, magnitude in enumerate((2.30383, 1.0, 1.0, 200.0, 200.0, 1.0)):
            for sign in (-1.0, 1.0):
                row = np.zeros(obs_dim, np.float32)
                row[frame * frame_dim + feature] = np.float32(sign * magnitude)
                rows.append(row)
    alternating = np.zeros(obs_dim, np.float32)
    alternating[0:frame_dim] = [2.30383, 1, 0, 200, -200, 1]
    alternating[frame_dim:2 * frame_dim] = [-2.30383, -1, 0, -200, 200, -1]
    alternating[2 * frame_dim:3 * frame_dim] = [0, 0, 1, 0, 0, 0]
    alternating[3 * frame_dim:4 * frame_dim] = [0, 0, -1, 0, 0, 0]
    rows.append(alternating)
    rng = np.random.default_rng(seed)
    low = np.tile(np.asarray([-2.30383, -1, -1, -200, -200, -1], np.float32), frames)
    high = -low
    if random_count:
        rows.extend(rng.uniform(low, high, size=(random_count, obs_dim)).astype(np.float32))
    return np.stack(rows).astype(np.float32)


def load_dataset(path: Path, obs_dim: int, max_rows: int, seed: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "obs" not in data:
            raise ValueError(f"dataset {path} has no 'obs' array")
        obs = np.asarray(data["obs"], dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != obs_dim:
        raise ValueError(f"dataset obs shape {obs.shape}; expected (N, {obs_dim})")
    if max_rows > 0 and len(obs) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(obs), max_rows, replace=False))
        obs = obs[idx]
    if not np.isfinite(obs).all():
        raise ValueError("dataset contains NaN or infinite observation values")
    return obs


def cpp_source(weights: dict[str, np.ndarray], obs_dim: int, hidden: int) -> str:
    def literals(values: np.ndarray) -> str:
        return ",".join(f"{float(v):.9g}f" for v in values)
    arrays = "\n".join(f"static const float {n}[{len(weights[n])}] = {{{literals(weights[n])}}};"
                        for n in ARRAY_NAMES)
    return f'''#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <vector>\nstatic const int D={obs_dim}, H={hidden};\n{arrays}\nint main() {{\n  std::vector<float> x(D);\n  while (std::fread(x.data(), sizeof(float), D, stdin) == (size_t)D) {{\n    float h1[H], h2[H];\n    for (int i=0;i<H;i++) {{ float s=POLICY_B1[i]; for(int j=0;j<D;j++) s += x[j]*POLICY_W1[i*D+j]; h1[i]=s>0.0f?s:0.0f; }}\n    for (int i=0;i<H;i++) {{ float s=POLICY_B2[i]; for(int j=0;j<H;j++) s += h1[j]*POLICY_W2[i*H+j]; h2[i]=s>0.0f?s:0.0f; }}\n    float s=POLICY_B3[0]; for(int j=0;j<H;j++) s += h2[j]*POLICY_W3[j];\n    float a=::tanhf(s); if(a>1.0f)a=1.0f;else if(a< -1.0f)a=-1.0f;\n    if(std::fwrite(&a,sizeof(float),1,stdout)!=1) return 3;\n  }}\n  return std::feof(stdin)?0:2;\n}}\n'''


def run_cpp(weights: dict[str, np.ndarray], obs: np.ndarray, obs_dim: int,
            hidden: int, compiler: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="rip-policy-validate-") as tmp:
        root = Path(tmp)
        src = root / "forward.cpp"
        exe = root / "forward"
        src.write_text(cpp_source(weights, obs_dim, hidden), encoding="utf-8")
        cmd = [compiler, "-std=c++11", "-O2", "-ffp-contract=off", str(src), "-o", str(exe)]
        built = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(f"C++ compile failed: {built.stderr.strip()}")
        run = subprocess.run([str(exe)], input=obs.astype("<f4").tobytes(),
                             capture_output=True, check=False)
        if run.returncode:
            raise RuntimeError(f"C++ harness exited {run.returncode}: {run.stderr.decode(errors='replace')}")
        expected = len(obs) * np.dtype(np.float32).itemsize
        if len(run.stdout) != expected:
            raise RuntimeError(f"C++ returned {len(run.stdout)} bytes, expected {expected}")
        return np.frombuffer(run.stdout, dtype=np.float32).copy().reshape(-1, 1)


def metrics(label: str, ref: np.ndarray, got: np.ndarray) -> dict:
    diff = np.abs(np.asarray(ref, np.float32).reshape(-1) -
                  np.asarray(got, np.float32).reshape(-1))
    return {"comparison": label, "samples": int(diff.size),
            "max_abs": float(diff.max(initial=0)),
            "rms": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
            "p99_abs": float(np.percentile(diff, 99)) if diff.size else 0.0,
            "max_index": int(diff.argmax()) if diff.size else -1}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--header", type=Path, required=True)
    p.add_argument("--student", type=Path, help="optional .pt checkpoint for provenance/weight comparison")
    p.add_argument("--dataset", type=Path, help="optional NPZ containing an obs array")
    p.add_argument("--max-dataset-rows", type=int, default=4096)
    p.add_argument("--random-edge-rows", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compiler", default=shutil.which("g++") or shutil.which("clang++") or "c++")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    arrays, dims, source = parse_header(args.header)
    required = {"POLICY_OBS_DIM", "POLICY_HIDDEN_DIM", "POLICY_OUT_DIM"}
    if set(dims) != required:
        raise ValueError(f"header is missing dimension macros: {sorted(required - set(dims))}")
    d, h, out = dims["POLICY_OBS_DIM"], dims["POLICY_HIDDEN_DIM"], dims["POLICY_OUT_DIM"]
    if d != 24 or out != 1:
        raise ValueError(f"deployment contract requires obs_dim=24 and act_dim=1; got {d}->{out}")
    expected_counts = {"POLICY_W1": h*d, "POLICY_B1": h, "POLICY_W2": h*h,
                       "POLICY_B2": h, "POLICY_W3": h*out, "POLICY_B3": out}
    for name, count in expected_counts.items():
        if arrays[name].size != count:
            raise ValueError(f"{name} has {arrays[name].size} values, expected {count}")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("header contains NaN or infinite weights")

    report: dict = {"header": str(args.header), "declared_source": source,
                    "checkpoint_status": "not supplied; provenance is declared by header only",
                    "architecture": [d, h, h, out], "activation": "ReLU/ReLU/tanh",
                    "weight_mapping": "row-major [out, in], matching PyTorch Linear",
                    "numeric": "header decimal literals parsed to float32; C++ scalar float, FMA disabled",
                    "comparisons": []}
    if args.student:
        ckpt_path = args.student
        ckpt_weights, meta = checkpoint_weights(ckpt_path, args.device)
        if (meta["obs_dim"], meta["hidden"], meta["act_dim"]) != (d, h, out):
            raise ValueError("checkpoint dimensions do not match header dimensions")
        for name in ARRAY_NAMES:
            if not np.array_equal(ckpt_weights[name], arrays[name]):
                delta = np.abs(ckpt_weights[name] - arrays[name])
                report["checkpoint_status"] = "weights differ after float32 parse"
                report.setdefault("weight_differences", {})[name] = float(delta.max(initial=0))
        if report["checkpoint_status"] != "weights differ after float32 parse":
            report["checkpoint_status"] = "checkpoint arrays equal parsed header float32 arrays"
        report["checkpoint"] = str(ckpt_path)
        report["checkpoint_val_mse"] = meta["val_mse"]
        report["python_reference"] = "PyTorch StudentMLP loaded from checkpoint"
        torch_model, torch = meta["model"], meta["torch"]
    else:
        from distill import StudentMLP
        import torch
        torch_model = StudentMLP(hidden=h, obs_dim=d, act_dim=out).eval()
        with torch.no_grad():
            for key, param in (("POLICY_W1", torch_model.fc1.weight),
                               ("POLICY_B1", torch_model.fc1.bias),
                               ("POLICY_W2", torch_model.fc2.weight),
                               ("POLICY_B2", torch_model.fc2.bias),
                               ("POLICY_W3", torch_model.fc3.weight),
                               ("POLICY_B3", torch_model.fc3.bias)):
                param.copy_(torch.from_numpy(arrays[key].reshape(tuple(param.shape))))
        report["python_reference"] = "PyTorch StudentMLP populated from parsed header"

    samples = edge_corpus(d, seed=args.seed, random_count=args.random_edge_rows)
    source_dataset_rows = 0
    if args.dataset:
        dataset = load_dataset(args.dataset, d, args.max_dataset_rows, args.seed)
        source_dataset_rows = len(dataset)
        samples = np.concatenate((samples, dataset), axis=0)
        report["dataset"] = str(args.dataset)
    report["edge_corpus_rows"] = int(len(samples) - source_dataset_rows)
    report["dataset_rows"] = source_dataset_rows
    if not np.isfinite(samples).all():
        raise ValueError("validation observations contain NaN or infinite values")
    with torch.no_grad():
        py = torch_model(torch.from_numpy(samples)).cpu().numpy().astype(np.float32)
    numpy_out = numpy_forward(arrays, samples, d, h)
    cpp_out = run_cpp(arrays, samples, d, h, args.compiler)
    report["comparisons"].append(metrics("PyTorch vs NumPy(header weights)", py, numpy_out))
    report["comparisons"].append(metrics("PyTorch vs compiled C++ scalar(header weights)", py, cpp_out))
    report["comparisons"].append(metrics("NumPy vs compiled C++ scalar", numpy_out, cpp_out))
    report["compiler"] = args.compiler
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
