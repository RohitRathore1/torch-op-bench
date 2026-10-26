import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def matmul_inputs(n, dt):
    return torch.randn(n, n, dtype=dt, device="cuda"), torch.randn(n, n, dtype=dt, device="cuda")


def softmax_inputs(size, dt):
    rows, cols = size
    return (torch.randn(rows, cols, dtype=dt, device="cuda"),)


OPS = {
    "matmul": dict(
        inputs=matmul_inputs,
        fn=lambda a, b: a @ b,
        sizes=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        flops=lambda n: 2 * n ** 3,
        bytes=lambda n, dt: 3 * n * n * dt.itemsize,
    ),
    # rows fixed, cols swept. plain softmax: eager is already one kernel, so this is
    # inductor's reduction vs ATen's, not a fusion test.
    "softmax": dict(
        inputs=softmax_inputs,
        fn=lambda x: torch.softmax(x, dim=-1),
        sizes=[(4096, c) for c in (64, 256, 1024, 4096, 16384, 65536)],
        flops=lambda s: 5 * s[0] * s[1],
        bytes=lambda s, dt: 2 * s[0] * s[1] * dt.itemsize,
        atol=1e-3,
    ),
    # TODO: layernorm, scale+mask+softmax, linear+gelu (the fusion cases)
}

def time_block(fn, args, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def measure(fn, args, warmup, iters, reps, flush=None):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        t.append(time_block(fn, args, iters))
    t.sort()
    q = statistics.quantiles(t, n=4) if len(t) > 1 else t * 3
    return dict(median_ms=statistics.median(t), q1_ms=q[0], q3_ms=q[2],
                min_ms=t[0], max_ms=t[-1], reps=reps, iters=iters)


def measure_compiled(op, args, mode, **kw):
    import torch._dynamo as dynamo
    dynamo.reset() 
    for a in args:
        dynamo.mark_static(a)
    f = torch.compile(op["fn"], mode=mode, dynamic=False)

    t0 = time.perf_counter()
    out = f(*args)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    ref = op["fn"](*args)
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), rtol=op.get("rtol", 1e-2), atol=op.get("atol", 1e-2))

    m = measure(f, args, **kw)
    m.update(compile_s=compile_s, max_abs_err=err, numerics_ok=ok)
    return m


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def snapshot():
    p = torch.cuda.get_device_properties(0)
    try:
        import triton
        triton_v = triton.__version__
    except ImportError:
        triton_v = None
    return dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        host=platform.node(),
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=triton_v,
        driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        gpu=p.name,
        sms=p.multi_processor_count,  # on MIG this is the slice size
        mem_gb=round(p.total_memory / 2 ** 30, 1),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        fp32_matmul_precision=torch.get_float32_matmul_precision(),
        git=sh("git rev-parse --short HEAD") or None,
    )


def size_key(size):
    return "x".join(map(str, size)) if isinstance(size, (tuple, list)) else str(size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="*", default=list(OPS))
    ap.add_argument("--dtypes", nargs="*", default=list(DTYPES))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--mode", default=None, help="torch.compile mode: reduce-overhead, max-autotune")
    ap.add_argument("--flush-l2", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="random config order, to catch drift")
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no GPU"
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = snapshot()
    snap["args"] = vars(args)
    (out_dir / "snapshot.json").write_text(json.dumps(snap, indent=2))
    print(f"[{run_id}] {snap['gpu']} ({snap['sms']} SMs)  torch {snap['torch']}  cuda {snap['cuda']}  mode={args.mode}")

    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda") if args.flush_l2 else None
    kw = dict(warmup=args.warmup, iters=args.iters, reps=args.reps, flush=flush)

    configs = [(o, d, s) for o in args.ops for d in args.dtypes for s in OPS[o]["sizes"]]
    if args.shuffle:
        random.shuffle(configs)

    with (out_dir / "results.jsonl").open("w") as f:
        for op_name, dt_name, size in configs:
            op, dt = OPS[op_name], DTYPES[dt_name]
            torch.manual_seed(0)
            inputs = op["inputs"](size, dt)

            # eager and compiled back to back, so clock drift hits both the same
            with torch.no_grad():
                eager = measure(op["fn"], inputs, **kw)
                comp = measure_compiled(op, inputs, args.mode, **kw)

            for mode, m in (("eager", eager), ("compiled", comp)):
                row = dict(run_id=run_id, op=op_name, dtype=dt_name, size=size_key(size),
                           size_raw=size, mode=mode, **m)
                if "flops" in op:
                    row["tflops"] = op["flops"](size) / (m["median_ms"] * 1e-3) / 1e12
                if "bytes" in op:
                    row["gbps"] = op["bytes"](size, dt) / (m["median_ms"] * 1e-3) / 1e9
                f.write(json.dumps(row) + "\n")

            flag = "" if comp["numerics_ok"] else "  !! numerics mismatch"
            print(f"{op_name:10s} {dt_name:5s} {size_key(size):>12s}  "
                  f"eager {eager['median_ms']:8.4f} ms  compiled {comp['median_ms']:8.4f} ms  "
                  f"x{eager['median_ms'] / comp['median_ms']:5.2f}  (compile {comp['compile_s']:.1f}s){flag}")

            del inputs
            torch.cuda.empty_cache()

    print(f"wrote {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
