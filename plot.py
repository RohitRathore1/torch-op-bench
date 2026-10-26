import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def xval(size_raw):
    if isinstance(size_raw, (int, float)):
        return size_raw
    n = 1
    for d in size_raw:
        n *= d
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()

    rows = []
    for d in args.runs:
        p = Path(d) / "results.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_runs = len({r["run_id"] for r in rows})

    # g[op][dtype][mode][size] -> rows (one per run)
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        g[r["op"]][r["dtype"]][r["mode"]][r["size"]].append(r)

    for op, by_dt in g.items():
        dts = sorted(by_dt)
        fig, axes = plt.subplots(1, len(dts), figsize=(6 * len(dts), 4.5), squeeze=False)
        fig2, axes2 = plt.subplots(1, len(dts), figsize=(6 * len(dts), 4), squeeze=False)

        for ax, ax2, dt in zip(axes[0], axes2[0], dts):
            med = {}
            for mode in ("eager", "compiled"):
                pts = []
                for size, rs in by_dt[dt][mode].items():
                    x = xval(rs[0]["size_raw"])
                    ms = sorted(r["median_ms"] for r in rs)
                    y = ms[len(ms) // 2]
                    if n_runs > 1:
                        lo, hi = y - ms[0], ms[-1] - y
                    else:
                        lo, hi = y - rs[0]["q1_ms"], rs[0]["q3_ms"] - y
                    pts.append((x, y, max(lo, 0), max(hi, 0)))
                    med[mode, x] = y
                pts.sort()
                xs, ys, los, his = zip(*pts)
                ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=mode)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{op} {dt}")
            ax.set_xlabel("size")
            ax.set_ylabel("ms / iter (median)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

            xs = sorted({x for _, x in med})
            ax2.plot(xs, [med["eager", x] / med["compiled", x] for x in xs], marker="o")
            ax2.axhline(1.0, color="gray", ls="--")
            ax2.set_xscale("log", base=2)
            ax2.set_title(f"{op} {dt}: eager / compiled  (>1 = compile helps)")
            ax2.set_xlabel("size")
            ax2.grid(True, alpha=0.3)

        bars = "run-to-run min/max" if n_runs > 1 else "IQR of reps"
        fig.suptitle(f"{op}  ({n_runs} run(s), error bars = {bars})")
        for f_, name in ((fig, "time"), (fig2, "speedup")):
            f_.tight_layout()
            f_.savefig(out / f"{op}_{name}.png", dpi=130)
            plt.close(f_)
        print(f"wrote {out}/{op}_time.png {out}/{op}_speedup.png")


if __name__ == "__main__":
    main()
