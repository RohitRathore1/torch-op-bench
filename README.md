# torch-op-bench

Quick check of when `torch.compile` helps a single PyTorch op and when it doesn't.
Times eager vs compiled over a size sweep, a few dtypes, writes jsonl, plots it.

```
pip install -r requirements.txt
./run.sh                          # 3 runs + plots
./run.sh --mode reduce-overhead   # anything after run.sh goes to bench.py
python bench.py --ops softmax --iters 20 --reps 3   # quick single run
```

Output: `results/<run_id>/results.jsonl` + `snapshot.json` (torch/cuda/gpu/tf32 state),
`plots/<stamp>/<op>_time.png` and `<op>_speedup.png`.

Ops live in the `OPS` dict at the top of `bench.py`. Add an entry, run, look at the plot.

Findings and caveats are in NOTES.md. Short version: on a single-kernel op compile
adds ~35us per call and otherwise ties eager; it only wins where it can fuse kernels
or where ATen's kernel has a weak spot.

Needs a GPU. On the team dev box: `gpu-shell dev` first, then `source /run/gpu-slice/$USER`.
