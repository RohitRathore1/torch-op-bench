# Working notes

Write here as you go. This is the material for the walkthrough.

Runs so far (2026-10-26, PyTorch dev node 163.75.83.85, MIG 3g.71gb slice of an H200):
- 20261026-124504: default compile mode, 3 runs, matmul fp32 + bf16, N = 64..8192
- 20261026-125744: --compile-mode reduce-overhead, same configs
- 20261026-131133: default mode, matmul + softmax (softmax: 4096 rows, 64..65536 cols)
Plots in plots/<run id>/, raw numbers in results/<run id>-r{1,2,3}/results.jsonl.

## Key takeaways

If you only read one section, read this one.

The biggest lesson is that torch.compile is not a speed button. Wrapping a single
operator in it adds roughly 35 microseconds to every call, and beyond that it mostly
just matches eager. It earns its keep in two situations: when it can fuse several
small kernels into one, so whole launches disappear, or when the operator is so
large that 35 microseconds is noise. Neither of those describes a lone matmul, and
only sometimes describes a lone softmax. That is why our plots spent most of their
time below the line.

Before you read any speedup ratio, work out which regime you are in. If a call takes
less than about 50 microseconds, you are measuring how long it takes Python and the
driver to launch work, not how long the GPU takes to do it. Above that, you are
measuring the kernel. A ratio of 0.2 at ten microseconds and a ratio of 1.3 at one
millisecond are not the same kind of fact, and the first one should not scare you.

Always look at the baseline, not just the ratio. The most convincing bf16 softmax
win we saw turned out to be a shape where eager was unusually slow, taking as long
as fp32 for half the bytes. Compile looked good because the thing it was being
compared to was bad. Checking achieved bandwidth against what the hardware can do
catches this in seconds.

Compile modes are tools for specific workloads, and the names can mislead. Reduce
overhead uses CUDA graphs, which are excellent when the CPU cannot issue a long
chain of small kernels fast enough. Wrapped around one kernel, the bookkeeping and
the input copies it adds cost more than the launch it saves. It made every number
worse. Try a mode, measure it, and do not assume the name tells you the outcome.

The quickest sanity check for a memory-bound kernel is to halve the data type width
and see whether time halves too. Eager softmax passed that test. Compiled bf16
softmax did not, and that one observation told us it was limited by something
other than bandwidth, which is exactly where a profiler should go next.

Record the environment next to every number. We were on a MIG slice, not a full
GPU. TF32 was off. Triton was a specific version. Torch was a specific commit. Any
one of those changes the numbers, and without the snapshot file beside the results
half of this document would be misread in a month. The one thing we still do not
record is our own git commit, because the project is not a git repo yet.

When you compare two things, interleave them. We time eager and compiled back to
back for each shape, rather than all eager runs then all compiled runs. That is the
main reason three independent runs on a shared box agree to three decimal places.

Keep compile time separate from per-call time and say which one you mean. Compiling
a shape cost half a second to a second here, then nothing on later runs because the
cache persists. That is a real cost but it is not a per-call cost, and mixing the
two is how people conclude both that compile is slow and that compile is fast from
the same data.

Expect the environment to eat the first half day. GPU access was locked behind a
scheduler, triton and matplotlib were missing, a directory the plotting code
assumed did not exist. None of it was PyTorch, all of it stopped PyTorch from
running. The fixes are written down above so nobody pays that again.

A negative result with a reason is a result. Saying that compile does not help
matmul because matmul is already one cuBLAS kernel and compile only adds guard
checks is more useful to the team than a plot that goes up and to the right with
no explanation attached.

Add one operator at a time and run the whole pipeline after each. Softmax was
thirty lines of code, and adding it surfaced a plotting bug and a bf16 regression.
Adding three operators at once would have buried both.

And finally, whenever a number surprises you, ask what the kernel is bound by:
launch, Python, bandwidth, compute, or copies. Every confusing result in this file
stopped being confusing once we answered that question.

## Decisions I made in the harness and why

- Layout: one bench.py (ops dict + timing + snapshot + main), one plot.py, run.sh.
  Started as a bench/ package with a registry and dataclasses; flattened it after
  the second op because it was more structure than two ops and 300 lines deserve.
  Promote back to a package if/when this becomes a real tool.


- Timing controls, all in bench.py: CUDA events +
  sync, warmup, compile timed separately, blocks of iters repeated `reps` times with
  median/IQR, optional L2 flush, allclose check vs eager, dynamo reset per config.
- Eager and compiled are measured back-to-back for the same config so clock/thermal
  drift hits both. Three full runs, plotted together, show run-to-run spread.
- Plot output dir is now created with parents=True. The plots/ dir is gitignored and
  never existed on a fresh checkout, so the very first `./run.sh` burned three
  benchmark runs and then died in the plot step. Fixed in plot.py.
- Softmax op (OPS dict in bench.py): rows fixed at 4096, cols swept 64 -> 65536, softmax over cols. Rows
  fixed so the only thing changing is row length, which is what decides how the
  reduction kernel is scheduled. Plain torch.softmax, no scale/mask, on purpose:
  eager softmax is already one fused kernel, so this asks "can inductor's generated
  reduction match ATen's hand-written one", not "does fusion help".
- Tuple sizes plot as total elements (rows*cols) on the x axis. plot.py used to
  take max(size), which would have put 4096x64 and 4096x4096 on the same x.
- Plain `./run.sh` defaults (no --flush-l2, no --shuffle, compile mode default) for
  the first pass. Flush/shuffle are the next thing to try.

## Things that went wrong / numbers I didn't trust at first

- No GPU from a plain SSH shell. nvidia-smi said "Failed to initialize NVML: Unknown
  Error", torch.cuda.is_available() was False, cuInit returned 100 (NO_DEVICE), and
  opening /dev/nvidiactl gave EPERM despite 0666 perms. Cause: a systemd drop-in
  (/etc/systemd/system/user-<uid>.slice.d/10-gpu-deny.conf) sets DevicePolicy=closed
  on every dev user's slice. GPUs are only reachable via Slurm: `gpu-shell dev`
  reserves a MIG slice for 12h, then `source /run/gpu-slice/$USER` in any shell.
- System python has no torch. The only usable interpreter is the from-source build
  in ~/.conda/envs/pytorch_nccl29_h200 (torch 2.11.0a0+git1057af0, CUDA 12.8).
- That env had no triton, so torch.compile died with TritonMissing on the first
  config. Installed triton==3.6.0 (the version pinned in
  pytorch/.ci/docker/triton_version.txt). Latest PyPI triton is 3.8.0; did not risk
  the mismatch. matplotlib was also missing; pip installed it.
- pip retries for ~30s on every install because the configured extra index
  (pypi.ngc.nvidia.com) does not resolve on this node. Use
  `--index-url https://pypi.org/simple` to skip it.
- Compiled matmul is 3-4x SLOWER than eager below N~1024 (bf16) / N~512 (fp32).
  First instinct was a measurement bug. It is not: compiled time is a flat ~35 us
  floor regardless of size while eager is ~10 us, i.e. fixed per-call overhead of
  the compiled wrapper/guards, not a worse kernel. Above that size both are 1.00x
  because both end up in the same cuBLAS GEMM.
- reduce-overhead made EVERYTHING slower, not faster. Small sizes dropped from ~0.3x
  to ~0.17x (compiled floor rose from ~35 us to ~55 us per call), and even bf16 8192
  went from 1.00x to 0.89x (2.62 -> 2.96 ms). I did not believe a CUDA-graph mode
  could lose at the compute-bound end, so I pulled it apart (see Surprises).
- The box is shared. Other users held 3 dev slices and were running during the
  benchmark. Within-slice noise was still small (3-run min/max bars are invisible on
  the plot), but absolute numbers should not be compared to a different node.

## Surprises

- The "GPU" is a MIG 3g.71gb slice (60 SMs, 71 GB), not a full H200. Inductor
  logged "Not enough SMs to use max_autotune_gemm mode", so --compile-mode
  max-autotune will be silently degraded here. The perf partition (whole GPU) needs
  the perfusers group, which this account is not in.
- TF32 is off for matmul by default (float32_matmul_precision=highest), so the fp32
  line is true fp32: 8192^3 took 54.8 ms, ~20 TFLOPS. bf16 at the same size was
  2.62 ms, ~420 TFLOPS. The harness snapshot records this, which is good; the
  walkthrough should say which fp32 is meant.
- Compile times were tiny (0.0-0.7 s) after the first config because inductor's
  on-disk cache persists across runs r1/r2/r3. Only r1's first config paid ~0.7 s.
- Snapshot's repo_git_commit is "<unavailable>" because the project is not a git
  repo. Worth a `git init` before the next run so provenance is recorded.

- reduce-overhead, taken apart (bf16, same harness timing, same slice):

      n=256:   eager 9.2 us | manual CUDAGraph replay 3.5 us | replay + copy inputs 12.6 us
               | default-mode compiled 32 us | reduce-overhead compiled 59 us
      n=8192:  eager 2.645 ms | replay 2.638 ms | replay + copy inputs 2.874 ms
               | default-mode compiled 2.625 ms | reduce-overhead compiled 2.956 ms

  Three things fall out of this:
  1. The graph itself is the fastest way to run the kernel (3.5 us vs 9.2 us eager).
     CUDA graphs do work.
  2. torch.compile's reduce-overhead path copies every input into graph-owned memory
     on every call. At 8192 that is two 134 MB copies, ~0.23 ms, which is almost all
     of the 0.31 ms gap at the large end. mark_static_address on the inputs did not
     remove the copy in this build; cudagraph_mark_step_begin changed nothing either.
  3. The rest (~50 us at small sizes) is Python: dynamo guards + the cudagraph-trees
     bookkeeping that runs before each replay. For a single kernel there is no
     launch gap for the graph to hide, so you pay the wrapper and get nothing back.
  reduce-overhead is for long chains of small kernels where the CPU cannot keep the
  GPU fed. One matmul is the opposite of that workload.
- TORCH_LOGS="cudagraphs" confirms a cudagraph tree was recorded for each config, so
  the slowdown is not "graphs were skipped". Worth keeping that log on when in doubt.

## Softmax: the first op where compile wins, and where it doesn't

Medians over 3 runs, MIG 3g.71gb slice. GB/s = (read + write) / time, the ideal
one-pass count, so it is a lower bound on what the kernel actually moved.

      cols   fp32 eager   fp32 compiled   speedup | bf16 eager   bf16 compiled   speedup
        64    0.006 ms      0.035 ms       0.16x  |  0.005 ms      0.034 ms       0.16x
       256    0.006 ms      0.035 ms       0.16x  |  0.006 ms      0.032 ms       0.19x
      1024    0.019 ms      0.035 ms       0.54x  |  0.017 ms      0.035 ms       0.49x
      4096    0.118 ms      0.091 ms       1.30x  |  0.117 ms      0.089 ms       1.32x
     16384    0.335 ms      0.297 ms       1.13x  |  0.267 ms      0.313 ms       0.85x
     65536    1.948 ms      1.476 ms       1.32x  |  0.928 ms      1.159 ms       0.80x

      achieved GB/s      fp32 eager / compiled        bf16 eager / compiled
      cols=4096              1140 / 1481                  571 / 752
      cols=16384             1605 / 1806                 1004 / 858
      cols=65536             1102 / 1455                 1157 / 926

What this says:

- Same ~35 us compiled floor as matmul. Eager softmax on short rows is even cheaper
  than eager matmul (5-6 us), so the small-size ratio is worse (0.16x). The guard
  overhead is per call and does not care which op is inside.
- fp32: compile wins from 4096 cols up, 1.1x to 1.3x. Inductor's fp32 reduction hits
  ~1.8 TB/s at 16384 cols, the highest bandwidth seen on this slice so far. ATen's
  fp32 kernel falls off at 65536 cols (1.1 TB/s), inductor's does not as much. This
  is the first honest "compile helps" result in the project.
- bf16 at 4096 cols: compile wins 1.32x, but look at the eager number. Eager bf16
  takes the same 0.117 ms as eager fp32 while moving half the bytes (571 GB/s). ATen's
  bf16 softmax has a bad spot at this row length. Compile "wins" because eager is
  slow there, not because the compiled kernel is fast (752 GB/s is still under the
  fp32 compiled number).
- bf16 wide rows: compile LOSES, 0.85x and 0.80x. Compiled bf16 at 65536 cols takes
  1.16 ms vs 1.48 ms for fp32: half the bytes, 78% of the time. So the compiled bf16
  kernel is not bandwidth-bound; something else (upcast to fp32 + exp throughput,
  or an extra pass over the row) is the limit. Eager bf16 does scale with bytes
  (0.93 ms vs 1.95 ms), so ATen's wide-row bf16 path is the better kernel here.
- Numerics all passed. Max abs error vs eager was 1e-4 at worst (bf16, 64 cols)
  and below 1e-8 for fp32.
- Run-to-run spread is invisible on the plot again, so the bf16 regression is real,
  not noise.

Takeaway for the walkthrough: compile beats a hand-written single kernel only when
that kernel has a weak spot (fp32 wide rows, bf16 at 4096). Against ATen's good
paths it is at best equal and sometimes 20% behind. The fusion story, where compile
removes whole kernels, has not been tested yet; that needs a multi-op chain.

## Questions beginners will ask (and the answers this data gives)

- "Isn't torch.compile supposed to make things faster? Why is it 3-4x slower here?"
  Compile wins by fusing many small kernels into fewer bigger ones. A single matmul
  is already one cuBLAS kernel; there is nothing to fuse. What is left is the
  per-call cost of the compiled wrapper (guard checks, ~25 us), which is visible
  only when the kernel itself is shorter than that.
- "Where is the crossover?" Around N=1024 for bf16 and N=512 for fp32 on this slice.
  Below it the op is launch-bound (time is flat ~10 us no matter the size); above it
  the op is compute-bound and both paths run the same kernel, so the ratio is 1.00.
- "The timer uses CUDA events. How can CPU overhead show up in GPU time?" Events are
  recorded on the stream. If the CPU takes 35 us to issue a 5 us kernel, the GPU
  idles 30 us between kernels and the event gap includes that idle. Event timing
  measures the pipeline, not just the kernel. That is the right thing to measure
  for "how long does calling this take", but call it what it is.
- "What does reduce-overhead do and why didn't it help?" It records the kernels
  into a CUDA graph and replays the graph, removing per-kernel launch cost. Replay
  of a one-kernel graph is 3.5 us, but the compiled wrapper around it costs ~50 us
  and copies inputs in. Net loss for one op. It pays off for long sequences.
- "What about max-autotune?" Not on this hardware: a MIG 3g slice has 60 SMs and
  inductor refuses max_autotune_gemm with that few. Needs the full GPU.
- "Why is fp32 20x slower than bf16 at 8192?" fp32 here is true IEEE fp32 (TF32 is
  off, float32_matmul_precision=highest), which Hopper runs on CUDA cores. bf16 runs
  on tensor cores. Turn TF32 on and fp32 matmul will get much closer to bf16, at
  the cost of ~10-bit mantissa.
- "Why do three runs agree to 3 decimal places? I thought GPUs were noisy." Each
  number is already a median of 7 blocks of 50 launches, eager and compiled are
  interleaved so drift hits both, and nobody else was on this MIG slice. Noise
  shows up when you share an SM partition, change clocks, or time single launches.
- "Why is compile time 0.0 s after the first config?" Inductor caches compiled
  kernels on disk (~/.cache or /tmp/torchinductor_<user>). Only the very first
  config in the first run paid ~0.7 s. Delete the cache if you want cold numbers.
- "What is a MIG slice?" The H200 is partitioned into 7 pieces; a 3g.71gb slice is
  3/7 of the SMs and 71 GB. All numbers here are for that slice, not the card.
  Peak-TFLOPS comparisons against datasheets will be off by that factor.
- "Why can't I see the GPU after SSH-ing in?" Policy on this node: login shells
  have GPU devices closed. Reserve via `gpu-shell dev`, then
  `source /run/gpu-slice/$USER`. New logins pick it up automatically.
- "Why does torch.compile need triton?" torch.compile = dynamo (captures the graph)
  + inductor (generates kernels). Inductor emits Triton code for GPU kernels; no
  Triton, no inductor backend. Pin the version to what the torch build expects.
- "What are dynamo.reset and mark_static for in the harness?" reset throws away
  compiled graphs from the previous size so each config compiles fresh (and compile
  time is measured honestly). mark_static + dynamic=False forces one specialised
  graph per shape instead of a dynamic-shape graph with extra guards.
- "Why warmup, and why median instead of mean?" First calls pay cuBLAS init,
  autotune and compile. Median ignores the occasional block hit by a clock change
  or another tenant; mean would be dragged by it.
- "What does the numerics check do?" Runs eager and compiled on the same inputs
  and allclose's them. Compiled code may reorder reductions or pick a different
  algorithm, so bf16 needs a loose tolerance. No mismatches so far.
- "So when should I expect compile to win?" On ops that are many small kernels
  today: elementwise chains, scale+mask+softmax, layernorm, attention glue,
  anything memory-bound with intermediates. Softmax on its own showed the other
  case: one generated kernel vs one hand-written kernel is a coin flip (1.3x win
  for fp32, 0.8x loss for wide bf16 rows). The win comes from deleting kernels,
  not from generating a better single one.
- "Compile beat eager on softmax at 4096 columns for bf16. Isn't that good news?"
  Only partly. Eager bf16 was unusually slow at that exact shape (same time as
  fp32 for half the bytes). Always check the eager bandwidth before celebrating a
  speedup; a ratio above 1 can mean the baseline is bad, not the new thing good.
- "How do I tell if a kernel is memory-bound?" Compare bytes moved per second to
  what the device can do, and check that halving the bytes (fp32 -> bf16) roughly
  halves the time. Eager softmax passed that test; compiled bf16 softmax did not,
  which is how we know it is limited by something other than bandwidth.

## Open questions for the team

- Answered: reduce-overhead does NOT close the small-size gap, it widens it, and
  it adds an input copy at large sizes. Is there a supported way to tell the
  compiled cudagraph path that inputs are stable so it skips the copy?
  mark_static_address did not do it on 2.11.0a0+git1057af0.
- Should fp32 be benchmarked with TF32 on, since that is what most training code
  uses? Or report both.
- Can we get time on the full perf GPU for the compute-bound end? The MIG slice
  caps large-GEMM numbers and disables max-autotune.
- Why is inductor's bf16 softmax 20% slower than ATen's on 16k/65k-wide rows when
  its fp32 kernel is 30% faster? Needs an ncu profile of the two kernels: is it the
  fp32 upcast + exp throughput, or a second pass over the row?
- Why does ATen's bf16 softmax drop to 571 GB/s at exactly 4096 cols (half its
  speed at 1024 and 16384)? Looks like a kernel-selection boundary in
  SoftMax.cu. Worth a bisect over cols 2048..8192.
- --flush-l2 and --shuffle have not been exercised yet. Do small-size results
  change once L2 is cold?
- Only matmul is registered. Which ops next: elementwise chains (where fusion
  should actually win), softmax, layernorm?

## The walkthrough, in plain words

### What we set out to do

We wanted to answer a simple question: when does `torch.compile` actually help a
PyTorch operator, and when doesn't it? So we wrote a small harness that times one
operator, matmul, at a range of sizes, in eager mode and in compiled mode, and plots
the ratio. If compiled is faster the ratio is above one. That's the whole idea.

### Getting it to run was half the work

The first surprise had nothing to do with PyTorch. When we logged into the dev box,
the GPUs simply weren't there. `nvidia-smi` threw an error, torch said CUDA wasn't
available, and even trying to open the device file by hand got "operation not
permitted" despite the file being world-writable. It turned out the admins
deliberately lock GPUs away from plain SSH shells. You have to ask Slurm for one with
`gpu-shell dev`, and it hands you a slice of an H200 for twelve hours. Once we knew
that, it was a one-line fix, but it cost real time to figure out.

Then the Python environment. The system Python has no torch at all. The only usable
one is a from-source build sitting in a conda env, and that env was missing two
things: matplotlib for the plots, and triton, which `torch.compile` cannot live
without. The first run died instantly on the triton error. We installed the exact
triton version that PyTorch checkout pins, and after that compile worked.

And one more: after three successful benchmark runs, the plotting step crashed
because the output folder didn't exist and the code didn't create parent
directories. One line to fix, but it meant the first full run produced numbers and
no pictures.

### What the numbers said

Once it ran, the picture was clean and a bit humbling. For small matrices, compiled
matmul is three to four times *slower* than eager. For big matrices, they're
identical. There is no size at which compile wins.

Here's why, and this is the part a newcomer should really get. A matmul is already
one kernel. cuBLAS runs it. There's nothing for the compiler to fuse or optimise. So
what compile adds is pure overhead: every time you call the compiled function, it
checks a set of guards to make sure the inputs still match what it compiled for.
That costs about 25 microseconds. When the kernel itself takes 10 microseconds, that
overhead is the whole story. When the kernel takes 50 milliseconds, the overhead
disappears into the noise and the two lines sit on top of each other.

So compiled isn't "slower". It's doing a fixed amount of extra bookkeeping, and
whether you notice depends on how long the actual work takes. The crossover on this
hardware was around a thousand-wide matrix for bf16 and five hundred for fp32.

### The reduce-overhead experiment

The natural follow-up question was: PyTorch has a mode called reduce-overhead that
uses CUDA graphs specifically to cut per-call cost. Surely that fixes the small-size
problem?

It made things worse. Everywhere. Small sizes got slower still, and even the biggest
matmul lost about ten percent.

That didn't make sense, so we pulled it apart on the node. Three findings:

First, CUDA graphs genuinely work. If you capture the matmul into a graph by hand
and replay it, it runs in 3.5 microseconds, faster than eager. The technology isn't
the problem.

Second, `torch.compile`'s version of it copies your inputs into its own private
memory every single call. For two big matrices that's a quarter of a gigabyte of
copying per call, and that accounts for almost all of the slowdown at large sizes.
We tried the documented way of telling it the inputs are stable
(`mark_static_address`), and on this build it didn't skip the copy.

Third, the rest of the overhead is Python. Before each graph replay there's guard
checking plus the graph-management bookkeeping, and together they cost around 50
microseconds. For a one-kernel function a graph has no launch gaps to hide, so you
pay all of that and get nothing back.

The lesson: reduce-overhead is built for long chains of small kernels where the CPU
can't issue work fast enough to keep the GPU busy. One matmul is the exact opposite
of that workload. Right tool, wrong job.

### Things to keep in mind when reading the plots

We ran on a MIG slice, not a full H200. That's three sevenths of the chip. Absolute
numbers will be lower than datasheets, and the max-autotune compile mode refuses to
run on that few SMs, so we couldn't test it.

fp32 was true fp32, with TF32 off. That's why it's twenty times slower than bf16 at
the top end. Turn TF32 on and that gap shrinks a lot. Worth deciding which fp32 we
actually care about.

The runs were remarkably stable, three repeats agreeing to three decimal places.
That's not luck. Each number is a median of seven blocks of fifty launches, eager
and compiled are measured back to back so any drift hits both, and nobody else was
on our slice.

### Softmax: a more interesting answer

Softmax was the second operator, and the first where the plot climbed above one.
For fp32 on rows of four thousand or more, compiled softmax was 10 to 30 percent
faster than eager. Inductor's generated reduction kernel simply moved bytes faster
than the hand-written one in ATen, peaking at around 1.8 terabytes a second on our
slice, the best number we've seen.

But bf16 told a different story. On the widest rows, compiled was 20 percent slower
than eager. And the one bf16 size where compile "won" turned out to be a shape where
eager bf16 was oddly slow, taking the same time as fp32 for half the data. The
speedup was real, but it was eager's weakness, not compile's strength.

The honest summary: when it's one generated kernel against one hand-written kernel,
compile is a coin flip. It wins where ATen has a weak spot and loses where ATen is
well tuned. The thing compile is actually good at, deleting kernels by fusing a
chain of small ops into one, we still haven't tested.

### Where this leaves us

We now have a harness that works end to end on the dev box, two operators measured,
and a clear picture: on single-kernel ops compile adds a fixed ~35 microsecond cost
per call and, above that, is roughly a wash. The next experiment is the one that
should actually show fusion paying off: a chain like scale, mask, softmax, or a
layernorm with the elementwise work around it, where eager launches several
kernels and compile launches one.
