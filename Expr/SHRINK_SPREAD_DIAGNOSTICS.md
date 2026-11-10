# Frozen-checkpoint inference diagnostics

These experiments load an existing MQAR checkpoint in evaluation mode, disable
parameter gradients, and evaluate under `torch.no_grad()`. There is no training,
optimizer, or checkpoint modification. Only the temporary recurrent-state update
is edited; state is reset for each forward pass.

## 1. Diagnose the recurring boundary loss first

Run from the repository root in the existing HPC environment:

```bash
python Expr/shrink_spread.py /path/to/checkpoint.pt \
  --preset boundary --heads 8 --head-dim 64 --lr 1e-2 \
  --out Expr/results/boundary_seed1234
```

Use `--plan-only` to inspect checkpoint selection and settings without CUDA.
Passing a directory selects every matching `.pt` checkpoint recursively; passing
one checkpoint keeps the run small. Selection uses the saved training config.

Defaults for this preset: **32 examples**, batch size 32, evaluation seed 1234,
slices `512:128,1024:256`, window 256, rank 1, gamma 0.1, eta 0. Eight settings
are evaluated per slice: kernel, replay, and key/random shrink at each schedule:

| Chunk / offset | First possible affected read (zero-based token) | Terminal context pair positions (one-based) |
|---|---:|---|
| 16 / 0 | 32 | 16, 24, 32, ... |
| 16 / 2 | 34 | 17, 25, 33, ... |
| 32 / 0 | 64 | 32, 48, 64, ... |

Offset 2 extends the initial unedited warm-up chunk to 18 tokens. Subsequent
chunks remain 16 tokens long. Direction estimates still use only tokens before
the current chunk. The final sequence chunk is never edited. A single unedited
replay suffices because chunk partitioning does not change its token recurrence.
The larger chunk also changes edit frequency and history at the first edit;
the shifted 16-token schedule is the cleaner test of boundary phase.

If the recall troughs move to the new terminal positions, that supports a
boundary-dependent failure. Compare each setting against replay at those same
positions, not against replay's default terminal group.

## 2. Test gentler edits after inspecting the boundary run

```bash
python Expr/shrink_spread.py /path/to/checkpoint.pt \
  --preset gentle --heads 8 --head-dim 64 --lr 1e-2 \
  --out Expr/results/gentle_seed1234
```

This uses the same sample/slice defaults and ten settings per slice: kernel,
replay, six key-based settings with gamma `{0.7,0.9}` crossed with eta
`{0,0.05,0.1}`, and two spread-only controls at eta `{0.05,0.1}`. Chunk=16,
offset=0, window=256, rank=1. Gamma is the **retained** common-direction amplitude;
larger gamma means weaker shrinkage. Spread still acts on the entire shrunk update.

Focused presets fix edit parameters and reject conflicting edit flags. Sampling
flags such as `--examples`, `--batch-size`, `--seed`, and `--slices` remain usable.
To configure edits manually, use `--preset sweep`; `--chunk-offset` shifts the
schedule and `--boundary-probes` enables the same probes and per-example counts.
The original sweep still includes its eta=0.5 spread-only control.

## Outputs and interpretation

Each checkpoint produces:

- `<checkpoint>.json`: config, replay validation, geometry, per-pair recall,
  per-example correct/total counts, position summaries, and detailed probes.
- `<checkpoint>-pair-recall.csv`: the original pair-position recall format.
- `<checkpoint>-summary.csv`: overall, terminal-pair, and other-pair accuracy;
  replay on the matching positions; paired sequence delta and standard error.
- `<checkpoint>-boundary-probes.csv`: per-layer, per-boundary, per-pair internal
  reconstruction measurements, pooled over examples and heads. This file is
  emitted when eligible context probes exist.

At each eligible context boundary, probes re-read the state before and after the
edit using the **value token's memory key**, comparing against that token's
internal value vector. They cover all value tokens in the current chunk, so
`terminal_pair` and `tokens_before_boundary` distinguish its last pair from
earlier pairs. Chunks extending beyond the KV context are excluded from probes.
Layer numbers are zero-based KDN call order within a forward pass.

`squared_error_change > 0` means the edit worsened immediate internal
reconstruction. Relative errors divide pooled squared error by pooled target
energy; zero-energy targets produce null. Projection shares measure how much
of the value token's key or write-gain vector lies in the selected subspace;
spread-only has no selected subspace and reports null. These are internal write
equation probes, **not vocabulary accuracy or future-query retrieval**. Probe
values never feed into the state or direction estimator.

Compare terminal versus earlier pairs within each layer and schedule. Strong
terminal reconstruction damage together with high projection share would help
explain the recall troughs. Final vocabulary recall remains the outcome measure.

Paired standard errors use sequences as the sampling unit, not individual
queries or heads. Small gains need replication. To repeat with a new seed,
change both `--seed` and `--out` (for example, seed 1235 and a `seed1235` output
directory); reusing a destination overwrites that checkpoint's earlier reports.
