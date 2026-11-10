# MQAR training with shrink

This trains the same two-layer IsoKDN retrieval model as `mqar_configs.py`
(BaseConv, then IsoKDN; H=8, head_dim=64, d_model=512) **from scratch**. It is not text pretraining
or an inference-only edit of the existing checkpoint. Start with two matched
arms: gamma=1 (no shrink) and gamma=0.9 (retain 90% along the estimated axis).
Training may teach the model to tolerate the edit; improvement is an experimental
question. The principal direction can carry useful signal, even after training.

## HPC commands

Use the existing CUDA/BF16, FLA, Zoology environment from `train_mqar.py`.
From the repository root, first check the plan and CUDA forward/backward parity:

```bash
python train_mqar_shrink.py --plan-only
python -m unittest discover -s Expr -p 'test_train_shrink.py' -v
```

The CUDA test compares the gamma=1 experimental path with the production fused
layer, including parameter gradients. It is skipped on a machine without CUDA.
Run it on the HPC GPU before spending a training budget; a skip is not GPU
validation. CPU tests check independent recurrence agreement, gradients across
an edit boundary, gate gradients and causality.

Small end-to-end smoke run (not evidence of convergence):

```bash
python train_mqar_shrink.py --epochs 1 --train-examples 32 --test-examples 32 \
  --output checkpoints/mqar_shrink_smoke
```

Pilot, two training runs with the same seed:

```bash
python train_mqar_shrink.py --gammas 1.0 0.9 --seeds 1234 \
  --heads 8 --head-dim 64 --epochs 10 --train-examples 2048 \
  --test-examples 128 --batch-size 8 --eval-batch-size 8 \
  --output checkpoints/mqar_shrink_pilot
```

Use `--seeds 1234 1235 1236` and a fresh output directory for a replicated run.
`--train-examples` is per slice, so the pilot has 10,240 training examples per
epoch. These are fixed examples reused across epochs. This Python recurrence is
slower than the fused kernel; the small model reduces memory, not the sequential
launch overhead. Adjust the budget after the smoke run. Do not interpret an
underfit control as evidence about whether shrink helps.

## What is held fixed

- Two TransformerBlocks with zoology's Hybrid mixer, as in `mqar_configs.py`:
  layer 0 BaseConv, layer 1 IsoKDN (H=8, head_dim=64, d_model=512 by default),
  vocabulary=8192, Identity state mixer. Only the IsoKDN layer is edited.
- Both arms use FP32 PyTorch gates, gain and memory recurrence inside the same
  BF16 projection/output path. The gamma=1 control does not use the fused memory
  recurrence, avoiding a backend difference between the training arms.
- Identical initialization seed, data configuration, learning rate, optimizer,
  batch size and epoch count. Training RNG is reset after data preparation so
  cache hits cannot change shuffle/dropout RNG. There is no early stopping in
  this experiment. No randomized axis estimation consumes the training RNG.
- Train on 64:4, 128:8, 256:16, 256:32, 256:64. Evaluate additionally at
  512:128 and 1024:256; these are length/load extrapolation tests.

The edit uses the leading eigenvectors of the **uncentered** moment of normalized
keys from the preceding window (default 256 tokens), separately per example and
head. Default rank=1, chunk=16, offset=0, no spread and no tail protection.
Only decayed writes made within the current chunk are shrunk. The first and final
chunks are unedited, as in the inference experiment. The edit occurs after the
chunk's outputs and affects subsequent tokens. Axis estimation is stop-gradient;
all recurrence, write, decay and gain paths remain differentiable. The scalar
uncertainty recurrence is unchanged by the edit. Exact eigenvectors are used;
the earlier inference sweep used approximate subspace iteration.

The fused gate producer has a private backward convention paired with its fused
memory consumer. The experimental branch therefore computes gates directly with
ordinary autograd rather than decoding that producer's prefix during training.
The production path remains the default when `shrink_gamma=None`.

## Reading the result

Each run saves the ordinary best-validation `.pt` and a fixed-budget `.last.pt`.
Final comparisons use `.last.pt`, with fresh held-out examples built from
`seed+100000`. Every trained model is evaluated with gamma=1 and every requested
shrink gamma on exactly the same held-out data. Default comparison:

| Training | Evaluation | Purpose |
| --- | --- | --- |
| No shrink | No shrink | Matched baseline |
| No shrink | Shrink | Inference-only damage |
| Shrink | Shrink | Adaptation and practical performance |
| Shrink | No shrink | Dependence on the learned intervention |

`comparison.csv` contains accuracy per length and terminal/other-pair accuracy.
Terminal groups use the same editable chunk boundaries in every cell, including
the no-shrink control. Per-run `.eval.json` retains per-example, per-pair counts
for paired uncertainty estimates; `.eval.csv` contains the corresponding summary.
Compare shrink-trained/shrink-evaluated against no-shrink/no-shrink at each
length across seeds. Recovering the inference-only loss demonstrates adaptation,
but beating the matched baseline is necessary to claim an accuracy benefit.

Checkpoint model configs record the training edit, and normal model loading
restores it. The older `Expr/shrink_spread.py` frozen-checkpoint sweep hooks the
fused memory kernel, which this experimental branch bypasses; use the new final
evaluation instead of that sweep for these checkpoints.
