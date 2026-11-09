"""Window-estimated shrink and spread of each chunk's state update, at inference.

The KDN kernel is replaced by a chunked PyTorch recurrence. Inside a chunk the
state evolves exactly as in the model, token by token. At the end of chunk c
its update

    dS = S_end - D S_start        (D = the chunk's per-channel decay product)

is edited before it is carried to the next chunk:

  1. Window: the last w keys before the chunk (past tokens only).
       iso_kdn   top-r eigenvectors U_r of the uncentered second moment K^T K / n,
                 which includes the mean outer-product contribution
       diag_kdn  top-r eigenvectors of the centered covariance, with that
                 contribution removed (not necessarily orthogonal to the mean)
     U_r comes from randomized subspace iteration; unsupported directions are
     discarded. Past-window directions estimate commonality in the next chunk.
  2. Shrink: dS' = (I - (1 - gamma) U_r U_r^T) dS. The state here is [key, value]
     (o = S^T q), so the key-space projector acts on the left. This is the
     plan's dS (I - ...) for a state stored as [value, key].
  3. Spread: O = NS5(dS'), five Newton-Schulz steps, rescaled to ||dS'||_F;
     dS'' = (1 - eta) dS' + eta O.
  4. Write: S_start(next) = D S_start + dS''.

Spread acts on the entire shrunk update, including the retained common part.
It redistributes singular-value strength, not identifiable unique associations;
subsequent error-driven writes can correct the modified state.
Keys, queries, gains and the read equation are untouched. The first chunk has
no past keys and is never edited, nor is the last (nothing reads it).
Intervention chunks default to 16 tokens, independently of the kernel chunks.
The first edit can affect reads at token index 32. Warm-up uses available history.

gamma = 1, eta = 0 is the unedited recurrence. It is always run as "replay",
next to the kernel's own accuracy, so the two can be compared.

Reported per setting and MQAR slice: recall accuracy, and averaged over every
edited (chunk, example, head):
  dS_erank / dS2_erank   effective rank of dS before and after the edit
  captured               share of the window moment's trace in span(U_r)
  power_quality          captured / the exact top-r share (1 = converged)
  rel_change             ||dS'' - dS||_F / ||dS||_F
  supported_rank         number of supported shrink directions (can be below r)
  common_before / common_shrunk / common_after
                         update energy fraction in the estimated common subspace
Zero-energy inputs have energy fractions 0; absent moment signal has quality 0.
first_effect_token and query_exposure_fraction describe when an eligible boundary
edit could affect reads, not whether that setting actually changed predictions.

Usage:
  python Expr/shrink_spread.py /path/to/*.pt --out Expr/results/shrink_spread
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import effective_rank  # noqa: E402
from state_geometry import (  # noqa: E402
    DEFAULT_SLICES,
    build_model,
    capture_kdn_calls,
    choose_gate,
    decode_alpha,
    json_safe,
    l2norm,
    replay_error,
    REPLAY_TOLERANCE,
)

NS5_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # Muon's quintic Newton-Schulz
DIAGNOSTICS = ("dS_erank", "dS2_erank", "captured", "power_quality", "rel_change",
               "supported_rank", "common_before", "common_shrunk", "common_after")


def newton_schulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization of [..., m, n], rescaled to the input's Frobenius norm."""
    a, b, c = NS5_COEFFICIENTS
    # Normalize before squaring to avoid scale-dependent underflow/overflow.
    magnitude = matrix.abs().amax(dim=(-2, -1), keepdim=True)
    scaled = matrix / torch.where(magnitude > 0, magnitude, 1.0)
    norm = scaled.norm(dim=(-2, -1), keepdim=True)
    x = scaled / torch.where(norm > 0, norm, 1.0)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    output_norm = x.norm(dim=(-2, -1), keepdim=True)
    return (x / torch.where(output_norm > 0, output_norm, 1.0)) * norm * magnitude


def top_directions(moment: torch.Tensor, init: torch.Tensor, steps: int,
                   rank: int, eigenvalue_floor: torch.Tensor) -> torch.Tensor:
    """Return leading Ritz directions, with zero columns for unsupported directions."""
    magnitude = moment.abs().amax(dim=(-2, -1), keepdim=True)
    normalized = moment / torch.where(magnitude > 0, magnitude, 1.0)
    basis, _ = torch.linalg.qr(init)
    for _ in range(steps):
        basis, _ = torch.linalg.qr(normalized @ basis)
    projected = basis.mT @ moment @ basis
    values, vectors = torch.linalg.eigh((projected + projected.mT) * 0.5)
    basis = basis @ vectors[..., -rank:]
    return basis * (values[..., -rank:] > eigenvalue_floor[..., None])[..., None, :]


def energy_share(basis, matrix):
    magnitude = matrix.abs().amax(dim=(-2, -1), keepdim=True)
    scaled = matrix / torch.where(magnitude > 0, magnitude, 1.0)
    energy = scaled.square().sum((-2, -1))
    return (basis.mT @ scaled).square().sum((-2, -1)) / torch.where(energy > 0, energy, 1.0)


class ChunkEdit:
    """Shrink and spread one chunk's update, using the keys of the last `window` tokens."""

    def __init__(self, kind: str, window: int, rank: int, gamma: float, eta: float,
                 power_steps: int, seed: int, directions: str = "keys"):
        if kind not in ("iso_kdn", "diag_kdn"):
            raise ValueError(f"Unsupported KDN kind: {kind}")
        if directions not in ("keys", "gains", "random", "none"):
            raise ValueError(f"Unknown direction source: {directions}")
        if directions == "gains" and kind != "diag_kdn":
            raise ValueError("Gain-based directions are a Diag-only ablation")
        if window < 1 or power_steps < 1 or (rank < 1 and directions != "none"):
            raise ValueError("window, rank and power_steps must be positive")
        if directions == "none" and (rank != 0 or gamma != 1):
            raise ValueError("Spread-only requires rank=0 and gamma=1")
        if not (0 <= gamma <= 1 and 0 <= eta <= 1):
            raise ValueError("gamma and eta must be in [0, 1]")
        self.kind, self.window, self.rank = kind, window, rank
        self.gamma, self.eta, self.power_steps, self.seed = gamma, eta, power_steps, seed
        self.directions = directions
        self.totals = {name: 0.0 for name in DIAGNOSTICS}
        self.count = 0
        self.first_effect_token = None

    def __call__(self, keys: torch.Tensor, start: int, update: torch.Tensor,
                 write: torch.Tensor | None = None) -> torch.Tensor:
        """keys/write [B, T, H, K]; update [B, H, K, V]. Never use vectors at or after start."""
        source = keys
        if self.directions == "gains":
            if write is None or write.shape != keys.shape:
                raise ValueError("Gain directions require write vectors with the key tensor's shape")
            source = write  # Raw g_t = kappa_t, retaining gain magnitudes.
        past = source[:, max(0, start - self.window):start].transpose(1, 2)
        if self.rank > keys.shape[-1]:
            raise ValueError("rank cannot exceed the key dimension")
        if past.shape[2] == 0:
            return update
        if self.directions == "none":
            basis = update.new_empty((*update.shape[:-1], 0))
            captured = quality = update.new_full(update.shape[:2], math.nan)
        else:
            mean = past.mean(2)
            raw_energy = past.square().sum(-1).mean(-1)
            if self.kind == "diag_kdn":
                past = past - mean[:, :, None]
            moment = past.mT @ past / past.shape[2]
            generator = torch.Generator(device=keys.device).manual_seed(
                self.seed + start if self.directions == "random" else self.seed)
            if self.directions == "random":
                # Independent across heads/chunks, shared across examples so
                # changing batch size does not change the control intervention.
                init = torch.randn(keys.shape[2], keys.shape[-1], self.rank,
                                   generator=generator, device=keys.device, dtype=moment.dtype)
                basis = torch.linalg.qr(init)[0][None].expand(keys.shape[0], -1, -1, -1)
            else:
                init = torch.randn(keys.shape[-1], min(keys.shape[-1], self.rank + 2),
                                   generator=generator, device=keys.device, dtype=moment.dtype)
                init = init.expand(*moment.shape[:2], -1, -1).clone()
                floor = torch.finfo(moment.dtype).eps * keys.shape[-1] * raw_energy
                basis = top_directions(moment, init, self.power_steps, self.rank, floor)
            trace = moment.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
            captured = (basis.mT @ moment @ basis).diagonal(dim1=-2, dim2=-1).sum(-1) / trace
            exact = torch.linalg.eigvalsh(moment)[..., -self.rank:].sum(-1) / trace
            quality = captured / exact.clamp_min(1e-12)

        shrunk = update - (1 - self.gamma) * basis @ (basis.mT @ update)
        edited = shrunk if self.eta == 0 else (1 - self.eta) * shrunk + self.eta * newton_schulz5(shrunk)

        self.totals["dS_erank"] += effective_rank(torch.linalg.svdvals(update)).sum().item()
        self.totals["dS2_erank"] += effective_rank(torch.linalg.svdvals(edited)).sum().item()
        self.totals["captured"] += captured.sum().item()
        self.totals["power_quality"] += quality.sum().item()
        self.totals["supported_rank"] += basis.square().sum((-2, -1)).sum().item()
        for name, matrix in (("common_before", update), ("common_shrunk", shrunk), ("common_after", edited)):
            self.totals[name] += (energy_share(basis, matrix).sum().item()
                                  if self.directions != "none" else math.nan)
        self.totals["rel_change"] += ((edited - update).norm(dim=(-2, -1))
                                      / update.norm(dim=(-2, -1)).clamp_min(1e-12)).sum().item()
        self.count += update.shape[0] * update.shape[1]
        return edited

    def diagnostics(self) -> dict:
        return {name: (value / self.count if self.count else math.nan) for name, value in self.totals.items()}


def chunked_recurrence(q, k, v, alpha, write, scale, chunk: int, edit=None) -> torch.Tensor:
    """KDN memory token by token; at each chunk boundary the chunk's update may be edited."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    # The model adapter enables BF16 autocast, including inside replacement
    # kernels. Keep replay, eigensolvers and NS arithmetic at their input precision.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _chunked_recurrence(q, k, v, alpha, write, scale, chunk, edit)


def _chunked_recurrence(q, k, v, alpha, write, scale, chunk, edit):
    batch, length, heads, key_dim = q.shape
    state = q.new_zeros(batch, heads, key_dim, v.shape[-1])
    outputs = torch.empty_like(v)
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        update = torch.zeros_like(state) if edit is not None else None
        for t in range(start, end):
            state = alpha[:, t, ..., None] * state
            delta = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
            increment = write[:, t, ..., None] * delta[..., None, :]
            state = state + increment
            if edit is not None:
                # Accumulate decayed writes directly: subtracting two large
                # states to recover a small dS can lose the update entirely.
                update = alpha[:, t, ..., None] * update + increment
            outputs[:, t] = scale * torch.einsum("bhk,bhkv->bhv", q[:, t], state)
        if edit is not None and end < length:
            count = edit.count
            edited = edit(k, start, update, write=write)
            state = state + (edited - update)
            if edit.count > count and edit.first_effect_token is None:
                edit.first_effect_token = end
    if not torch.isfinite(outputs).all():
        raise FloatingPointError("Nonfinite outputs in shrink/spread recurrence")
    return outputs


@contextmanager
def replace_kdn(run):
    """Route every KDN kernel call to run(call) -> fp32 outputs [B, T, H, V]."""
    import LinearAttn.iso_kdn as iso
    import LinearAttn.kdn_ops.diag_kdn_chunk as diag

    iso_kernel, diag_kernel = iso._chunk_kda_precumsum_paired, diag.chunk_kalman

    def iso_run(q, k, v, g_cumsum, beta, scale=None):
        q_unit, k_unit = (l2norm(x).to(x.dtype).float() for x in (q, k))
        return run(dict(kind="iso_kdn", q=q_unit, k=k_unit, v=v.float(), g_cumsum=g_cumsum.float(),
                        write=beta.float()[..., None] * k_unit,
                        scale=q.shape[-1] ** -0.5 if scale is None else float(scale))).to(v.dtype)

    def diag_run(q_memory, k_memory, kappa, v, g_cumsum, scale):
        return run(dict(kind="diag_kdn", q=q_memory.float(), k=k_memory.float(), v=v.float(),
                        g_cumsum=g_cumsum.float(), write=kappa.float(), scale=float(scale)))

    iso._chunk_kda_precumsum_paired, diag.chunk_kalman = iso_run, diag_run
    try:
        yield
    finally:
        iso._chunk_kda_precumsum_paired, diag.chunk_kalman = iso_kernel, diag_kernel


def settings_grid(args, kind: str) -> list[dict]:
    grid = [dict(name="replay", directions="none", window=0, rank=0, gamma=1.0, eta=0.0)]
    sources = ("keys", "gains", "random") if kind == "diag_kdn" else ("keys", "random")
    for source, window, rank, gamma, eta in itertools.product(
            sources, args.windows, args.ranks, args.gammas, args.etas):
        grid.append(dict(name=f"{source}-w{window}-r{rank}-g{gamma:g}-e{eta:g}", directions=source,
                         window=window, rank=rank, gamma=gamma, eta=eta))
    # Exactly one spread-only control; it has no direction/window dependence.
    grid.append(dict(name="spread-only", directions="none", window=args.windows[0], rank=0,
                     gamma=1.0, eta=0.5))
    return grid


def query_pair_positions(inputs, labels, pairs):
    """Map supervised queries to their original context pair index (zero-based)."""
    context_keys = inputs[:, :2 * pairs:2]
    matches = inputs[..., None] == context_keys[:, None, :]
    answer = labels != -100
    if not (matches.sum(-1)[answer] == 1).all():
        raise ValueError("Each supervised query must match exactly one context key")
    return torch.where(answer, matches.to(torch.int64).argmax(-1), -1)


class Recall:
    def __init__(self, pairs):
        self.correct = torch.zeros(pairs, dtype=torch.int64)
        self.total = torch.zeros_like(self.correct)

    def update(self, predictions, labels, pair_positions):
        answer = labels != -100
        indices = pair_positions[answer]
        hits = predictions[answer] == labels[answer]
        self.total += torch.bincount(indices, minlength=len(self.total)).cpu()
        self.correct += torch.bincount(indices[hits], minlength=len(self.total)).cpu()

    def report(self):
        total, correct = self.total.sum().item(), self.correct.sum().item()
        return {"accuracy": correct / total if total else None, "correct": correct, "total": total,
                "recall_by_pair_position": [
                    {"pair_position": i + 1, "key_token_index": 2 * i,
                     "correct": c, "total": n, "accuracy": c / n if n else None}
                    for i, (c, n) in enumerate(zip(self.correct.tolist(), self.total.tolist()))]}


def evaluate_batch(model, inputs, labels, pair_positions, recall):
    with torch.no_grad():
        predictions = model(inputs).argmax(-1)
    recall.update(predictions, labels, pair_positions)


def validate_replay(call, gate, chunk):
    alpha = decode_alpha(call["g_cumsum"], gate[0], gate[1])
    output = chunked_recurrence(call["q"], call["k"], call["v"], alpha, call["write"], call["scale"], chunk)
    error = replay_error(output, call["out"])
    if not error <= REPLAY_TOLERANCE:
        raise RuntimeError(f"{call['kind']} replay differs from kernel by {error:.3g}")
    return error


def checkpoint_spec(checkpoint):
    """Read architecture and training LR from saved config, never infer from filenames."""
    mixers = []
    names = {"zoology_mixers.IsoKDNMixer": "iso_kdn", "zoology_mixers.DiagKDNMixer": "diag_kdn"}
    def visit(node):
        if isinstance(node, dict):
            if node.get("name") in names:
                kwargs = node["kwargs"]
                mixers.append((names[node["name"]], int(kwargs["num_heads"]), int(kwargs["head_dim"])))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(checkpoint["config"]["model"])
    if not mixers:
        return None
    if len(set(mixers)) != 1:
        raise ValueError("Sweep selection requires homogeneous KDN kind/head dimensions")
    kind, heads, head_dim = mixers[0]
    return dict(kind=kind, heads=heads, head_dim=head_dim,
                learning_rate=float(checkpoint["config"]["learning_rate"]))


def matches_target(spec, args):
    return (spec is not None and spec["heads"] == args.heads and spec["head_dim"] == args.head_dim
            and math.isclose(spec["learning_rate"], args.lr, rel_tol=1e-8, abs_tol=0.0))


def write_pair_recall_csv(result, destination):
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seq_len", "num_kv_pairs", "setting", "pair_position",
                                                   "key_token_index", "correct", "total", "accuracy"])
        writer.writeheader()
        for row in result["slices"]:
            for setting, entry in row["settings"].items():
                for pair in entry["recall_by_pair_position"]:
                    writer.writerow(dict(seq_len=row["seq_len"], num_kv_pairs=row["num_kv_pairs"],
                                         setting=setting, **pair))


def run_checkpoint(path: Path, slices, settings, args, checkpoint=None) -> dict:
    from zoology.data.multiquery_ar import MQARConfig

    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    vocab_size = checkpoint["config"]["model"]["vocab_size"]
    result = {"checkpoint": str(path), "run_id": checkpoint["config"].get("run_id"),
              "best_epoch": checkpoint["epoch"], "chunk": args.chunk, "power_steps": args.power_steps,
              "seed": args.seed, "direction_window": "past_only", "checkpoint_spec": checkpoint_spec(checkpoint),
              "pair_position_definition": "1-based order in the original key-value context",
              "gain_moment": "centered covariance of raw write gain vectors",
              "random_directions": "orthonormal; independent by head/chunk; seed fixed across settings",
              "spread_target": "whole_shrunk_update", "gate_decodings": {},
              "slices": []}
    gates = {}
    for seq_len, pairs in slices:
        # MQAR also uses torch RNG for filler tokens; its NumPy seed alone is
        # insufficient to pair evaluation examples across different checkpoints.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.seed)
            data = MQARConfig(vocab_size=vocab_size, input_seq_len=seq_len, num_examples=args.examples,
                              num_kv_pairs=pairs).build(seed=args.seed)
        batches = []
        for i in range(0, args.examples, args.batch_size):
            inputs, labels = data.inputs[i:i + args.batch_size].cuda(), data.labels[i:i + args.batch_size].cuda()
            batches.append((inputs, labels, query_pair_positions(inputs, labels, pairs)))
        row = {"seq_len": seq_len, "num_kv_pairs": pairs, "examples": args.examples, "settings": {}}

        recall = Recall(pairs)
        worst_error = 0.0
        for inputs, labels, pair_positions in batches:
            calls = []
            with capture_kdn_calls(calls):
                evaluate_batch(model, inputs, labels, pair_positions, recall)
            if not calls:
                raise RuntimeError(f"{path.name} contains no supported KDN calls")
            for call in calls:
                kind = call["kind"]
                if kind not in gates:
                    gates[kind] = choose_gate(call)
                    gate = gates[kind]
                    result["gate_decodings"][kind] = {"base": gate[0], "chunk": gate[1],
                                                       "replay_error": gate[2]}
                worst_error = max(worst_error, validate_replay(call, gates[kind], args.chunk))
        row["settings"]["kernel"] = recall.report()
        row["replay_error"] = worst_error

        for setting in settings:
            edits = []

            def run(call, setting=setting, edits=edits):
                gate = gates[call["kind"]]
                alpha = decode_alpha(call["g_cumsum"], gate[0], gate[1])
                edit = None
                if setting["name"] != "replay":
                    edit = ChunkEdit(call["kind"], setting["window"], setting["rank"], setting["gamma"],
                                     setting["eta"], args.power_steps, args.seed, setting["directions"])
                    edits.append(edit)
                return chunked_recurrence(call["q"], call["k"], call["v"], alpha, call["write"],
                                          call["scale"], args.chunk, edit)

            recall = Recall(pairs)
            exposed = exposure_total = 0
            with replace_kdn(run):
                for inputs, labels, pair_positions in batches:
                    previous_edits = len(edits)
                    evaluate_batch(model, inputs, labels, pair_positions, recall)
                    total = (labels != -100).sum().item()
                    positions = torch.arange(labels.shape[1], device=labels.device)[None]
                    for edit in edits[previous_edits:]:
                        exposure_total += total
                        if edit.first_effect_token is not None:
                            exposed += ((labels != -100) & (positions >= edit.first_effect_token)).sum().item()
            entry = {**recall.report(), **{k: v for k, v in setting.items() if k != "name"}}
            count = sum(edit.count for edit in edits)
            entry["evaluated_chunk_example_heads"] = count
            first_tokens = [edit.first_effect_token for edit in edits if edit.first_effect_token is not None]
            entry["first_effect_token"] = min(first_tokens) if first_tokens else None
            entry["query_exposure_fraction"] = exposed / exposure_total if exposure_total else 0.0
            for name in DIAGNOSTICS:
                entry[name] = sum(edit.totals[name] for edit in edits) / count if count else math.nan
            row["settings"][setting["name"]] = entry
        result["slices"].append(row)
        print_slice(row)
    return result


def print_slice(row: dict) -> None:
    print(f"\n  {row['seq_len']}:{row['num_kv_pairs']}")
    print(f"  {'setting':<22}{'acc':>7}" + "".join(f"{name:>14}" for name in DIAGNOSTICS))
    for name, entry in row["settings"].items():
        cells = "".join(f"{entry.get(d, math.nan):>14.3f}" for d in DIAGNOSTICS)
        print(f"  {name:<22}{entry['accuracy']:>7.3f}{cells}")


def main() -> None:
    def floats(text):
        return [float(item) for item in text.split(",")]

    def ints(text):
        return [int(item) for item in text.split(",")]

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--heads", type=int, default=8, help="filter checkpoint head count (default: 8)")
    parser.add_argument("--head-dim", type=int, default=64, help="filter checkpoint key head size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-2, help="filter checkpoint training LR (default: 1e-2)")
    parser.add_argument("--plan-only", action="store_true", help="inspect matching checkpoints/settings without CUDA")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "shrink_spread")
    parser.add_argument("--slices", default=DEFAULT_SLICES, help="seq_len:num_kv_pairs, comma-separated")
    parser.add_argument("--windows", type=ints, default=[256])
    parser.add_argument("--ranks", type=ints, default=[1, 2])
    parser.add_argument("--gammas", type=floats, default=[0.1, 0.3])
    parser.add_argument("--etas", type=floats, default=[0.0, 0.5])
    parser.add_argument("--chunk", type=int, default=16,
                        help="intervention chunk length, independent of kernel chunks (default: 16)")
    parser.add_argument("--power-steps", type=int, default=2)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if min(args.chunk, args.power_steps, args.examples, args.batch_size) < 1:
        parser.error("chunk, power-steps, examples and batch-size must be positive")
    if not args.windows or not args.ranks or any(x < 1 for x in args.windows + args.ranks):
        parser.error("windows and ranks must be nonempty positive integer lists")
    if not args.gammas or not args.etas or any(not 0 <= x <= 1 for x in args.gammas + args.etas):
        parser.error("gammas and etas must be nonempty lists of values in [0, 1]")

    slices = [tuple(int(part) for part in item.split(":")) for item in args.slices.split(",")]
    paths = sorted({p for path in args.checkpoints for p in (path.rglob("*.pt") if path.is_dir() else [path])})
    if not paths:
        parser.error("No checkpoint files found")
    selected = 0
    for path in paths:
        if not path.is_file():
            parser.error(f"Checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        spec = checkpoint_spec(checkpoint)
        if not matches_target(spec, args):
            print(f"Skipping {path}: {spec}")
            continue
        selected += 1
        settings = settings_grid(args, spec["kind"])
        print(f"Selected {path}: {spec}; {len(settings) + 1} settings including kernel/replay")
        if args.plan_only:
            continue
        if not torch.cuda.is_available():
            parser.error("CUDA is required to evaluate KDN checkpoints; use --plan-only to inspect the sweep")
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"\n{path.stem}")
        result = run_checkpoint(path, slices, settings, args, checkpoint=checkpoint)
        destination = args.out / f"{path.stem}.json"
        destination.write_text(json.dumps(json_safe(result), allow_nan=False, indent=1))
        pair_csv = args.out / f"{path.stem}-pair-recall.csv"
        write_pair_recall_csv(result, pair_csv)
        print(f"Saved {destination}")
        print(f"Saved {pair_csv}")
    if not selected:
        parser.error(f"No Iso/Diag checkpoints match H={args.heads}, d_k={args.head_dim}, LR={args.lr:g}")


if __name__ == "__main__":
    main()
