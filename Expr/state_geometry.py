"""Recurrent-state geometry of trained KDN models on MQAR.

For each checkpoint saved by train_mqar.py and each MQAR test size, this runs the
model, captures what the KDN kernel receives, and replays the recurrence token by
token to rebuild the state the kernel never returns:

    S_t = Diag(alpha_t) S_{t-1}
    delta_t = v_t - S_t^T k_t          (what the state got wrong about v_t)
    S_t = S_t + g_t delta_t^T          (the write)
    o_t = scale * S_t^T q_t            (the read)

g_t is beta_t * k_t for Isotropic KDN and kappa_t for Diagonal KDN. The replayed
outputs are checked against the kernel's for every example and head, and the
script stops if any disagree or are not finite.

Matrices, per example and head, all head_dim x head_dim:
  state          S at the end of the key-value context and at the last token
  write          sum of g_t delta_t^T over the context: everything written, without decay
  read_activity  Q^T O over the query positions: reads weighted by how often each
                 query direction was used
  read_operator  Q^+ O, the least-squares map from queries to outputs. The state
                 changes between queries, so this is one map fitted to all reads;
                 read/fit_residual says how well it fits, and read/query_rank how
                 many query directions it is fitted on. With fewer queries than
                 head_dim it is rank deficient and its rotation is undefined.
Each gets effective, stable and numerical rank and a polar decomposition M = U P
(see metrics.polar_summary). Rotation and reflection are averaged only over the
examples where the matrix has full rank; */full_rank is that fraction. Examples
with defined stretch are counted separately in */stretch_spread_valid; zero
matrices are excluded from its average. Undefined measurements are saved as null.
The rows written (g_t), the errors written (delta_t), the queries and the read outputs
also get an effective rank as stacked matrices.

Mean key, for unit-length keys.
  plain     norm of the running mean of k_t: 1 when every key points the same
            way, near 0 when they are spread out
  decayed   the same, weighted by the state's per-channel decay alpha_t: the
            shared key content the state still holds. Each channel decays at its
            own rate, so the mean's norm can exceed 1; it is reported as a
            coherence, ||E_w[k]|| / sqrt(E_w[||k||^2]) with weights per channel,
            which is between 0 and 1 and equals the plain norm without decay.
Both are also given per position type (key, value, query, filler), plain over the
whole sequence and decayed at the last token. The share of the state's energy
along the decayed mean direction is reported too; a random direction gets
1/head_dim on average.

Usage:
  python Expr/state_geometry.py /path/to/mqar-checkpoints/<jobid>/*.pt --out Expr/results
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RecurrentMemory" / "src"))

from metrics import RANK_TOLERANCE, effective_rank, numerical_rank, polar_summary, stable_rank  # noqa: E402

DEFAULT_SLICES = "64:4,64:8,64:16,128:32,256:64,512:128,1024:256"
# The kernels receive the decay as a log-gate cumulative sum. Its base and the
# chunk it restarts at are internal, so each is tried and the one whose replay
# matches the kernel is used. None means one sum over the whole sequence.
GATE_CANDIDATES = ((2.0, 64), (2.0, 32), (math.e, 64), (math.e, 32), (2.0, None), (math.e, None))
REPLAY_TOLERANCE = 5e-2


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


@contextmanager
def capture_kdn_calls(calls: list):
    """Record the inputs and output of every KDN kernel call, one per KDN layer."""
    import LinearAttn.iso_kdn as iso
    import LinearAttn.kdn_ops.diag_kdn_chunk as diag

    iso_kernel, diag_kernel = iso._chunk_kda_precumsum_paired, diag.chunk_kalman

    def iso_capture(q, k, v, g_cumsum, beta, scale=None):
        out = iso_kernel(q, k, v, g_cumsum, beta, scale)
        q_unit, k_unit = l2norm(q), l2norm(k)  # The kernel normalizes q and k itself.
        calls.append(dict(
            kind="iso_kdn", q=q_unit, k=k_unit, v=v.float(), g_cumsum=g_cumsum.float(),
            write=beta.float()[..., None] * k_unit, out=out.float(),
            scale=q.shape[-1] ** -0.5 if scale is None else float(scale),
        ))
        return out

    def diag_capture(q_memory, k_memory, kappa, v, g_cumsum, scale):
        out = diag_kernel(q_memory, k_memory, kappa, v, g_cumsum, scale)
        calls.append(dict(
            kind="diag_kdn", q=q_memory.float(), k=k_memory.float(), v=v.float(),
            g_cumsum=g_cumsum.float(), write=kappa.float(), out=out.float(), scale=float(scale),
        ))
        return out

    iso._chunk_kda_precumsum_paired, diag.chunk_kalman = iso_capture, diag_capture
    try:
        yield
    finally:
        iso._chunk_kda_precumsum_paired, diag.chunk_kalman = iso_kernel, diag_kernel


def decode_log_alpha(g_cumsum: torch.Tensor, base: float, chunk: int | None) -> torch.Tensor:
    """Per-token natural log decay, without exponentiating tiny retention weights."""
    g = g_cumsum.clone()
    g[:, 1:] -= g_cumsum[:, :-1]
    if chunk is not None:
        g[:, ::chunk] = g_cumsum[:, ::chunk]
    return math.log(base) * g


def decode_alpha(g_cumsum: torch.Tensor, base: float, chunk: int | None) -> torch.Tensor:
    """Per-token decay from a cumulative log-gate [B, T, H, K]."""
    return decode_log_alpha(g_cumsum, base, chunk).exp()


def update_decayed_moments(mean, mean_square, log_weight, key, log_alpha, mask):
    """Update normalized moments with unit-weight observations where mask is true.

    Keeping mass in log space preserves conditional moments through long gaps
    without observations. Inputs broadcast over examples, position types, heads
    and channels; unseen types have log_weight = -inf.
    """
    log_weight = torch.logaddexp(log_weight + log_alpha,
                                torch.where(mask, 0.0, -math.inf))
    fraction = torch.where(mask, -log_weight, -math.inf).exp()
    mean = (1 - fraction) * mean + fraction * key
    mean_square = (1 - fraction) * mean_square + fraction * key.square()
    return mean, mean_square, log_weight


def key_coherence(mean, mean_square, log_weight):
    """Normalized mean norm; undefined when a channel has no retained observations."""
    value = mean.norm(dim=-1) / mean_square.sum(-1).sqrt().clamp_min(1e-12)
    seen = torch.isfinite(log_weight).all(-1)
    return torch.where(seen, value, torch.full_like(value, math.nan))


def recurrence(q, k, v, alpha, write, scale, on_step=None):
    """Token-by-token KDN memory. Returns the outputs and the errors written, [B, T, H, V]."""
    batch, length, heads, key_dim = q.shape
    state = q.new_zeros(batch, heads, key_dim, v.shape[-1])
    outputs, deltas = torch.empty_like(v), torch.empty_like(v)
    for t in range(length):
        state = alpha[:, t, ..., None] * state
        delta = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + write[:, t, ..., None] * delta[..., None, :]
        outputs[:, t] = scale * torch.einsum("bhk,bhkv->bhv", q[:, t], state)
        deltas[:, t] = delta
        if on_step is not None:
            on_step(t, state)
    return outputs, deltas


def replay_error(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst relative error over examples and heads of [B, T, H, V] outputs; inf if anything is not finite.

    Heads whose output is under 1e-3 of the strongest are compared against that
    floor, so a near-silent head can't fail on rounding alone.
    """
    if not (torch.isfinite(estimate).all() and torch.isfinite(reference).all()):
        return math.inf
    difference = (estimate - reference).pow(2).sum((1, 3)).sqrt()
    size = reference.pow(2).sum((1, 3)).sqrt()
    floor = (1e-3 * size.max()).clamp_min(1e-12)
    return (difference / torch.maximum(size, floor)).max().item()


def choose_gate(call: dict, examples: int = 4) -> tuple[float, int | None, float]:
    """Find the gate decoding whose replay reproduces the kernel output."""
    sub = {name: call[name][:examples] for name in ("q", "k", "v", "write", "g_cumsum", "out")}
    errors = {}
    for base, chunk in GATE_CANDIDATES:
        alpha = decode_alpha(sub["g_cumsum"], base, chunk)
        outputs, _ = recurrence(sub["q"], sub["k"], sub["v"], alpha, sub["write"], call["scale"])
        errors[(base, chunk)] = replay_error(outputs, sub["out"])
    (base, chunk), error = min(errors.items(), key=lambda item: item[1])
    if not error <= REPLAY_TOLERANCE:
        listing = ", ".join(f"base={b:.3g} chunk={c}: {e:.3g}" for (b, c), e in errors.items())
        raise RuntimeError(f"No gate decoding reproduces the {call['kind']} kernel output ({listing})")
    return base, chunk, error


def analyse_call(call: dict, base: float, chunk: int | None, context_end: int,
                 query_mask: torch.Tensor, stride: int) -> tuple[dict, list[int], float]:
    """Measure one KDN layer on one batch. Values are summed over the batch, shape [..., H]."""
    q, k, v, write = call["q"], call["k"], call["v"], call["write"]
    log_alpha = decode_log_alpha(call["g_cumsum"], base, chunk)
    alpha = log_alpha.exp()
    batch, length, heads, key_dim = q.shape
    steps = sorted(set(range(0, length, stride)) | {context_end, length - 1})
    step_set = set(steps)
    trace_erank, trace_stable = [], []
    snapshots = {}

    def on_step(t, state):
        if t in step_set:
            singular_values = torch.linalg.svdvals(state)
            trace_erank.append(effective_rank(singular_values).sum(0))
            trace_stable.append(stable_rank(singular_values).sum(0))
        if t == context_end:
            snapshots["context"] = state.clone()
        if t == length - 1:
            snapshots["final"] = state.clone()

    outputs, deltas = recurrence(q, k, v, alpha, write, call["scale"], on_step)
    error = replay_error(outputs, call["out"])

    context = slice(0, context_end + 1)
    write_matrix = torch.einsum("bthk,bthv->bhkv", write[:, context], deltas[:, context])
    reads = int(query_mask[0].sum())
    q_reads = q[query_mask].view(batch, reads, heads, key_dim).transpose(1, 2)  # [B, H, n, K]
    o_reads = outputs[query_mask].view(batch, reads, heads, -1).transpose(1, 2)  # [B, H, n, V]
    read_activity = q_reads.mT @ o_reads
    read_operator = torch.linalg.pinv(q_reads, rtol=RANK_TOLERANCE) @ o_reads
    fit_residual = (q_reads @ read_operator - o_reads).norm(dim=(-2, -1)) / o_reads.norm(dim=(-2, -1)).clamp_min(1e-12)

    stats = {"trace/state_erank": torch.stack(trace_erank), "trace/state_stable_rank": torch.stack(trace_stable)}
    for name, matrix in (("state_context", snapshots["context"]), ("state_final", snapshots["final"]),
                         ("write", write_matrix), ("read_activity", read_activity),
                         ("read_operator", read_operator)):
        for metric, value in polar_summary(matrix).items():
            stats[f"{name}/{metric}"] = value.nansum(0)
            if metric == "stretch_spread":
                stats[f"{name}/stretch_spread_valid"] = torch.isfinite(value).sum(0)
    stats["read/fit_residual"] = fit_residual.sum(0)
    stats["read/query_rank"] = numerical_rank(torch.linalg.svdvals(q_reads)).sum(0)
    for name, rows in (("written_rows", write[:, context].transpose(1, 2)),
                       ("written_errors", deltas[:, context].transpose(1, 2)),
                       ("queries", q_reads), ("read_outputs", o_reads)):
        stats[f"stack/{name}_erank"] = effective_rank(torch.linalg.svdvals(rows)).sum(0)

    # Mean key. Position types: 0 key, 1 value, 2 query, 3 filler; index 4 is all tokens.
    positions = torch.arange(length, device=k.device)[None, :].expand(batch, -1)
    in_context = positions <= context_end
    type_names = ("key", "value", "query", "filler")
    masks = torch.stack([in_context & (positions % 2 == 0), in_context & (positions % 2 == 1),
                         query_mask, ~in_context & ~query_mask, torch.ones_like(query_mask)], dim=2).to(k.dtype)

    plain_mean = k.cumsum(1) / torch.arange(1, length + 1, device=k.device, dtype=k.dtype)[None, :, None, None]
    mean = torch.zeros(batch, masks.shape[2], heads, key_dim, device=k.device, dtype=k.dtype)
    mean_square = torch.zeros_like(mean)
    log_weight = torch.full_like(mean, -math.inf)
    decayed_trace = []
    for t in range(length):
        mask_t = masks[:, t, :, None, None].bool()
        mean, mean_square, log_weight = update_decayed_moments(
            mean, mean_square, log_weight, k[:, t, None], log_alpha[:, t, None], mask_t)
        if t in step_set:
            decayed_trace.append(key_coherence(mean[:, 4], mean_square[:, 4], log_weight[:, 4]))
        if t == context_end:
            context_coherence = key_coherence(mean[:, 4], mean_square[:, 4], log_weight[:, 4])
            context_mean = mean[:, 4]
            direction = context_mean / context_mean.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    final_coherence = key_coherence(mean, mean_square, log_weight)  # [B, 5, H]

    stats["trace/mean_key_plain"] = plain_mean.norm(dim=-1)[:, steps].sum(0)
    stats["trace/mean_key_decayed_coherence"] = torch.stack(decayed_trace, dim=1).sum(0)
    stats["mean_key/plain_context"] = plain_mean[:, context_end].norm(dim=-1).sum(0)
    stats["mean_key/decayed_coherence_context"] = context_coherence.sum(0)
    for name, matrix in (("state_context", snapshots["context"]), ("write", write_matrix)):
        along = torch.einsum("bhk,bhkv->bhv", direction, matrix).pow(2).sum(-1)
        stats[f"mean_key/energy_along_mean_{name}"] = (along / matrix.pow(2).sum((-2, -1)).clamp_min(1e-12)).sum(0)
    type_means = {}
    for index, name in enumerate(type_names):
        weights = masks[:, :, index, None, None]
        type_means[name] = (k * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        stats[f"mean_key/{name}_positions_plain"] = type_means[name].norm(dim=-1).sum(0)
        stats[f"mean_key/{name}_positions_decayed_coherence"] = final_coherence[:, index].sum(0)
    stats["mean_key/cos_key_vs_query_positions"] = torch.nn.functional.cosine_similarity(
        type_means["key"], type_means["query"], dim=-1).sum(0)
    return stats, steps, error


def average(totals: dict, examples: int) -> dict:
    """Average polar measurements over valid examples and other sums over the batch."""
    averaged = {}
    for name, value in totals.items():
        if name.endswith(("/rotation_deg", "/reflection", "/stretch_spread")):
            count = "/stretch_spread_valid" if name.endswith("/stretch_spread") else "/full_rank"
            defined = totals[name.rsplit("/", 1)[0] + count]
            value = torch.where(defined > 0, value / defined.clamp_min(1), torch.full_like(value, math.nan))
        else:
            value = value / examples
        averaged[name] = value.tolist()
    return averaged


def json_safe(value):
    """Represent undefined numeric measurements as JSON null, including nested traces."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def build_model(checkpoint: dict):
    from zoology.config import ModelConfig
    from zoology.model import LanguageModel

    model = LanguageModel(ModelConfig.model_validate(checkpoint["config"]["model"]))
    model.load_state_dict(checkpoint["model"])
    for module in model.modules():
        # DiagKDNMixer re-zeros a projection on its first forward pass, which is
        # meant for a fresh model and would erase the trained weights here.
        if hasattr(module, "_restore_omega_init"):
            module._restore_omega_init = False
    return model.cuda().eval()


def analyse_checkpoint(path: Path, slices, examples: int, batch_size: int, stride: int, seed: int) -> dict:
    from zoology.data.multiquery_ar import MQARConfig

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    vocab_size = checkpoint["config"]["model"]["vocab_size"]
    gate = None
    result = {
        "checkpoint": str(path),
        "run_id": checkpoint["config"].get("run_id"),
        "best_epoch": checkpoint["epoch"],
        "training_metrics": checkpoint["metrics"],
        "slices": [],
    }
    for seq_len, pairs in slices:
        data = MQARConfig(vocab_size=vocab_size, input_seq_len=seq_len, num_examples=examples,
                          num_kv_pairs=pairs).build(seed=seed)
        totals, correct, answered, worst_error, steps = {}, 0, 0, 0.0, None
        for start in range(0, examples, batch_size):
            inputs = data.inputs[start:start + batch_size].cuda()
            labels = data.labels[start:start + batch_size].cuda()
            calls = []
            with torch.no_grad(), capture_kdn_calls(calls):
                predictions = model(inputs).argmax(-1)
            answer = labels != -100
            correct += (predictions[answer] == labels[answer]).sum().item()
            answered += answer.sum().item()
            if not calls:
                raise RuntimeError(f"{path.name} has no KDN layer")
            for layer, call in enumerate(calls):
                if gate is None:
                    gate = choose_gate(call)
                    result["gate_decoding"] = {"kind": call["kind"], "base": gate[0], "chunk": gate[1],
                                               "replay_error": gate[2]}
                stats, steps, error = analyse_call(call, gate[0], gate[1], 2 * pairs - 1, answer, stride)
                if not error <= REPLAY_TOLERANCE:
                    raise RuntimeError(f"Replay differs from the kernel by {error:.3g} on {seq_len}:{pairs}")
                worst_error = max(worst_error, error)
                layer_totals = totals.setdefault(f"kdn_layer_{layer}", {})
                for name, value in stats.items():
                    layer_totals[name] = layer_totals.get(name, 0) + value
        result["slices"].append({
            "seq_len": seq_len,
            "num_kv_pairs": pairs,
            "examples": examples,
            "accuracy": correct / answered,
            "replay_error": worst_error,
            "trace_steps": steps,
            "layers": {layer: average(values, examples) for layer, values in totals.items()},
        })
    return result


def print_summary(result: dict) -> None:
    print(f"\n{result['run_id']} (best epoch {result['best_epoch']}), gate {result.get('gate_decoding')}")
    columns = ("state_context/erank", "write/erank", "read_operator/erank", "read/fit_residual",
               "state_context/rotation_deg", "state_context/stretch_spread",
               "mean_key/decayed_coherence_context", "mean_key/energy_along_mean_state_context")
    labels = ("S erank", "W erank", "R erank", "R resid", "S rot", "S spread", "k coher", "S on mean")
    print(f"{'len:pairs':>10}{'acc':>7}" + "".join(f"{label:>10}" for label in labels) + "   (layer 0, mean over heads)")
    for row in result["slices"]:
        layer = next(iter(row["layers"].values()))
        cells = "".join(f"{sum(layer[c]) / len(layer[c]):>10.3f}" for c in columns)
        print(f"{row['seq_len']:>5}:{row['num_kv_pairs']:<4}{row['accuracy']:>7.3f}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--slices", default=DEFAULT_SLICES, help="seq_len:num_kv_pairs, comma-separated")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1, help="record the state rank every this many tokens")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    slices = [tuple(int(part) for part in item.split(":")) for item in args.slices.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.checkpoints:
        result = analyse_checkpoint(path, slices, args.examples, args.batch_size, args.stride, args.seed)
        destination = args.out / f"{path.stem}.json"
        destination.write_text(json.dumps(json_safe(result), allow_nan=False))
        print_summary(result)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
