"""Read-only probes of an edit's immediate effect on internal memory reconstruction.

For each value token in the current context chunk, re-read the state with that
token's memory key and compare against its internal value vector. This is the
write equation's reconstruction target, NOT vocabulary accuracy or a future
query read. Probe vectors never influence edits. Results pool examples/heads
within each layer, boundary and original pair position.
"""
from __future__ import annotations

import csv
import math

import torch


class BoundaryProbes:
    def __init__(self, pairs):
        self.pairs = pairs
        self.totals = {}

    def observe(self, layer, start, end, keys, values, writes, before, after, basis, scheduled_end=None):
        scheduled_end = end if scheduled_end is None else scheduled_end
        # Probe only full context chunks. Never treat query/filler tokens as pairs.
        if end > 2 * self.pairs:
            return
        indices = torch.arange(start, end, device=keys.device)
        indices = indices[indices % 2 == 1]
        if not indices.numel():
            return
        k, v, g = keys[:, indices], values[:, indices], writes[:, indices]
        pre = torch.einsum("bphk,bhkv->bphv", k, before)
        post = torch.einsum("bphk,bhkv->bphv", k, after)
        metrics = [(pre - v).square().sum(-1), (post - v).square().sum(-1),
                   v.square().sum(-1), (post - pre).square().sum(-1)]
        for vectors in (k, g):
            if basis is None:
                share = torch.full_like(metrics[0], math.nan)
            else:
                projected = torch.einsum("bphk,bhkr->bphr", vectors, basis)
                energy = vectors.square().sum(-1)
                share = projected.square().sum(-1) / torch.where(energy > 0, energy, 1.)
            metrics.append(share)
        # One host transfer per boundary; preserve sums for unequal batch sizes.
        sums = torch.stack(metrics, -1).sum((0, 2)).detach().cpu().tolist()
        count = keys.shape[0] * keys.shape[2]
        for token, row in zip(indices.tolist(), sums):
            key = (layer, start, end, scheduled_end, token // 2 + 1)
            previous = self.totals.setdefault(key, [0.] * 7)
            for i, value in enumerate([count, *row]):
                previous[i] += value

    def report(self):
        rows = []
        for (layer, start, end, scheduled_end, pair), sums in sorted(self.totals.items()):
            count, pre, post, target, drift, key_share, gain_share = sums
            rows.append(dict(
                layer=layer, chunk_start=start, chunk_end_exclusive=end,
                scheduled_chunk_end_exclusive=scheduled_end, edit_token_index=end,
                pair_position=pair, value_token_index=2 * pair - 1,
                tokens_before_boundary=end - 2 * pair, terminal_pair=2 * pair == end,
                scheduled_terminal_pair=2 * pair == scheduled_end,
                scheduled_penultimate_pair=2 * pair == scheduled_end - 2,
                example_heads=int(count),
                squared_error_before=pre / count, squared_error_after=post / count,
                squared_error_change=(post - pre) / count,
                target_energy=target / count,
                relative_squared_error_before=pre / target if target > 0 else None,
                relative_squared_error_after=post / target if target > 0 else None,
                read_change_energy=drift / count,
                key_projection_share=key_share / count if math.isfinite(key_share) else None,
                gain_projection_share=gain_share / count if math.isfinite(gain_share) else None,
            ))
        return rows


def summarize_positions(report, seq_len, pairs, chunk, offset, ranges, tail_policy="none"):
    """Keep logical chunk groups fixed when moving the edit before the final pair."""
    terminal = {end // 2 for start, end in ranges(seq_len, chunk, offset)
                if start > 0 and end < seq_len and end <= 2 * pairs and end % 2 == 0}
    rows = report["recall_by_pair_position"]
    penultimate = {p - 1 for p in terminal if p > 1}
    edit_terminal = {((end - 2) if tail_policy == "before_pair" else end) // 2
                     for start, end in ranges(seq_len, chunk, offset)
                     if start > 0 and end < seq_len and end <= 2 * pairs
                     and end % 2 == 0}

    def aggregate(selected):
        correct = sum(row["correct"] for row in selected)
        total = sum(row["total"] for row in selected)
        return dict(correct=correct, total=total, accuracy=correct / total if total else None)

    return dict(
        terminal_pair_positions=sorted(terminal),
        penultimate_pair_positions=sorted(penultimate),
        edit_terminal_pair_positions=sorted(edit_terminal),
        terminal=aggregate([row for row in rows if row["pair_position"] in terminal]),
        penultimate=aggregate([row for row in rows if row["pair_position"] in penultimate]),
        earlier=aggregate([row for row in rows if row["pair_position"] not in terminal | penultimate]),
        edit_terminal=aggregate([row for row in rows if row["pair_position"] in edit_terminal]),
        other=aggregate([row for row in rows if row["pair_position"] not in terminal]),
        quarters=[aggregate(rows[i * pairs // 4:(i + 1) * pairs // 4]) for i in range(4)],
    )


def paired_sequence_delta(entry, replay):
    """Keep the sequence as the uncertainty unit, rather than individual queries."""
    if "per_example_correct_by_pair" not in entry:
        return None
    if entry["per_example_total_by_pair"] != replay["per_example_total_by_pair"]:
        raise ValueError("Paired settings must evaluate the same query counts per example/pair")
    deltas = [(sum(a) - sum(b)) / sum(n)
              for a, b, n in zip(entry["per_example_correct_by_pair"],
                                  replay["per_example_correct_by_pair"],
                                  entry["per_example_total_by_pair"]) if sum(n)]
    if not deltas:
        return None
    mean = sum(deltas) / len(deltas)
    se = (math.sqrt(sum((x - mean) ** 2 for x in deltas) / (len(deltas) * (len(deltas) - 1)))
          if len(deltas) > 1 else None)
    return dict(examples=len(deltas), mean_accuracy_delta=mean, standard_error=se)


def write_diagnostic_csvs(result, stem):
    """Write compact recall summaries and detailed per-layer boundary probes."""
    summaries, probes = [], []
    for row in result["slices"]:
        for name, entry in row["settings"].items():
            prefix = dict(seq_len=row["seq_len"], num_kv_pairs=row["num_kv_pairs"], setting=name)
            summary = entry.get("position_summary")
            if summary is not None:
                baseline = entry["replay_position_summary"]
                delta = entry.get("paired_sequence_delta") or {}
                summaries.append(dict(
                    **prefix, chunk=entry["chunk"], chunk_offset=entry["chunk_offset"],
                    tail_policy=entry.get("tail_policy", "none"),
                    accuracy=entry["accuracy"], replay_accuracy=row["settings"]["replay"]["accuracy"],
                    terminal_accuracy=summary["terminal"]["accuracy"],
                    replay_terminal_accuracy=baseline["terminal"]["accuracy"],
                    penultimate_accuracy=summary["penultimate"]["accuracy"],
                    replay_penultimate_accuracy=baseline["penultimate"]["accuracy"],
                    earlier_accuracy=summary["earlier"]["accuracy"],
                    replay_earlier_accuracy=baseline["earlier"]["accuracy"],
                    edit_terminal_accuracy=summary["edit_terminal"]["accuracy"],
                    replay_edit_terminal_accuracy=baseline["edit_terminal"]["accuracy"],
                    other_accuracy=summary["other"]["accuracy"],
                    replay_other_accuracy=baseline["other"]["accuracy"],
                    terminal_total=summary["terminal"]["total"], other_total=summary["other"]["total"],
                    penultimate_total=summary["penultimate"]["total"], earlier_total=summary["earlier"]["total"],
                    edit_terminal_total=summary["edit_terminal"]["total"],
                    paired_mean_delta=delta.get("mean_accuracy_delta"),
                    paired_standard_error=delta.get("standard_error"),
                ))
            probes.extend(dict(**prefix, **probe) for probe in entry.get("boundary_probes", []))
    destinations = []
    for suffix, records in (("summary", summaries), ("boundary-probes", probes)):
        if records:
            path = stem.with_name(f"{stem.name}-{suffix}.csv")
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            destinations.append(path)
    return destinations
