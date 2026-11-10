"""Train matched MQAR models with and without shrink from scratch; see Expr/TRAIN_SHRINK.md.

Same two-layer architecture as mqar_configs.py: BaseConv, then IsoKDN.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "RecurrentMemory" / "src"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gammas", type=float, nargs="+", default=[1., .9])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=64)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-examples", type=int, default=2048,
                        help="Examples per training slice, fixed across epochs")
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mqar_shrink"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    from kdn_shrink import validate_shrink
    for gamma in args.gammas:
        validate_shrink(gamma, args.chunk, args.window, args.rank, args.head_dim)
    for name in ("heads", "epochs", "train_examples", "test_examples", "batch_size", "eval_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.lr < float("inf") or not 0 < args.grad_clip < float("inf"):
        parser.error("learning rate and grad clip must be positive and finite")
    if len(set(args.gammas)) != len(args.gammas) or len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate gammas or seeds would overwrite a run")
    return args


def experiment_plan(args):
    train_slices = [(64, 4), (128, 8), (256, 16), (256, 32), (256, 64)]
    test_slices = [(64, 4), (64, 8), (64, 16), (128, 32), (256, 64), (512, 128), (1024, 256)]
    runs = []
    for seed in args.seeds:
        for gamma in args.gammas:
            name = (f"iso_kdn-h{args.heads}-d{args.head_dim}-g{gamma:g}"
                    f"-c{args.chunk}-w{args.window}-r{args.rank}-lr{args.lr:g}-seed{seed}")
            runs.append(dict(run_id=name, seed=seed, gamma=gamma))
    return dict(train_slices=train_slices, test_slices=test_slices,
                train_examples_per_slice=args.train_examples,
                test_examples_per_slice=args.test_examples, epochs=args.epochs,
                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                layers=2, d_model=args.heads*args.head_dim, runs=runs)


def make_configs(args, plan):
    from zoology.config import DataConfig, LoggerConfig, ModelConfig, ModuleConfig, TrainConfig
    from zoology.data.multiquery_ar import MQARConfig

    data = DataConfig(
        train_configs=[MQARConfig(vocab_size=8192, input_seq_len=length,
                                  num_kv_pairs=pairs, num_examples=args.train_examples)
                       for length, pairs in plan["train_slices"]],
        test_configs=[MQARConfig(vocab_size=8192, input_seq_len=length,
                                 num_kv_pairs=pairs, num_examples=args.test_examples)
                      for length, pairs in plan["test_slices"]],
        batch_size=(args.batch_size, args.eval_batch_size),
        **({"cache_dir": os.environ["MQAR_CACHE_DIR"]} if os.environ.get("MQAR_CACHE_DIR") else {}),
    )
    logger = (LoggerConfig(project_name=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"])
              if os.environ.get("WANDB_PROJECT") and os.environ.get("WANDB_ENTITY") else LoggerConfig())
    configs = []
    for run in plan["runs"]:
        # Same block layout as mqar_configs.py: Hybrid alternates by layer, so
        # layer 0 is BaseConv and layer 1 is the IsoKDN memory.
        conv = dict(name="zoology.mixers.base_conv.BaseConv",
                    kwargs={"l_max": 1024, "kernel_size": 3, "implicit_long_conv": True})
        memory = dict(name="zoology_mixers.IsoKDNMixer", kwargs=dict(
            num_heads=args.heads, head_dim=args.head_dim, shrink_gamma=run["gamma"],
            shrink_chunk=args.chunk, shrink_window=args.window, shrink_rank=args.rank))
        model = ModelConfig(
            block_type="TransformerBlock", d_model=plan["d_model"], n_layers=plan["layers"],
            sequence_mixer=ModuleConfig(name="zoology.mixers.hybrid.Hybrid",
                                        kwargs={"configs": [conv, memory]}),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            vocab_size=8192, max_position_embeddings=0, name=run["run_id"],
        )
        configs.append(TrainConfig(
            model=model, data=data, learning_rate=args.lr, max_epochs=args.epochs,
            logger=logger, slice_keys=["num_kv_pairs"], seed=run["seed"],
            sweep_id="mqar-shrink", run_id=run["run_id"],
        ))
    return configs


def evaluate_final(config, args, plan):
    """Fresh, identical held-out examples for all train-gamma/eval-gamma cells."""
    import torch
    from zoology.data.multiquery_ar import MQARConfig
    from zoology.utils import set_determinism
    sys.path.insert(0, str(ROOT / "Expr"))
    from state_geometry import build_model
    from shrink_spread import Recall, query_pair_positions

    checkpoint_path = args.output / (config.run_id + ".last.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    layers = [m for m in model.modules() if hasattr(m, "shrink_gamma")]
    if len(layers) != 1:
        raise RuntimeError(f"Expected exactly one shrink-enabled memory layer; found {len(layers)}")
    training_gamma = layers[0].shrink_gamma
    results, rows = [], []
    for length, pairs in plan["test_slices"]:
        eval_seed = config.seed + 100_000
        set_determinism(eval_seed)
        data = MQARConfig(vocab_size=8192, input_seq_len=length,
                          num_kv_pairs=pairs, num_examples=args.test_examples).build(seed=eval_seed)
        for gamma in sorted(set([1., *args.gammas]), reverse=True):
            layers[0].shrink_gamma = gamma
            recall = Recall(pairs, per_example=True)
            with torch.no_grad():
                for start in range(0, args.test_examples, args.eval_batch_size):
                    inputs = data.inputs[start:start+args.eval_batch_size].cuda()
                    labels = data.labels[start:start+args.eval_batch_size].cuda()
                    positions = query_pair_positions(inputs, labels, pairs)
                    recall.update(model(inputs).argmax(-1), labels, positions)
            report = recall.report()
            # Same scheduled terminal positions for every cell, even gamma=1.
            terminal = torch.tensor([(2*(p+1)) % args.chunk == 0
                                     and 2*(p+1) > args.chunk
                                     and 2*(p+1) < length for p in range(pairs)])
            row = dict(run_id=config.run_id, seed=config.seed, train_gamma=training_gamma,
                       eval_gamma=gamma, seq_len=length, num_kv_pairs=pairs,
                       accuracy=report["accuracy"])
            for label, mask in (("terminal", terminal), ("other", ~terminal)):
                correct, total = recall.correct[mask].sum().item(), recall.total[mask].sum().item()
                row[label+"_correct"], row[label+"_total"] = correct, total
                row[label+"_accuracy"] = correct / total if total else None
            rows.append(row)
            results.append(dict(**row, eval_seed=eval_seed, recall=report))
            print(f"train gamma={training_gamma:g}, eval gamma={gamma:g}, "
                  f"{length}:{pairs}: accuracy={report['accuracy']:.4f}")
    (args.output / (config.run_id + ".eval.json")).write_text(json.dumps(results, indent=2)+"\n")
    with (args.output / (config.run_id + ".eval.csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    plan = experiment_plan(args)
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return 0
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU and the existing Zoology/FLA environment")
    configs = make_configs(args, plan)
    args.output.mkdir(parents=True, exist_ok=True)
    for config in configs:
        for suffix in (".pt", ".last.pt"):
            if (args.output / (config.run_id + suffix)).exists():
                raise FileExistsError("Output already contains this run; choose a new --output directory")
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    from train_mqar import train
    rows = []
    for config in configs:
        # Reset seed before model construction AND data preparation per arm.
        # eigh uses no random numbers, so the intervention does not advance RNG.
        train(config, args.output, args.grad_clip, fixed_budget=True)
        rows.extend(evaluate_final(config, args, plan))
        with (args.output / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
