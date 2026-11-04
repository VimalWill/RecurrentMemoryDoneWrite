"""MQAR recall sweep for GDN-2, KDN and MesaNet, following BASED Figure 2.

Data, depth, block layout and learning rates match zoology's
experiments/paper_configs/arxiv24_based_figure2/configs.py, except that the test
batch size is written as an integer. Each row of STATE_POINTS is one memory size;
KDN needs d_model = heads * head_dim with head_dim in {64, 128}.

Environment: MQAR_MIXERS, MQAR_EPOCHS, MQAR_LRS, MQAR_MEMORY (comma-separated filters),
MQAR_CACHE_DIR, WANDB_PROJECT, WANDB_ENTITY.
"""
import os
import uuid

import numpy as np
from zoology.config import DataConfig, LoggerConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB_SIZE = 8_192
N_LAYERS = 2  # Hybrid alternates by layer: layer 0 is BaseConv, layer 1 is the memory mixer.

# memory floats in the mixer layer: (d_model, heads, head_dim)
STATE_POINTS = {
    4_096: (64, 1, 64),
    8_192: (128, 2, 64),
    16_384: (256, 4, 64),
    32_768: (256, 2, 128),
    65_536: (512, 4, 128),
}
MIXERS = {
    "gdn2": "zoology_mixers.GDN2Mixer",
    "iso_kdn": "zoology_mixers.IsoKDNMixer",
    "diag_kdn": "zoology_mixers.DiagKDNMixer",
    "mesa_net": "zoology_mixers.MesaNetMixer",
}


def _env_list(name, default, cast=str):
    value = os.environ.get(name)
    return [cast(item) for item in value.split(",")] if value else list(default)


def _make_data():
    train = [(64, 100_000, 4), (128, 20_000, 8), (256, 20_000, 16), (256, 20_000, 32), (256, 20_000, 64)]
    test = [(64, 4), (64, 8), (64, 16), (128, 32), (256, 64), (512, 128), (1024, 256)]
    return DataConfig(
        train_configs=[
            MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=length, num_examples=examples, num_kv_pairs=pairs)
            for length, examples, pairs in train
        ],
        test_configs=[
            MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=length, num_examples=1_000, num_kv_pairs=pairs)
            for length, pairs in test
        ],
        batch_size=(256, 32),
        **({"cache_dir": os.environ["MQAR_CACHE_DIR"]} if os.environ.get("MQAR_CACHE_DIR") else {}),
    )


def _make_model(mixer_name, memory):
    d_model, heads, head_dim = STATE_POINTS[memory]
    if mixer_name == "mesa_net":
        # MesaNet stores two matrices per head, so half the heads gives the same memory.
        if heads % 2:
            return None
        heads //= 2
    conv = dict(
        name="zoology.mixers.base_conv.BaseConv",
        kwargs={"l_max": 1024, "kernel_size": 3, "implicit_long_conv": True},
    )
    mixer = dict(name=MIXERS[mixer_name], kwargs={"num_heads": heads, "head_dim": head_dim})
    return ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=N_LAYERS,
        sequence_mixer=ModuleConfig(name="zoology.mixers.hybrid.Hybrid", kwargs={"configs": [conv, mixer]}),
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=0,
        name=f"{mixer_name}-m{memory}",
    )


sweep_name = f"mqar-recall-{uuid.uuid4().hex[:6]}"
logger = (
    LoggerConfig(project_name=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"])
    if os.environ.get("WANDB_PROJECT") and os.environ.get("WANDB_ENTITY")
    else LoggerConfig()
)
data = _make_data()

configs = []
for mixer_name in _env_list("MQAR_MIXERS", MIXERS):
    for memory in _env_list("MQAR_MEMORY", STATE_POINTS, int):
        model = _make_model(mixer_name, memory)
        if model is None:
            continue
        for lr in _env_list("MQAR_LRS", np.logspace(-3, -1.5, 4), float):
            run_id = f"{model.name}-lr{lr:.1e}"
            configs.append(
                TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=float(lr),
                    max_epochs=int(os.environ.get("MQAR_EPOCHS", 32)),
                    logger=logger,
                    slice_keys=["num_kv_pairs"],
                    sweep_id=sweep_name,
                    run_id=run_id,
                )
            )
