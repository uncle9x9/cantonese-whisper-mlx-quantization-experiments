# Copyright © 2023-2024 Apple Inc.
#
# Vendored from ml-explore/mlx-examples, commit 796f5b53cab69a3d48a44233ce21aae889e94a08,
# path whisper/convert.py (MIT licence; full text in LICENSE-convert.py-MIT). Three changes from upstream:
#  1. load_torch_weights_and_config() is patched to merge SHARDED safetensors
#     (model-00001-of-00002.safetensors + model-00002-of-00002.safetensors +
#     model.safetensors.index.json), because khleeloo/whisper-large-v3-cantonese
#     ships shards and upstream only knows how to open a single "model.safetensors"
#     file. Without the patch this script raises FileNotFoundError on this checkpoint.
#  2. The upstream OpenAI .pt download table (_MODELS/_ALIGNMENT_HEADS) and its
#     _download() helper are removed as unused for HF-repo conversions; _download()
#     is stubbed to raise if ever called.
#  3. The output weights file is named weights.safetensors (upstream: model.safetensors),
#     the file name mlx-whisper 0.4.3 loads.
# 2026-11-12 experiment: reject existing output directories, require exact remapped
# parameter keys/shapes, and validate requested affine quantization settings.

import argparse
import copy
import hashlib
import json
import os
import urllib
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_whisper import torch_whisper
from mlx_whisper.whisper import ModelDimensions, Whisper
from tqdm import tqdm

_VALID_DTYPES = {"float16", "float32"}

_MODELS = {}  # upstream OpenAI .pt table omitted; not used for this experiment
_ALIGNMENT_HEADS = {}


def _download(url: str, root: str) -> str:
    raise NotImplementedError("not used for HF-repo conversions")


def available_models() -> List[str]:
    return list(_MODELS.keys())


def hf_to_pt(weights, config):
    config = {
        "n_mels": config["num_mel_bins"],
        "n_audio_ctx": config["max_source_positions"],
        "n_audio_state": config["d_model"],
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_vocab": config["vocab_size"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_state": config["d_model"],
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }

    def remap(k):
        k = k.replace("model.", "")
        k = k.replace(".layers", ".blocks")
        k = k.replace(".self_attn", ".attn")
        k = k.replace(".attn_layer_norm", ".attn_ln")
        k = k.replace(".encoder_attn.", ".cross_attn.")
        k = k.replace(".encoder_attn_layer_norm", ".cross_attn_ln")
        k = k.replace(".final_layer_norm", ".mlp_ln")
        k = k.replace(".q_proj", ".query")
        k = k.replace(".k_proj", ".key")
        k = k.replace(".v_proj", ".value")
        k = k.replace(".out_proj", ".out")
        k = k.replace(".fc1", ".mlp1")
        k = k.replace(".fc2", ".mlp2")
        k = k.replace("embed_positions.weight", "positional_embedding")
        k = k.replace("decoder.embed_tokens", "decoder.token_embedding")
        k = k.replace("encoder.layer_norm", "encoder.ln_post")
        k = k.replace("decoder.layer_norm", "decoder.ln")
        return k

    weights.pop("proj_out.weight", None)
    weights = {remap(k): v for k, v in weights.items()}
    return weights, config


def _load_possibly_sharded_safetensors(model_dir: Path) -> dict:
    """PATCH vs upstream: upstream only tries `model.safetensors`. khleeloo's repo
    is sharded (model-00001-of-00002.safetensors, model-00002-of-00002.safetensors,
    model.safetensors.index.json — see model.safetensors.index.json weight_map).
    Merge all shards named model*.safetensors before handing off to hf_to_pt.
    """
    single = model_dir / "model.safetensors"
    if single.is_file():
        return dict(mx.load(str(single)).items())

    index_path = model_dir / "model.safetensors.index.json"
    shard_files = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(
            f"No model.safetensors or sharded model-*-of-*.safetensors found in {model_dir}"
        )
    if index_path.is_file():
        with open(index_path) as f:
            index = json.load(f)
        expected = set(index.get("weight_map", {}).values())
        found = {p.name for p in shard_files}
        missing = expected - found
        if missing:
            raise FileNotFoundError(f"Missing safetensors shards: {missing}")

    weights = {}
    for shard in shard_files:
        weights.update(mx.load(str(shard)))
    return weights


def load_torch_weights_and_config(
    name_or_path: str,
    download_root: str = None,
):
    if download_root is None:
        download_root = os.path.join(os.path.expanduser("~"), ".cache/whisper")

    alignment_heads = None
    if name_or_path in _MODELS:
        alignment_heads = _ALIGNMENT_HEADS[name_or_path]
        name_or_path = _download(_MODELS[name_or_path], download_root)
    elif not Path(name_or_path).exists():
        from huggingface_hub import snapshot_download

        # PATCH vs upstream: upstream's allow_patterns is
        #   ["*.json", "pytorch_model.bin", "model.safetensors", "*.txt"]
        # which matches neither model-00001-of-00002.safetensors nor
        # model.safetensors.index.json. Widened to model*.safetensors*.
        name_or_path = snapshot_download(
            repo_id=name_or_path,
            allow_patterns=[
                "*.json",
                "pytorch_model.bin",
                "model*.safetensors*",
                "*.txt",
            ],
        )

    if name_or_path.endswith(".pt"):
        checkpoint = torch.load(name_or_path, map_location="cpu", weights_only=False)
        weights, config = checkpoint["model_state_dict"], checkpoint["dims"]
    else:
        name_or_path = Path(name_or_path)
        pt_path = name_or_path / "pytorch_model.bin"
        if pt_path.is_file():
            weights = torch.load(pt_path, map_location="cpu")
        else:
            weights = _load_possibly_sharded_safetensors(name_or_path)
        with open(name_or_path / "config.json", "r") as fp:
            config = json.load(fp)
        weights, config = hf_to_pt(weights, config)

    return weights, config, alignment_heads


def convert(name_or_path: str, dtype: mx.Dtype = mx.float16):
    def remap(key, value):
        key = key.replace("mlp.0", "mlp1")
        key = key.replace("mlp.2", "mlp2")
        if "conv" in key and value.ndim == 3:
            value = value.swapaxes(1, 2)
        if isinstance(value, torch.Tensor):
            value = mx.array(value.detach())
        return key, value.astype(dtype)

    weights, config, alignment_heads = load_torch_weights_and_config(name_or_path)
    weights.pop("encoder.positional_embedding", None)
    weights = dict(remap(k, v) for k, v in weights.items())

    model_dims = ModelDimensions(**config)
    model = Whisper(model_dims, dtype)
    expected = dict(tree_flatten(model.parameters()))
    # MLX generates alignment-head indices; HF has no learned weight for this buffer.
    # Preserve the same default as the original converter, while checking all weights.
    weights.setdefault("alignment_heads", expected["alignment_heads"])
    if set(weights) != set(expected):
        raise ValueError(f"Parameter mismatch: missing={set(expected)-set(weights)}, extra={set(weights)-set(expected)}")
    for key, value in weights.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"Shape mismatch for {key}: {value.shape} != {expected[key].shape}")
    model.load_weights(list(weights.items()), strict=True)

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    model = Whisper(ModelDimensions(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    nn.quantize(model, args.q_group_size, args.q_bits)

    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper weights to MLX.")
    parser.add_argument("--torch-name-or-path", type=str, required=True)
    parser.add_argument("--mlx-path", type=str, default="mlx_models")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("-q", "--quantize", action="store_true")
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--q-bits", type=int, default=4)

    args = parser.parse_args()

    if Path(args.mlx_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.mlx_path}")
    if args.quantize and (args.q_bits not in (2, 3, 4, 5, 6, 8) or args.q_group_size not in (32, 64, 128)):
        raise ValueError("Unsupported affine bits/group size")

    assert args.dtype in _VALID_DTYPES, f"dtype {args.dtype} not found in {_VALID_DTYPES}"
    dtype = getattr(mx, args.dtype)

    print("[INFO] Loading")
    model = convert(args.torch_name_or_path, dtype)
    config = asdict(model.dims)
    weights = dict(tree_flatten(model.parameters()))

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Saving")
    mx.save_safetensors(str(mlx_path / "weights.safetensors"), weights)

    with open(str(mlx_path / "config.json"), "w") as f:
        config["model_type"] = "whisper"
        json.dump(config, f, indent=4)
