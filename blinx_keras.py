"""
blinx_keras: Batched Multi-LoRA in Keras 3.

Enables serving multiple independent LoRA adapters simultaneously within the same batch
during inference with minimal overhead on Keras models (Dense and EinsumDense).

@author: nmilosev@alphadeep.ai
"""

import contextvars
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import keras
from keras import layers, ops

# Re-use core rules and config from blinx
from blinx import LoraConfig, LoraConfigManager, LoraRule

# Context variable holding the active batched LoRA dictionary during forward pass
_current_lora = contextvars.ContextVar("blinx_current_lora", default=None)


def compute_lora_delta(
    x: Any,
    lora_a: Any,
    lora_b: Any,
    alpha: float,
    rank: int,
    lora_out_transpose: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Computes batched LoRA delta using universal einsum contraction.

    Args:
        x: Input tensor with leading batch dim, shape (B, ..., D_in).
        lora_a: Batched LoRA A matrix, shape (B, D_in, R).
        lora_b: Batched LoRA B matrix, shape (B, R, D_out).
        alpha: LoRA alpha scaling factor.
        rank: LoRA rank R.
        lora_out_transpose: Optional output transpose permutation.

    Returns:
        LoRA delta tensor matching base layer output shape, (B, ..., D_out).
    """
    scaling = float(alpha) / float(rank)
    # Project input down to rank R: (B, ..., D_in) x (B, D_in, R) -> (B, ..., R)
    xa = ops.einsum("b...i,bir->b...r", x, lora_a)
    # Project rank R up to output dim: (B, ..., R) x (B, R, D_out) -> (B, ..., D_out)
    delta = ops.einsum("b...r,bro->b...o", xa, lora_b) * scaling

    if lora_out_transpose is not None:
        delta = ops.transpose(delta, lora_out_transpose)

    return delta


def _normalize_adapter_dict(params: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Normalizes various adapter dictionary structures into {layer_key: {'a': np.ndarray, 'b': np.ndarray}}."""
    result: Dict[str, Dict[str, np.ndarray]] = {}

    # Check if already in {layer_key: {'a': ..., 'b': ...}} format
    sample_key = next(iter(params.keys())) if params else None
    if sample_key and isinstance(params[sample_key], dict) and "a" in params[sample_key] and "b" in params[sample_key]:
        for k, v in params.items():
            result[k] = {
                "a": np.asarray(v["a"], dtype=np.float32),
                "b": np.asarray(v["b"], dtype=np.float32),
            }
        return result

    # Otherwise, handle flattened keys like 'fc1/lora_kernel_a' or 'fc1/a' or 'fc1/lora/a'
    for k, v in params.items():
        arr = np.asarray(v, dtype=np.float32)
        if "/lora_kernel_a" in k:
            layer = k.replace("/lora_kernel_a", "")
            result.setdefault(layer, {})["a"] = arr
        elif "/lora_kernel_b" in k:
            layer = k.replace("/lora_kernel_b", "")
            result.setdefault(layer, {})["b"] = arr
        elif k.endswith("/a"):
            layer = k[:-2]
            result.setdefault(layer, {})["a"] = arr
        elif k.endswith("/b"):
            layer = k[:-2]
            result.setdefault(layer, {})["b"] = arr
        elif "/lora/a" in k:
            layer = k.replace("/lora/a", "")
            result.setdefault(layer, {})["a"] = arr
        elif "/lora/b" in k:
            layer = k.replace("/lora/b", "")
            result.setdefault(layer, {})["b"] = arr

    return result


def stack_lora_adapters(
    adapters: List[Optional[Dict[str, Any]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Stacks a list of individual adapter parameter dictionaries into a batched dictionary.

    Args:
        adapters: List of B adapter dictionaries. An entry can be None to indicate
                  no adaptation (base model output with zeros delta).

    Returns:
        Batched dictionary where each layer's 'a' has shape (B, D_in, R)
        and 'b' has shape (B, R, D_out).
    """
    if not adapters:
        return {}

    normalized = [
        _normalize_adapter_dict(ad) if ad is not None else None for ad in adapters
    ]

    # Find layer keys and reference shapes from the first non-None adapter
    layer_shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for ad in normalized:
        if ad is not None:
            for layer_key, weights in ad.items():
                if layer_key not in layer_shapes and "a" in weights and "b" in weights:
                    layer_shapes[layer_key] = (weights["a"].shape, weights["b"].shape)

    batched: Dict[str, Dict[str, np.ndarray]] = {}
    for layer_key, (shape_a, shape_b) in layer_shapes.items():
        list_a: List[np.ndarray] = []
        list_b: List[np.ndarray] = []

        for ad in normalized:
            if ad is not None and layer_key in ad:
                list_a.append(ad[layer_key]["a"])
                list_b.append(ad[layer_key]["b"])
            else:
                # Fill with zeros for base model (zero delta)
                list_a.append(np.zeros(shape_a, dtype=np.float32))
                list_b.append(np.zeros(shape_b, dtype=np.float32))

        batched[layer_key] = {
            "a": np.stack(list_a, axis=0),
            "b": np.stack(list_b, axis=0),
        }

    return batched


def build_lora_batch(
    adapters: Union[Dict[str, Any], List[Any]],
    routing: Optional[List[Any]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Builds a batched LoRA dictionary from a pool of adapters and a routing sequence.

    Example:
        pool = {"medical": adapter_A, "legal": adapter_B}
        routing = ["medical", "legal", "medical", None]  # None = base model
        batched_lora = build_lora_batch(pool, routing)

    Args:
        adapters: Dictionary of {name: adapter_dict} or list of adapter dicts.
        routing: Optional list of adapter keys/indices defining the order for the batch.

    Returns:
        Batched LoRA parameter dictionary of shape (B, ...).
    """
    if routing is None:
        if isinstance(adapters, list):
            return stack_lora_adapters(adapters)
        raise ValueError("routing must be provided when adapters is a dictionary.")

    adapter_list: List[Optional[Dict[str, Any]]] = []
    for item in routing:
        if item is None:
            adapter_list.append(None)
        elif isinstance(adapters, dict):
            if item not in adapters:
                raise KeyError(f"Adapter key '{item}' not found in provided adapters pool.")
            adapter_list.append(adapters[item])
        elif isinstance(adapters, (list, tuple)):
            adapter_list.append(adapters[int(item)])
        else:
            raise TypeError(f"Unsupported adapters container type: {type(adapters)}")

    return stack_lora_adapters(adapter_list)


class BatchedLora(layers.Layer):
    """Batched LoRA wrapper layer for Keras.

    Can wrap individual Dense or EinsumDense submodules in custom models.
    """

    def __init__(
        self,
        base_layer: layers.Layer,
        lora_config: Optional[LoraConfig] = None,
        x_arg: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_layer = base_layer
        self.lora_config = lora_config
        self.x_arg = x_arg

    def _find_lora_rule(self) -> Optional[LoraRule]:
        if not self.lora_config:
            return None
        layer_id = getattr(self.base_layer, "path", None) or self.base_layer.name
        for rule in self.lora_config.rules:
            if re.match(rule.pattern, layer_id) or re.match(rule.pattern, self.base_layer.name):
                return rule
        return None

    def call(self, *args, lora: Optional[Dict[str, Any]] = None, **kwargs):
        out = self.base_layer(*args, **kwargs)

        # Check explicitly passed lora or context lora
        active_lora = lora if lora is not None else _current_lora.get()
        if active_lora is None:
            return out

        layer_name = self.base_layer.name
        layer_path = getattr(self.base_layer, "path", None)

        la, lb = None, None
        if layer_path and layer_path in active_lora:
            la = active_lora[layer_path]["a"]
            lb = active_lora[layer_path]["b"]
        elif layer_name in active_lora:
            la = active_lora[layer_name]["a"]
            lb = active_lora[layer_name]["b"]

        if la is None or lb is None:
            return out

        x = args[self.x_arg]
        rule = self._find_lora_rule()
        alpha = rule.alpha if rule and hasattr(rule, "alpha") and rule.alpha else (self.lora_config.alpha if self.lora_config else 1.0)
        rank = rule.rank if rule and hasattr(rule, "rank") and rule.rank else (self.lora_config.lora_rank if self.lora_config else la.shape[-1])
        out_transpose = rule.lora_out_transpose if rule else None

        delta = compute_lora_delta(x, la, lb, alpha=alpha, rank=rank, lora_out_transpose=out_transpose)
        return out + delta


def enable_batched_lora(model: keras.Model, config: LoraConfig) -> keras.Model:
    """Hooks a Keras model to support batched multi-adapter LoRA inference.

    Traverses model layers, finds matching Dense and EinsumDense layers according
    to config.rules, and intercepts calls to add the LoRA delta when active LoRA
    parameters are provided.

    Also wraps model.__call__ so that model(x, lora=batched_lora) routes the
    batched adapters to each layer.

    Args:
        model: keras.Model instance (Functional, Subclassed, or ZeroModels).
        config: LoraConfig specifying matching rules, rank, and alpha.

    Returns:
        The hooked model instance.
    """
    target_layer_types = (layers.Dense, layers.EinsumDense)

    for layer in model._flatten_layers():
        if isinstance(layer, target_layer_types):
            layer_id = getattr(layer, "path", None) or layer.name
            matched_rule = None
            for rule in config.rules:
                if re.match(rule.pattern, layer_id) or re.match(rule.pattern, layer.name):
                    matched_rule = rule
                    break

            if matched_rule is None:
                continue

            orig_layer_call = layer.call

            def make_hooked_call(layer_ref, rule_ref, orig_call):
                def hooked_call(inputs, *args, **kwargs):
                    out = orig_call(inputs, *args, **kwargs)
                    active_lora = _current_lora.get()
                    if active_lora is None:
                        return out

                    layer_name = layer_ref.name
                    layer_path = getattr(layer_ref, "path", None)

                    la, lb = None, None
                    if layer_path and layer_path in active_lora:
                        la = active_lora[layer_path]["a"]
                        lb = active_lora[layer_path]["b"]
                    elif layer_name in active_lora:
                        la = active_lora[layer_name]["a"]
                        lb = active_lora[layer_name]["b"]

                    if la is None or lb is None:
                        return out

                    alpha = rule_ref.alpha if hasattr(rule_ref, "alpha") and rule_ref.alpha else config.alpha
                    rank = rule_ref.rank if hasattr(rule_ref, "rank") and rule_ref.rank else config.lora_rank
                    out_transpose = getattr(rule_ref, "lora_out_transpose", None)

                    delta = compute_lora_delta(
                        inputs, la, lb, alpha=alpha, rank=rank, lora_out_transpose=out_transpose
                    )
                    return out + delta

                return hooked_call

            layer.call = make_hooked_call(layer, matched_rule, orig_layer_call)

    # Hook model.call to intercept lora argument in Keras execution graph
    orig_model_call = model.call

    def hooked_model_call(inputs, *args, lora: Optional[Dict[str, Any]] = None, **kwargs):
        if lora is not None:
            normalized_lora = _normalize_adapter_dict(lora)
            # Ensure leading batch dimension
            sample_layer = next(iter(normalized_lora.values())) if normalized_lora else None
            if sample_layer is not None and sample_layer["a"].ndim == 2:
                normalized_lora = {
                    k: {
                        "a": np.expand_dims(v["a"], 0),
                        "b": np.expand_dims(v["b"], 0),
                    }
                    for k, v in normalized_lora.items()
                }

            token = _current_lora.set(normalized_lora)
            try:
                return orig_model_call(inputs, *args, **kwargs)
            finally:
                _current_lora.reset(token)

        return orig_model_call(inputs, *args, **kwargs)

    model.call = hooked_model_call

    # Also hook __call__ via dynamic subclass so instance invocation always captures lora
    orig_dunder_call = model.__class__.__call__

    def hooked_dunder_call(self, inputs, *args, lora: Optional[Dict[str, Any]] = None, **kwargs):
        if lora is not None:
            normalized_lora = _normalize_adapter_dict(lora)
            sample_layer = next(iter(normalized_lora.values())) if normalized_lora else None
            if sample_layer is not None and sample_layer["a"].ndim == 2:
                normalized_lora = {
                    k: {
                        "a": np.expand_dims(v["a"], 0),
                        "b": np.expand_dims(v["b"], 0),
                    }
                    for k, v in normalized_lora.items()
                }

            token = _current_lora.set(normalized_lora)
            try:
                return orig_dunder_call(self, inputs, *args, **kwargs)
            finally:
                _current_lora.reset(token)

        return orig_dunder_call(self, inputs, *args, **kwargs)

    cls_name = f"BatchedLora_{model.__class__.__name__}"
    model.__class__ = type(cls_name, (model.__class__,), {"__call__": hooked_dunder_call})
    return model


def enable_lora_training(
    model: keras.Model,
    config: LoraConfig,
    trainable_bias: str = "none",
) -> keras.Model:
    """Prepares a Keras model for fine-tuning a single LoRA adapter.

    Enables native Keras LoRA on target Dense and EinsumDense layers, freezes base
    weights, keeps LoRA weights trainable, and freezes non-target layers.

    Args:
        model: keras.Model to adapt.
        config: LoraConfig specifying rank, alpha, and layer rules.
        trainable_bias: "none", "all", or "lora_only".

    Returns:
        The prepared model.
    """
    target_layer_types = (layers.Dense, layers.EinsumDense)
    model.trainable = True

    for layer in model._flatten_layers():
        if layer is model:
            continue

        layer_id = getattr(layer, "path", None) or layer.name
        matched_rule = None
        for rule in config.rules:
            if re.match(rule.pattern, layer_id) or re.match(rule.pattern, layer.name):
                matched_rule = rule
                break

        if matched_rule and isinstance(layer, target_layer_types) and hasattr(layer, "enable_lora"):
            rank = getattr(matched_rule, "rank", None) or config.lora_rank
            alpha = getattr(matched_rule, "alpha", None) or config.alpha

            if not getattr(layer, "lora_enabled", False):
                layer.enable_lora(rank=rank, lora_alpha=alpha)

            layer.trainable = True

            for weight in layer.weights:
                if "lora_" in weight.name:
                    weight._trainable = True
                elif "bias" in weight.name:
                    weight._trainable = trainable_bias in ("all", "lora_only")
                else:
                    weight._trainable = False
        else:
            layer.trainable = False
            for weight in layer.weights:
                if "bias" in weight.name and trainable_bias == "all":
                    weight._trainable = True
                else:
                    weight._trainable = False

    return model


def extract_lora_params(model: keras.Model) -> Dict[str, Dict[str, np.ndarray]]:
    """Extracts LoRA weights from a fine-tuned Keras model.

    Returns:
        Dictionary mapping layer identifier to {'a': ndarray, 'b': ndarray}.
    """
    params: Dict[str, Dict[str, np.ndarray]] = {}

    for layer in model._flatten_layers():
        if getattr(layer, "lora_enabled", False):
            layer_id = getattr(layer, "path", None) or layer.name
            params[layer_id] = {
                "a": ops.convert_to_numpy(layer.lora_kernel_a),
                "b": ops.convert_to_numpy(layer.lora_kernel_b),
            }

    return params


def init_lora_params(
    model: keras.Model,
    config: LoraConfig,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Initializes a single adapter's parameters dictionary conforming to config rules.

    Args:
        model: keras.Model instance to initialize parameters for.
        config: LoraConfig specifying rules and rank.
        seed: Random seed for initialization of 'a' matrix.

    Returns:
        Dictionary mapping layer identifier to {'a': ndarray, 'b': ndarray}.
    """
    rng = np.random.RandomState(seed)
    params: Dict[str, Dict[str, np.ndarray]] = {}
    target_layer_types = (layers.Dense, layers.EinsumDense)

    for layer in model._flatten_layers():
        if isinstance(layer, target_layer_types):
            layer_id = getattr(layer, "path", None) or layer.name
            matched_rule = None
            for rule in config.rules:
                if re.match(rule.pattern, layer_id) or re.match(rule.pattern, layer.name):
                    matched_rule = rule
                    break

            if matched_rule is None:
                continue

            kernel = getattr(layer, "kernel", None)
            if kernel is None:
                continue

            kernel_shape = kernel.shape
            rank = getattr(matched_rule, "rank", None) or config.lora_rank

            # Determine input and output dims
            in_dim = kernel_shape[0]
            out_dim = kernel_shape[-1]

            shape_a = (in_dim, rank)
            shape_b = (rank, out_dim)

            # He / Normal initialization scaled by 1/sqrt(in_dim)
            lora_a = rng.randn(*shape_a).astype(np.float32) * (1.0 / np.sqrt(in_dim))
            lora_b = np.zeros(shape_b, dtype=np.float32)

            params[layer_id] = {"a": lora_a, "b": lora_b}

    return params


def save_lora_adapter(path: str, params: Dict[str, Any], config: LoraConfig) -> None:
    """Saves adapter configuration and weights to a directory.

    Saves adapter_config.json and adapter_weights.npz.
    If safetensors is installed, also saves adapter_model.safetensors.

    Args:
        path: Target directory path.
        params: LoRA parameters dictionary.
        config: LoraConfig instance.
    """
    os.makedirs(path, exist_ok=True)
    norm_params = _normalize_adapter_dict(params)

    # 1. Save adapter_config.json
    rules_data = []
    for r in config.rules:
        rule_dict = {
            "pattern": r.pattern,
            "kernel_name": r.kernel_name,
            "in_posn": r.in_posn,
            "out_posn": r.out_posn,
            "scan_posn": r.scan_posn,
            "lora_out_transpose": r.lora_out_transpose,
            "kernel_dim": r.kernel_dim,
        }
        rules_data.append(rule_dict)

    config_dict = {
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "alpha": config.alpha,
        "rules": rules_data,
    }

    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    # 2. Flatten for file persistence
    flat_weights: Dict[str, np.ndarray] = {}
    for layer_id, weights in norm_params.items():
        flat_weights[f"{layer_id}/a"] = weights["a"]
        flat_weights[f"{layer_id}/b"] = weights["b"]

    np.savez(os.path.join(path, "adapter_weights.npz"), **flat_weights)

    # 3. Optionally save safetensors
    try:
        import safetensors.numpy

        safetensors.numpy.save_file(
            flat_weights, os.path.join(path, "adapter_model.safetensors")
        )
    except ImportError:
        pass


def load_lora_adapter(path: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], LoraConfig]:
    """Loads adapter configuration and weights from a directory.

    Args:
        path: Directory containing adapter_config.json and weights.

    Returns:
        Tuple of (lora_params_dict, lora_config).
    """
    config_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config in {path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    rules_data = config_dict.pop("rules", [])
    rules = []
    for r in rules_data:
        rules.append(LoraRule(**r))

    config = LoraConfig(rules=rules, **config_dict)

    # Load weights
    safetensors_path = os.path.join(path, "adapter_model.safetensors")
    npz_path = os.path.join(path, "adapter_weights.npz")
    legacy_npz_path = os.path.join(path, "lora_adapters.npz")

    flat_weights: Dict[str, np.ndarray] = {}
    if os.path.exists(safetensors_path):
        import safetensors.numpy

        flat_weights = safetensors.numpy.load_file(safetensors_path)
    elif os.path.exists(npz_path):
        with np.load(npz_path) as data:
            flat_weights = {k: data[k] for k in data.files}
    elif os.path.exists(legacy_npz_path):
        with np.load(legacy_npz_path) as data:
            flat_weights = {k: data[k] for k in data.files}
    else:
        raise FileNotFoundError(f"No adapter weights found in {path}")

    params = _normalize_adapter_dict(flat_weights)
    return params, config
