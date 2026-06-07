import functools
import re
import os
import orbax.checkpoint
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass
from flax.training import orbax_utils
import flax.serialization
from typing import Optional, List, Tuple


@dataclass
class LoraRule:
    pattern: str
    in_posn: List[int]
    out_posn: List[int]
    scan_posn: Optional[int]
    lora_out_transpose: Optional[Tuple[int, ...]] = None
    kernel_name: str = "kernel"
    kernel_dim: int = -1


@dataclass
class LoraConfig:
    model_name: str
    lora_rank: int
    alpha: int
    rules: List[LoraRule]


class LoraConfigManager:
    # TODO better handling if multiple models
    configs: List[LoraConfig] = []


@dataclass
class BatchedLora:
    """
    Batched LoRA in Jax - main wrapper function.

    A universal LoRA wrapper that handles arbitrary input shapes.
    This module has no params, it expects a batch of LoRA matrices. It can wrap around any linear op like Dense,
    Einsum etc. and it can also be used directly to manipulate dot products.

    TODO: NNX compatibility

    @author: nmilosev@alphadeep.ai
    """
    base_layer: nn.Module
    x_arg: int = 0  # carry (input) index

    def find_lora_rule(self, suffix: str = ""):
        for rule in self.lora_config.rules:
            if re.match(rule.pattern, "/".join(self.base_layer.path) + suffix) and rule.kernel_name in self.base_layer.variables["params"].keys():
                return rule
        raise Exception(
            f"blinx lora called but without compatible rule for layer path: {self.base_layer.path}")

    def find_lora_params(self, lora, lora_rule):
        key_a = self.base_layer.path + (lora_rule.kernel_name, "lora", "a")
        key_b = self.base_layer.path + (lora_rule.kernel_name, "lora", "b")

        lora_a = get_by_path(lora, key_a)
        lora_b = get_by_path(lora, key_b)

        if lora_a is None or lora_b is None:
            raise Exception(
                f"blinx lora called but without lora weights: {self.base_layer.path} {key_a=} {key_b=}")

        return lora_a, lora_b

    def compute_lora_delta(self, lora_rule, x, lora_a, lora_b):
        contract_dims = ((lora_rule.kernel_dim % x.ndim,), (1,))
        batch_dims = ((0,), (0,))

        x_a = jax.lax.dot_general(
            x, lora_a,
            dimension_numbers=(contract_dims, batch_dims)
        )

        contract_dims_b = ((x_a.ndim - 1,), (1,))
        batch_dims_b = ((0,), (0,))

        delta = jax.lax.dot_general(
            x_a, lora_b,
            dimension_numbers=(contract_dims_b, batch_dims_b)
        )
        if lora_rule.lora_out_transpose is not None:
            delta = jnp.transpose(delta, lora_rule.lora_out_transpose)
        scaling = self.lora_config.alpha / self.lora_config.lora_rank

        return delta * scaling

    def __call__(self, *args, lora, **kwargs):
        out = self.base_layer(*args, **kwargs)

        lora_rule = self.find_lora_rule()
        lora_a, lora_b = self.find_lora_params(lora, lora_rule)
        x = args[self.x_arg]

        return out + self.compute_lora_delta(lora_rule, x, lora_a, lora_b)

    @property
    def lora_config(self):
        return LoraConfigManager.configs[0]


def init_lora_params(params, lora_config: LoraConfig):
    def init_lora_param(name, param, lora_config,
                        rng=jax.random.PRNGKey(0)):
        name = "/".join([n.key for n in name])
        for rule in lora_config.rules:
            if re.match(rule.pattern, name) and name.endswith(rule.kernel_name):
                in_dims = jnp.array(param.shape)[jnp.array(rule.in_posn)]
                out_dims = jnp.array(param.shape)[jnp.array(rule.out_posn)]
                if rule.scan_posn is not None:
                    # scan, B, S, I, R -> B, S, R, O
                    scan_dim = param.shape[rule.scan_posn]
                    shape_a = (scan_dim, *in_dims, lora_config.lora_rank)
                    shape_b = (scan_dim, lora_config.lora_rank, *out_dims)
                else:
                    # no scan, normal B, I, R -> B, R, O
                    shape_a = (*in_dims, lora_config.lora_rank)
                    shape_b = (lora_config.lora_rank, *out_dims)

                key_a, rng_key = jax.random.split(rng)
                lora_a = jax.random.normal(
                    key_a, shape_a) * (1.0 / jnp.sqrt(in_dims[0]))
                lora_b = jnp.zeros(shape_b)

                return {"lora": {"a": lora_a, "b": lora_b}}
    lora_params = jax.tree.map_with_path(functools.partial(
        init_lora_param, lora_config=lora_config), params)
    return lora_params


def save_lora_adapter(path: str, params: dict, config: LoraConfig):
    ckpt_path = os.path.join(path, "checkpoint")
    config_dict = flax.serialization.to_state_dict(config)

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)

    checkpointer.save(
        os.path.abspath(ckpt_path),
        params,
        save_args=save_args,
        force=True
    )

    print(f"blinx adapter saved to: {path}")


def load_lora_adapter(path: str) -> Tuple[dict, LoraConfig]:
    config_path = os.path.join(path, "adapter_config.json")
    ckpt_path = os.path.join(path, "checkpoint")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config in {path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    rules_data = config_dict.pop("rules")
    rules = [LoraRule(**r) for r in rules_data.values()]
    config = LoraConfig(rules=rules, **config_dict)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    lora_params = checkpointer.restore(os.path.abspath(ckpt_path))

    return lora_params, config


def get_by_path(tree, path):
    subtree = tree
    for key in path:
        subtree = subtree[key]
    return subtree
