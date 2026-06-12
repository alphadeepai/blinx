<p align="center">
  <img src="logo.png" width="400" alt="blinx logo" />
</p>

# blinx: Batched LoRA in JAX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-Powered-blue.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-linen-blueviolet.svg)](https://github.com/google/flax)

**blinx** is a lightweight, high-performance JAX/Flax library designed for **Batched LoRA (Low-Rank Adaptation)**. It enables serving multiple independent, fine-tuned adapters simultaneously within the same batch during inference with minimal overhead, making it ideal for high-throughput Multi-LoRA model serving.

---

## Key Features

- **Dynamic Adapter Batching**: Slice and map different LoRA adapters to different sequences in the same input batch.
- **Universal Flax Wrapper**: Wrap around any linear operations like `nn.Dense` or custom `Einsum` submodules, or use it directly on raw parameters.
- **Support for Scanned Layers**: Out-of-the-box support for models with scanned/stacked parameters along a scan dimension (e.g., layers grouped via `jax.lax.scan`).
- **Zero-overhead Inference**: Uses JAX's native `dot_general` batching dimensions to perform low-rank operations efficiently.
- **Simple Disk Serialization**: Serializes adapter checkpoints and configurations using `orbax` and `flax.serialization`.

---

## Inspiration

**blinx** is inspired by the design of [lorax](https://github.com/predibase/lorax) by Predibase, bringing flexible, multi-adapter inference capabilities to JAX-based architectures.

---

## How It Works

Adding LoRA capabilities to your JAX model using `blinx` consists of three main steps:
1. **Configure rules** using `LoraRule` and `LoraConfig`.
2. **Initialize** the adapter parameters.
3. **Wrap** your model layers using `BatchedLora`.

### 1. Define LoRA Rules and Config

To adapt a model, you specify rules mapping to layers using regex patterns on their Flax module path:

```python
from blinx import LoraConfig, LoraConfigManager, LoraRule

# Configure rules matching target layers in your architecture
lora_config = LoraConfig(
    model_name="my_model",
    lora_rank=16,
    alpha=16,
    rules=[
        # Rule for standard MLP Block layers
        LoraRule(
            pattern=".*MlpBlock.*",
            kernel_name="kernel",
            in_posn=[1],
            out_posn=[2],
            scan_posn=0
        ),
        # Rule for custom Einsum attention projections
        LoraRule(
            pattern=".*q_einsum.*",
            kernel_name="w",
            in_posn=[2],
            out_posn=[1, 3],
            scan_posn=0
        ),
    ]
)

# Register configuration globally
LoraConfigManager.configs.append(lora_config)
```

### 2. Initialize Adapter Parameters

Using the existing base model parameter tree, initialize matching LoRA parameters:

```python
from blinx import init_lora_params

# Initialize matching LoRA weights (will return a dictionary tree of weights)
# conforming to your LoraConfig rules
lora_params = init_lora_params(base_params, lora_config)
```

> [!NOTE]
> `init_lora_params` creates a single adapter's weight tree. To serve multiple adapters in a batch, you can stack the individual adapter dictionaries along a batch axis using `jax.tree.map(lambda *x: jnp.stack(x), ...)` to produce a batched `lora` parameters tree of shape `(batch_size, ...)`.

### 3. Integrate into Flax linen Modules

There are two primary ways to apply `BatchedLora` within a Flax module:

#### Option A: Wrapping Submodules (Implicit Wrapper)
You can directly wrap any `nn.Module` subclass (e.g., standard `nn.Dense` or custom `Einsum` layers). `BatchedLora` automatically intercepts the module calls, retrieves the relevant LoRA weights, computes the adapter output, and adds it:

```python
from blinx import BatchedLora
import flax.linen as nn

class MyAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, lora):
        # Wrap the projection submodule.
        # x_arg=1 indicates that 'x' is the 1st positional argument passed to q_einsum
        q = BatchedLora(self.q_einsum, x_arg=1)("BTD,NDH->BTNH", x, lora=lora)
        return q
```

#### Option B: Manual Computation (Explicit Wrapper)
For layers where weights are defined directly as parameters using `self.param` (rather than submodules), you can query rules and apply the adapter delta manually:

```python
from blinx import BatchedLora
import flax.linen as nn
import jax.numpy as jnp

class CustomMLP(nn.Module):
    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x, lora):
        w_gating = self.param("gating_einsum", init_fn, (2, self.features, self.hidden_dim))
        
        # Instantiate BatchedLora on self
        bl = BatchedLora(self)
        
        # Query rules and retrieve lora parameter matrices
        lora_rule = bl.find_lora_rule(suffix="/gating_einsum")
        lora_a, lora_b = bl.find_lora_params(lora, lora_rule)

        # Apply the adapter delta manually and combine with the base layer output
        ff_gate = jnp.dot(x, w_gating[0]) + bl.compute_lora_delta(
            lora_rule, x, lora_a, lora_b.transpose(2, 0, 1, 3)[0]
        )
        return nn.gelu(ff_gate)
```

---

## Saving and Loading Adapters

blinx provides built-in utilities using `orbax` to serialize adapters to disk:

```python
from blinx import save_lora_adapter, load_lora_adapter

# Save adapter weights and configuration
save_lora_adapter("path/to/adapter_directory", lora_params, lora_config)

# Restore adapter weights and configuration
loaded_params, loaded_config = load_lora_adapter("path/to/adapter_directory")
```

---

## Verified Compatible Models

| Model | Repository | Description |
| :--- | :--- | :--- |
| **PaliGemma** / **PaliGemma 2** | [google/big_vision](https://github.com/google-research/big-vision) | Verified compatibility with Vision-Language Models (VLM) using JAX/Flax implementation. |

---

## Acknowledgements

This project was built with the support of the [Google TPU Research Cloud (TRC)](https://sites.research.google/trc/).

<p align="left">
  <img src="logotrc.png" width="200" alt="Google TPU Research Cloud Logo" />
</p>

---

## License

This project is licensed under the **MIT License**. See below for details.

### Copyright

Copyright (c) 2026 AlphaDeep Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including its software documentation files
(the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
