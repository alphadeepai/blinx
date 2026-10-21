# blinx_keras: Batched Multi-LoRA in Keras 3

`blinx_keras` is the dedicated Keras 3 companion module for **blinx**. It brings **Batched LoRA multi-adapter serving** to Keras models, enabling you to serve multiple independent fine-tuned adapters simultaneously within the same batch during inference with zero per-request overhead.

---

## Key Features

- **Dynamic Multi-Adapter Batching**: Route different LoRA adapters to different samples in the same batch in a single forward pass.
- **Unified Configuration**: Reuses the exact same `LoraRule` and `LoraConfig` structures from `blinx`.
- **Targeted Layer Support**: Optimized for `keras.layers.Dense` and `keras.layers.EinsumDense`.
- **Universal Tensor Contractions**: Uses `keras.ops.einsum` to compute low-rank deltas across arbitrary tensor ranks (2D, 3D sequence inputs, 4D vision inputs).
- **Simple Disk Serialization**: Serializes adapters to `.npz` and `.safetensors` alongside `adapter_config.json`.

---

## Installation & Requirements

Ensure Keras 3 and your preferred backend (JAX, PyTorch, or TensorFlow) are installed:

```bash
pip install keras "jax[cpu]"
```

Set your backend environment variable (e.g., JAX):
```bash
export KERAS_BACKEND=jax
```

---

## How It Works: Multi-Adapter Serving

Suppose you have an API serving 4 requests in a single batch:
- Sample 0 uses `Adapter A`
- Sample 1 uses `Adapter B`
- Sample 2 uses `Adapter A`
- Sample 3 uses the unadapted **Base Model**

```python
import numpy as np
from blinx import LoraRule, LoraConfig
from blinx_keras import (
    load_lora_adapter,
    build_lora_batch,
    enable_batched_lora,
)

# 1. Load adapters from disk
adapter_A, config_A = load_lora_adapter("adapters/adapter_a")
adapter_B, config_B = load_lora_adapter("adapters/adapter_b")

# 2. Define the adapter pool and batch routing sequence
adapter_pool = {
    "adapter_a": adapter_A,
    "adapter_b": adapter_B,
}
routing = ["adapter_a", "adapter_b", "adapter_a", None]  # None = base model output

# 3. Build the batched LoRA parameter dictionary (leading dimension B = 4)
batched_lora = build_lora_batch(adapter_pool, routing)

# 4. Enable batched LoRA on your Keras model once
enable_batched_lora(model, config_A)

# 5. Execute inference in ONE forward pass
# x_batch shape: (4, ...)
outputs = model(x_batch, lora=batched_lora)
```

Inside each target `Dense` or `EinsumDense` layer, `blinx_keras` computes:
$$\Delta = (X \cdot A \cdot B) \times \frac{\alpha}{R}$$
in parallel across the batch, adapting each sample $i$ with adapter $i$ without duplicating the base model weights.

---

## Training and Saving Adapters

You can also use `blinx_keras` to train individual adapters with standard Keras `model.fit()`:

```python
import keras
from keras import layers
from blinx import LoraRule, LoraConfig
from blinx_keras import (
    enable_lora_training,
    extract_lora_params,
    save_lora_adapter,
)

# 1. Define model and LoRA configuration
config = LoraConfig(
    model_name="my_model",
    lora_rank=8,
    alpha=16.0,
    rules=[
        LoraRule(pattern=".*dense.*", in_posn=[0], out_posn=[1], scan_posn=None),
    ],
)

# 2. Inject LoRA for fine-tuning
# Freezes base weights (_kernel) and non-target layers while activating LoRA parameters
enable_lora_training(model, config)

# 3. Train with standard Keras optimizer and fit
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train, epochs=3)

# 4. Extract and save adapter weights
params = extract_lora_params(model)
save_lora_adapter("adapters/my_fine_tuned_adapter", params, config)
```

---

## Advanced Example: Batched LoRA on Attention (EinsumDense)

`blinx_keras` seamlessly adapts attention projections and custom layers:

```python
import keras
from keras import layers
from blinx import LoraRule, LoraConfig
from blinx_keras import enable_batched_lora, stack_lora_adapters

# Define rules for attention projections
config = LoraConfig(
    model_name="transformer",
    lora_rank=8,
    alpha=16.0,
    rules=[
        LoraRule(pattern=".*q_proj.*", in_posn=[0], out_posn=[1], scan_posn=None),
        LoraRule(pattern=".*v_proj.*", in_posn=[0], out_posn=[1], scan_posn=None),
    ],
)

# Hook model for batched LoRA
enable_batched_lora(model, config)

# Stack two distinct adapters into a batch of size 2
batched_lora = stack_lora_adapters([adapter_1, adapter_2])

# Run batched inference on sequences simultaneously
predictions = model(seq_batch, lora=batched_lora)
```

---

## Running Tests

All automated unit and integration tests can be run using `pytest`:

```bash
PYTHONPATH=. KERAS_BACKEND=jax pytest tests/test_blinx_keras.py -v
```
