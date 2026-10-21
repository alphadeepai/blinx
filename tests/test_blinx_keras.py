import os
import shutil
import tempfile
import pytest
import numpy as np

# Set JAX backend for Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers, ops
from blinx import LoraConfig, LoraRule
from blinx_keras import (
    BatchedLora,
    build_lora_batch,
    compute_lora_delta,
    enable_batched_lora,
    enable_lora_training,
    extract_lora_params,
    init_lora_params,
    load_lora_adapter,
    save_lora_adapter,
    stack_lora_adapters,
)


@pytest.fixture
def dense_model():
    inp = layers.Input(shape=(8,), name="input")
    x = layers.Dense(16, name="dense_1")(inp)
    out = layers.Dense(4, name="dense_2")(x)
    return keras.Model(inp, out, name="dense_test_model")


@pytest.fixture
def einsum_model():
    inp = layers.Input(shape=(6, 8), name="input")
    out = layers.EinsumDense("abc,cd->abd", output_shape=(6, 4), name="einsum_1")(inp)
    return keras.Model(inp, out, name="einsum_test_model")


def test_compute_lora_delta():
    # Input: (2, 3, 4) -> 2 batch items, seq length 3, dim 4
    x = np.ones((2, 3, 4), dtype=np.float32)
    # LoRA A: (2, 4, 2), LoRA B: (2, 2, 5)
    la = np.ones((2, 4, 2), dtype=np.float32)
    lb = np.ones((2, 2, 5), dtype=np.float32)

    delta = compute_lora_delta(x, la, lb, alpha=4.0, rank=2)
    delta_np = ops.convert_to_numpy(delta)

    # Expected: (4 * 1) * 2 * (4/2) = 16.0
    assert delta_np.shape == (2, 3, 5)
    assert np.allclose(delta_np, 16.0)


def test_multi_adapter_batched_inference_dense(dense_model):
    config = LoraConfig(
        model_name="dense_model",
        lora_rank=2,
        alpha=2.0,
        rules=[
            LoraRule(pattern=".*dense_1.*", in_posn=[0], out_posn=[1], scan_posn=None),
            LoraRule(pattern=".*dense_2.*", in_posn=[0], out_posn=[1], scan_posn=None),
        ],
    )

    # Initialize 2 distinct adapters with different random seeds
    ad1 = init_lora_params(dense_model, config, seed=42)
    ad2 = init_lora_params(dense_model, config, seed=99)

    # Scale ad2 significantly so its outputs clearly diverge from ad1
    for k in ad2:
        ad2[k]["b"] = ad2[k]["b"] + 5.0
    for k in ad1:
        ad1[k]["b"] = ad1[k]["b"] + 1.0

    # Stack both into a batch of size 2
    batched_lora = stack_lora_adapters([ad1, ad2])
    assert batched_lora["dense_1"]["a"].shape == (2, 8, 2)
    assert batched_lora["dense_1"]["b"].shape == (2, 2, 16)

    # Enable batched LoRA on model
    enable_batched_lora(dense_model, config)

    # Generate test inputs
    np.random.seed(123)
    x0 = np.random.randn(1, 8).astype(np.float32)
    x1 = np.random.randn(1, 8).astype(np.float32)
    x_batch = np.concatenate([x0, x1], axis=0)  # (2, 8)

    # Run batched inference in ONE forward pass
    out_batched = ops.convert_to_numpy(dense_model(x_batch, lora=batched_lora))

    # Run individual forward passes
    out_single_0 = ops.convert_to_numpy(dense_model(x0, lora=ad1))
    out_single_1 = ops.convert_to_numpy(dense_model(x1, lora=ad2))

    # Assert exact numerical equivalence
    assert np.allclose(out_batched[0], out_single_0[0], atol=1e-5)
    assert np.allclose(out_batched[1], out_single_1[0], atol=1e-5)

    # Verify that different adapters produce different adaptations
    out_cross_0 = ops.convert_to_numpy(dense_model(x0, lora=ad2))
    assert not np.allclose(out_single_0[0], out_cross_0[0])


def test_multi_adapter_batched_inference_einsum(einsum_model):
    config = LoraConfig(
        model_name="einsum_model",
        lora_rank=3,
        alpha=3.0,
        rules=[
            LoraRule(pattern=".*einsum_1.*", in_posn=[0], out_posn=[1], scan_posn=None),
        ],
    )

    ad1 = {
        "einsum_1": {
            "a": np.random.randn(8, 3).astype(np.float32),
            "b": np.random.randn(3, 4).astype(np.float32),
        }
    }
    ad2 = {
        "einsum_1": {
            "a": np.random.randn(8, 3).astype(np.float32) * 2.0,
            "b": np.random.randn(3, 4).astype(np.float32) * 2.0,
        }
    }

    batched_lora = stack_lora_adapters([ad1, ad2])
    enable_batched_lora(einsum_model, config)

    x0 = np.random.randn(1, 6, 8).astype(np.float32)
    x1 = np.random.randn(1, 6, 8).astype(np.float32)
    x_batch = np.concatenate([x0, x1], axis=0)

    out_batched = ops.convert_to_numpy(einsum_model(x_batch, lora=batched_lora))
    out_single_0 = ops.convert_to_numpy(einsum_model(x0, lora=ad1))
    out_single_1 = ops.convert_to_numpy(einsum_model(x1, lora=ad2))

    assert np.allclose(out_batched[0], out_single_0[0], atol=1e-5)
    assert np.allclose(out_batched[1], out_single_1[0], atol=1e-5)


def test_build_lora_batch_routing_with_base_model(dense_model):
    config = LoraConfig(
        model_name="dense_model",
        lora_rank=2,
        alpha=2.0,
        rules=[LoraRule(pattern=".*dense.*", in_posn=[0], out_posn=[1], scan_posn=None)],
    )

    ad_a = init_lora_params(dense_model, config, seed=1)
    ad_b = init_lora_params(dense_model, config, seed=2)
    for k in ad_a:
        ad_a[k]["b"] += 1.0
    for k in ad_b:
        ad_b[k]["b"] += 2.0

    pool = {"adapter_a": ad_a, "adapter_b": ad_b}
    routing = ["adapter_a", "adapter_b", "adapter_a", None]

    batched_lora = build_lora_batch(pool, routing)
    enable_batched_lora(dense_model, config)

    x = np.random.randn(4, 8).astype(np.float32)
    out_batched = ops.convert_to_numpy(dense_model(x, lora=batched_lora))

    # Sample 0: adapter_a
    out_0 = ops.convert_to_numpy(dense_model(x[0:1], lora=ad_a))
    assert np.allclose(out_batched[0], out_0[0], atol=1e-5)

    # Sample 1: adapter_b
    out_1 = ops.convert_to_numpy(dense_model(x[1:2], lora=ad_b))
    assert np.allclose(out_batched[1], out_1[0], atol=1e-5)

    # Sample 2: adapter_a
    out_2 = ops.convert_to_numpy(dense_model(x[2:3], lora=ad_a))
    assert np.allclose(out_batched[2], out_2[0], atol=1e-5)

    # Sample 3: None -> must match unadapted base model output!
    out_base_3 = ops.convert_to_numpy(dense_model(x[3:4]))
    assert np.allclose(out_batched[3], out_base_3[0], atol=1e-5)


def test_batched_lora_layer_wrapper():
    base_dense = layers.Dense(4, name="wrapped_dense")
    base_dense.build((None, 6))

    config = LoraConfig(
        model_name="layer_test",
        lora_rank=2,
        alpha=2.0,
        rules=[LoraRule(pattern=".*wrapped.*", in_posn=[0], out_posn=[1], scan_posn=None)],
    )

    lora_layer = BatchedLora(base_dense, lora_config=config)

    x = np.ones((2, 6), dtype=np.float32)
    la = np.ones((2, 6, 2), dtype=np.float32)
    lb = np.ones((2, 2, 4), dtype=np.float32)

    batched_lora = {"wrapped_dense": {"a": la, "b": lb}}

    out_base = ops.convert_to_numpy(base_dense(x))
    out_lora = ops.convert_to_numpy(lora_layer(x, lora=batched_lora))

    # Delta: (6 * 1) * 2 * (2.0 / 2) = 12.0
    diff = out_lora - out_base
    assert np.allclose(diff, 12.0, atol=1e-5)


def test_lora_training_and_extraction():
    # Build fresh model
    inp = layers.Input(shape=(4,))
    out = layers.Dense(2, name="train_dense")(inp)
    model = keras.Model(inp, out)

    initial_kernel = ops.convert_to_numpy(model.get_layer("train_dense").kernel).copy()

    config = LoraConfig(
        model_name="train_model",
        lora_rank=2,
        alpha=2.0,
        rules=[LoraRule(pattern=".*train_dense.*", in_posn=[0], out_posn=[1], scan_posn=None)],
    )

    # Prepare for LoRA training
    enable_lora_training(model, config)

    # Base kernel must be frozen
    dense_layer = model.get_layer("train_dense")
    assert dense_layer.lora_enabled is True
    assert dense_layer.weights[0]._trainable is False
    assert dense_layer.lora_kernel_a._trainable is True
    assert dense_layer.lora_kernel_b._trainable is True

    # Compile and train
    model.compile(optimizer="adam", loss="mse")
    x_data = np.random.randn(32, 4).astype(np.float32)
    y_data = np.random.randn(32, 2).astype(np.float32)
    history = model.fit(x_data, y_data, epochs=3, verbose=0)
    assert history.history["loss"][-1] <= history.history["loss"][0]

    # Check base kernel did not change
    after_kernel = ops.convert_to_numpy(dense_layer.weights[0])
    assert np.allclose(initial_kernel, after_kernel)

    # Extract LoRA params
    params = extract_lora_params(model)
    assert "train_dense" in params
    assert params["train_dense"]["a"].shape == (4, 2)
    assert params["train_dense"]["b"].shape == (2, 2)


def test_save_and_load_adapter_roundtrip():
    config = LoraConfig(
        model_name="test_model",
        lora_rank=4,
        alpha=8.0,
        rules=[LoraRule(pattern=".*dense.*", in_posn=[0], out_posn=[1], scan_posn=None)],
    )

    params = {
        "dense_1": {
            "a": np.random.randn(8, 4).astype(np.float32),
            "b": np.random.randn(4, 16).astype(np.float32),
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_path = os.path.join(tmp_dir, "test_adapter")
        save_lora_adapter(adapter_path, params, config)

        assert os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
        assert os.path.exists(os.path.join(adapter_path, "adapter_weights.npz"))

        loaded_params, loaded_config = load_lora_adapter(adapter_path)

        assert loaded_config.model_name == config.model_name
        assert loaded_config.lora_rank == config.lora_rank
        assert loaded_config.alpha == config.alpha
        assert len(loaded_config.rules) == 1
        assert loaded_config.rules[0].pattern == ".*dense.*"

        assert np.allclose(loaded_params["dense_1"]["a"], params["dense_1"]["a"])
        assert np.allclose(loaded_params["dense_1"]["b"], params["dense_1"]["b"])


def test_zeromodels_multi_adapter_detr():
    from zeromodels.models.detr import DETRDetect, DetrConfig

    cfg = DetrConfig(
        vision_config={
            "backbone_variant": "ResNet18",
            "hidden_dim": 32,
            "num_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "dropout_rate": 0.0,
            "num_queries": 4,
            "num_classes": 5,
            "image_size": 64,
        }
    )
    model = DETRDetect(cfg)

    # Find target attention out_proj layer in encoder
    target_layers = [
        l for l in model._flatten_layers()
        if isinstance(l, layers.Dense) and "encoder_layers_0_self_attn_out_proj" in l.name
    ]
    assert len(target_layers) > 0
    target_layer = target_layers[0]

    config = LoraConfig(
        model_name="detr",
        lora_rank=4,
        alpha=4.0,
        rules=[
            LoraRule(
                pattern=".*encoder_layers_0_self_attn_out_proj.*",
                in_posn=[0],
                out_posn=[1],
                scan_posn=None,
            )
        ],
    )

    enable_batched_lora(model, config)

    in_dim = target_layer.kernel.shape[0]
    out_dim = target_layer.kernel.shape[1]

    ad1 = {
        target_layer.name: {
            "a": np.random.randn(in_dim, 4).astype(np.float32),
            "b": np.random.randn(4, out_dim).astype(np.float32),
        }
    }
    ad2 = {
        target_layer.name: {
            "a": np.random.randn(in_dim, 4).astype(np.float32) * 5.0,
            "b": np.random.randn(4, out_dim).astype(np.float32) * 5.0,
        }
    }

    batched_lora = stack_lora_adapters([ad1, ad2])

    x0 = np.random.randn(1, 800, 800, 3).astype(np.float32)
    x1 = np.random.randn(1, 800, 800, 3).astype(np.float32)
    x_batch = np.concatenate([x0, x1], axis=0)

    out_batched = model(x_batch, lora=batched_lora)
    logits_batched = ops.convert_to_numpy(out_batched["logits"])

    out_single_0 = model(x0, lora=ad1)
    logits_single_0 = ops.convert_to_numpy(out_single_0["logits"])

    out_single_1 = model(x1, lora=ad2)
    logits_single_1 = ops.convert_to_numpy(out_single_1["logits"])

    assert np.allclose(logits_batched[0], logits_single_0[0], atol=1e-4)
    assert np.allclose(logits_batched[1], logits_single_1[0], atol=1e-4)
    assert not np.allclose(logits_batched[0], logits_batched[1])


def test_llm_multi_adapter_gemma3():
    """Tests batched multi-adapter LoRA serving on a Gemma-3-1b architecture."""
    import zeromodels.models.gemma3 as g3

    cfg = g3.Gemma3Config(
        text_config={
            "vocab_size": 1000,
            "embed_dim": 64,
            "mlp_dim": 128,
            "num_layers": 2,
            "num_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 32,
        }
    )
    model = g3.Gemma3TextGenerate(cfg)

    # Target attention output projections in Gemma-3
    config = LoraConfig(
        model_name="gemma3_1b_it",
        lora_rank=4,
        alpha=4.0,
        rules=[
            LoraRule(
                pattern=".*attention/output_proj.*",
                in_posn=[0],
                out_posn=[1],
                scan_posn=None,
            ),
        ],
    )
    enable_batched_lora(model, config)

    target_layers = [
        l for l in model._flatten_layers()
        if isinstance(l, layers.Dense) and "output_proj" in l.name
    ]
    assert len(target_layers) > 0

    # Build 2 distinct adapters using unique layer paths
    ad1 = {}
    ad2 = {}
    for tl in target_layers:
        key = getattr(tl, "path", None) or tl.name
        in_dim, out_dim = tl.kernel.shape[0], tl.kernel.shape[1]
        ad1[key] = {
            "a": np.random.randn(in_dim, 4).astype(np.float32),
            "b": np.random.randn(4, out_dim).astype(np.float32),
        }
        ad2[key] = {
            "a": np.random.randn(in_dim, 4).astype(np.float32) * 4.0,
            "b": np.random.randn(4, out_dim).astype(np.float32) * 4.0,
        }

    batched_lora = stack_lora_adapters([ad1, ad2])

    # Batch of 2 token sequences
    np.random.seed(42)
    inputs_batch = {
        "input_ids": np.random.randint(0, 1000, size=(2, 8), dtype=np.int32),
        "attention_mask": np.ones((2, 8), dtype=np.int32),
    }
    inputs_0 = {
        "input_ids": inputs_batch["input_ids"][0:1],
        "attention_mask": inputs_batch["attention_mask"][0:1],
    }
    inputs_1 = {
        "input_ids": inputs_batch["input_ids"][1:2],
        "attention_mask": inputs_batch["attention_mask"][1:2],
    }

    # Run batched multi-adapter inference in ONE pass
    out_batched = model(inputs_batch, lora=batched_lora)
    logits_batched = ops.convert_to_numpy(out_batched["logits"])

    # Run individual single-adapter passes
    out_single_0 = model(inputs_0, lora=ad1)
    logits_single_0 = ops.convert_to_numpy(out_single_0["logits"])

    out_single_1 = model(inputs_1, lora=ad2)
    logits_single_1 = ops.convert_to_numpy(out_single_1["logits"])

    # Assert exact numerical equivalence between batched and individual runs
    assert np.allclose(logits_batched[0], logits_single_0[0], atol=1e-4)
    assert np.allclose(logits_batched[1], logits_single_1[0], atol=1e-4)
    assert not np.allclose(logits_batched[0], logits_batched[1])

