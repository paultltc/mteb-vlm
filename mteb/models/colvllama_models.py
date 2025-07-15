from __future__ import annotations

import logging
from functools import partial

import torch
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb.requires_package import (
    requires_package,
)

logger = logging.getLogger(__name__)


class ColVLlamaWrapper(ColPaliEngineWrapper):
    """Wrapper for ColVLlama model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/colvllama-sllm_210",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColVLlama, ColVLlamaProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColVLlama,
            processor_class=ColVLlamaProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )


class BiVLlamaWrapper(ColPaliEngineWrapper):
    """Wrapper for BiVLlama model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/bivllama-sllm_210",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiVLlama, BiVLlamaProcessor

        super().__init__(
            model_name=model_name,
            model_class=BiVLlama,
            processor_class=BiVLlamaProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

colvllama_sllm_210 = ModelMeta(
    loader=partial(
        ColVLlamaWrapper,
        model_name="SmolVEncoder/colvllama-sllm_210-vidore",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="SmolVEncoder/colvllama-sllm_210-vidore",
    languages=["eng-Latn"],
    revision="89fe19e05e02c811992b011c79775a0f943496a9",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/SmolVEncoder/colvllama-sllm_210-vidore",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

bivllama_sllm_210 = ModelMeta(
    loader=partial(
        BiVLlamaWrapper,
        model_name="SmolVEncoder/bivllama-sllm_210-vidore",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="SmolVEncoder/bivllama-sllm_210-vidore",
    languages=["eng-Latn"],
    revision="2c8a78f4794f834703a2081134d54e54f155a2b6",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/SmolVEncoder/bivllama-sllm_210-vidore",
    similarity_fn_name="dot",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)


# -------------------- IMAGENET ---------------------#

colvllama_sllm_210_imagenet = ModelMeta(
    loader=partial(
        ColVLlamaWrapper,
        model_name="SmolVEncoder/colvllama-sllm_210-imagenet",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="SmolVEncoder/colvllama-sllm_210-imagenet",
    languages=["eng-Latn"],
    revision="6b3d90eccedd441bfdaaf1ec8b30d4bdc5dc49fd",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,      
    max_tokens=8192,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/SmolVEncoder/colvllama-sllm_210-imagenet",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

bivllama_sllm_210_imagenet = ModelMeta(
    loader=partial(
        BiVLlamaWrapper,
        model_name="SmolVEncoder/bivllama-sllm_210-imagenet",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="SmolVEncoder/bivllama-sllm_210-imagenet",
    languages=["eng-Latn"],
    revision="d99bd5d6a8d1ff539142d02b8d71f5667ff61d5d",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,      
    max_tokens=8192,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/SmolVEncoder/bivllama-sllm_210-imagenet",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)