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


class ColEuroVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2 model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/colvbert-eurobert_210-vidore",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColVBert, ColVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColVBert,
            processor_class=ColVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )


class BiEuroVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for BiVBert model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/bivbert-eurobert_210-vidore",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiVBert, BiVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=BiVBert,
            processor_class=BiVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

colvbert_eurobert_210 = ModelMeta(
    loader=partial(
        ColEuroVBertWrapper,
        model_name="SmolVEncoder/colvbert-eurobert_210-vidore",
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="SmolVEncoder/colvbert-eurobert_210-vidore",
    languages=["eng-Latn"],
    revision="05628b2a88243682d5e65ebeaf261385aeed6f34",
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
    reference="https://huggingface.co/SmolVEncoder/colvbert-eurobert_210-vidore",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)