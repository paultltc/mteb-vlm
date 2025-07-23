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


class ColModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for ColModernVBert model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/colvbert-modernbert_base-vidore",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColModernVBert,
            processor_class=ColModernVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])


class BiModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for BiVBert model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/bivbert-slbert_210",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiModernVBert, BiModernVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=BiModernVBert,
            processor_class=BiModernVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])

colvbert_modernvbert_base = ModelMeta(
    loader=partial(
        ColModernVBertWrapper,
        model_name="SmolVEncoder/colvbert-modernbert_base-vidore",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="SmolVEncoder/colvbert-modernbert_base-vidore",
    languages=["eng-Latn"],
    revision="c71ee9a431b74e87c138460f38be01248984d2f4",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=252_000_000,
    memory_usage_mb=480,
    max_tokens=8192,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/SmolVEncoder/colvbert-modernbert_base-vidore",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)