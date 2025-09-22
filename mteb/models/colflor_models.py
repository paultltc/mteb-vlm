from __future__ import annotations

import logging
from functools import partial

from tqdm import tqdm
import torch
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb.requires_package import (
    requires_package,
)

logger = logging.getLogger(__name__)


class ColFlorWrapper(ColPaliEngineWrapper):
    """Wrapper for ColFlor model."""

    def __init__(
        self,
        model_name: str = "ahmed-masry/ColFlor",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColFlor, ColFlorProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColFlor,
            processor_class=ColFlorProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

    def get_text_embeddings(
        self,
        texts,
        batch_size: int = 32,
        **kwargs,
    ):
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = [
                    self.processor.query_prefix
                    + t
                    + self.processor.query_augmentation_token * 10
                    for t in texts[i : i + batch_size]
                ]
                inputs = self.processor.process_texts(batch).to(self.device)
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded


colflor = ModelMeta(
    loader=partial(
        ColFlorWrapper,
        model_name="ahmed-masry/ColFlor",
        # revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="ahmed-masry/ColFlor",
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-11-03",
    modalities=["image", "text"],
    n_parameters=2_210_000_000,
    memory_usage_mb=7200,
    max_tokens=32768,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/ahmed-masry/ColFlor",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)