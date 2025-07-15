from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from mteb.models.siglip_models import SiglipModelWrapper

class Siglip2ModelWrapper(SiglipModelWrapper):
    """
    Model wrapper for the Siglip2 models.
    This class inherits from Siglip2ModelWrapper and adapts the behavior for Siglip2 models.
    It is used to differentiate between Siglip and Siglip2 models as Siglip2 models require to explicitly pass max_length=64 when processing text inputs.
    (https://huggingface.co/docs/transformers/model_doc/siglip2)
    """

    def __init__(self, model_name: str, device: str = "cuda", **kwargs: Any):
        super().__init__(model_name=model_name, device=device, **kwargs)

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64,            # Explicitly set max_length to 64 for Siglip2 models
                ).to(self.device)
                text_outputs = self.model.get_text_features(**inputs)
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings


siglip_training_datasets = {
    # WebLI https://arxiv.org/abs/2209.06794
}

siglip2_base_patch16_224 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-base-patch16-224",
    ),
    name="google/siglip2-base-patch16-224",
    languages=["eng-Latn"],
    revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=375_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-base-patch16-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_base_patch16_256 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-base-patch16-256",
    ),
    name="google/siglip2-base-patch16-256",
    languages=["eng-Latn"],
    revision="b078df89e446d623010d890864d4207fe6399f61",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=375_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-base-patch16-256",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_base_patch16_384 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-base-patch16-384",
    ),
    name="google/siglip2-base-patch16-384",
    languages=["eng-Latn"],
    revision="f775b65a79762255128c981547af89addcfe0f88",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=376_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-base-patch16-384",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_base_patch16_512 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-base-patch16-512",
    ),
    name="google/siglip2-base-patch16-512",
    languages=["eng-Latn"],
    revision="a89f5c5093f902bf39d3cd4d81d2c09867f0724b",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=376_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-base-patch16-512",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_base_patch16_224 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-base-patch16-224",
    ),
    name="google/siglip2-base-patch16-224",
    languages=["eng-Latn"],
    revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=203_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-base-patch16-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_large_patch16_256 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-large-patch16-256",
    ),
    name="google/siglip2-large-patch16-256",
    languages=["eng-Latn"],
    revision="d0da9f876e7d66b4e250cd2450c3ba2ce735e447",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=882_000_000,
    memory_usage_mb=2488,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-large-patch16-256",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_large_patch16_384 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-large-patch16-384",
    ),
    name="google/siglip2-large-patch16-384",
    languages=["eng-Latn"],
    revision="ce005573a40965dfd21fd937fbdeeebf2439fc35",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=882_000_000,
    memory_usage_mb=2489,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-large-patch16-384",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_large_patch16_512 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-large-patch16-512",
    ),
    name="google/siglip2-large-patch16-512",
    languages=["eng-Latn"],
    revision="49488218e80259885f3be61d7a9455faf833b7a8",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=882_000_000,
    memory_usage_mb=2489,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-large-patch16-512",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_so400m_patch16_384 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-so400m-patch16-384",
    ),
    name="google/siglip2-so400m-patch16-384",
    languages=["eng-Latn"],
    revision="dd658faac399427308559e2c3ac1e99cbe43845d",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1_014_000_000,
    memory_usage_mb=3349,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-so400m-patch16-384",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip2_so400m_patch16_512 = ModelMeta(
    loader=partial(
        Siglip2ModelWrapper,
        model_name="google/siglip2-so400m-patch16-512",
    ),
    name="google/siglip2-so400m-patch16-512",
    languages=["eng-Latn"],
    revision="ceea1cba8130d8271436da4828633198c176a775",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1_014_000_000,
    memory_usage_mb=3349,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip2-so400m-patch16-512",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)