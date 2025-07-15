from __future__ import annotations

import logging
from functools import partial
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    suggest_package,
)

logger = logging.getLogger(__name__)

class SmolVLMWrapper:
    """Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py"""

    TEXT_PROMPT = "{}\nDescribe above sentence in one word: "
    IMAGE_PROMPT = "Describe above image in one word: "
    FUSED_PROMPT = "Describe above image for the following question: {}"

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        device: str = None,
        **kwargs,
    ):
        requires_image_dependencies()
        if suggest_package(
            self,
            "flash_attn",
            model_name,
            "pip install flash-attn --no-build-isolation",
        ):
            import flash_attn  # noqa

        # requires_package(self, "peft", model_name, "pip install 'mteb[peft]'")
        # from peft import LoraConfig, PeftModel  # noqa

        self.pooling = "last"
        self.normalize = True
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs,
        ).to(self.device)

        model.eval()
        self.mdl = model

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        return self.get_text_embeddings(texts=sentences)

    def encode_input(self, input):
        hidden_states = self.mdl(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input["attention_mask"])
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == "last":
            reps = last_hidden_state[:, -1, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            reps = sum_embeddings / sum_mask
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _build_prompt(self, query: str, images: list[Image.Image] = []):
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": query},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return prompt

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        return self.get_fused_embeddings(
            texts=None,
            images=images,
            task_name=task_name,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        return self.get_fused_embeddings(
            texts=texts,
            images=None,
            task_name=task_name,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        kwargs.update(
            task_name=task_name, prompt_type=prompt_type, batch_size=batch_size
        )

        # text_embeddings is not None and image_embeddings is not None
        if texts is None:
            length = len(images.dataset) if isinstance(images, DataLoader) else len(images)
            texts = [self.IMAGE_PROMPT] * length
        elif texts is not None and images is None:
            texts = [self.TEXT_PROMPT.format(t) for t in texts]
        else:
            texts = [self.FUSED_PROMPT.format(t) for t in texts]

        if images is None:
            images = [[]] * len(texts)

        if not isinstance(texts, DataLoader):
            texts = DataLoader(texts, batch_size=batch_size)

        if not isinstance(images, DataLoader):
            images = DataLoader(images, batch_size=batch_size)

        iterator = zip(texts, images)
    
        all_fused_embeddings = []

        with torch.no_grad():
            for batch_txt, batch_im in iterator:
                if len(batch_im) == 0:
                    texts = [self._build_prompt(t) for t in batch_txt]
                    images = None
                else:
                    texts = [self._build_prompt(t, [i]) for (t,i) in zip(batch_txt, batch_im)]
                    images = [[F.to_pil_image(i.to("cpu")).convert("RGB")] for i in batch_im]

                inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                text_outputs = self.encode_input(inputs)
                all_fused_embeddings.append(text_outputs.cpu())


        fused_embeddings = torch.cat(all_fused_embeddings, dim=0)
        return fused_embeddings


smolvlm_training_datasets = {
    # TODO: Add the training datasets here
}

smolvlm_256m = ModelMeta(
    loader=partial(
        SmolVLMWrapper,
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
    ),
    name="HuggingFaceTB/SmolVLM-256M-Instruct",
    languages=["eng-Latn"],
    revision="7e3e67edbbed1bf9888184d9df282b700a323964",
    release_date="2025-01-20",
    modalities=["image", "text"],
    n_parameters=256_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=576,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=smolvlm_training_datasets,
)

smolvlm_500m = ModelMeta(
    loader=partial(
        SmolVLMWrapper,
        model_name="HuggingFaceTB/SmolVLM-500M-Instruct",
    ),
    name="HuggingFaceTB/SmolVLM-500M-Instruct",
    languages=["eng-Latn"],
    revision="a7da5b986cb59b408707209984f360a5f4ad7e47",
    release_date="2025-01-20",
    modalities=["image", "text"],
    n_parameters=507_000_000,
    memory_usage_mb=1200,
    max_tokens=8192,
    embed_dim=960,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=smolvlm_training_datasets,
)

smolvlm2_256m = ModelMeta(
    loader=partial(
        SmolVLMWrapper,
        model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    ),
    name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    languages=["eng-Latn"],
    revision="067788b187b95ebe7b2e040b3e4299e342e5b8fd",
    release_date="2025-02-11",
    modalities=["image", "text"],
    n_parameters=256_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=576,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=smolvlm_training_datasets,
)

smolvlm2_500m = ModelMeta(
    loader=partial(
        SmolVLMWrapper,
        model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    ),
    name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    languages=["eng-Latn"],
    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    release_date="2025-02-11",
    modalities=["image", "text"],
    n_parameters=507_000_000,
    memory_usage_mb=1200,
    max_tokens=8192,
    embed_dim=960,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=smolvlm_training_datasets,
)


smolvlm2_2b = ModelMeta(
    loader=partial(
        SmolVLMWrapper,
        model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    ),
    name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    languages=["eng-Latn"],
    revision="482adb537c021c86670beed01cd58990d01e72e4",
    release_date="2025-02-11",
    modalities=["image", "text"],
    n_parameters=2_250_000_000,
    memory_usage_mb=5000,
    max_tokens=8192,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=smolvlm_training_datasets,
)

smolvlm2_2b_base = ModelMeta(
    loader=partial(
        SmolVLMWrapper,
        model_name="HuggingFaceTB/SmolVLM2-2.2B-Base",
    ),
    name="HuggingFaceTB/SmolVLM2-2.2B-Base",
    languages=["eng-Latn"],
    revision="482adb537c021c86670beed01cd58990d01e72e4",
    release_date="2025-02-11",
    modalities=["image", "text"],
    n_parameters=2_250_000_000,
    memory_usage_mb=5000,
    max_tokens=8192,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Base",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=smolvlm_training_datasets,
)
