from __future__ import annotations

import logging
from functools import partial
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
    suggest_package,
)

from .qwen2_5_vl_embed.qwen2_5_vl_embed import Qwen2_5ForEmbedding

logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "document"]


class MoCaWrapper:
    """Adapted from https://github.com/TIGER-AI-Lab/MoCa/blob/main/src/model.py"""

    def __init__(
        self,
        model_name: str = "moca-embed/MoCa-Qwen25VL-3B",
        processor_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

        requires_package(self, "peft", model_name, "pip install 'mteb[peft]'")
        from peft import LoraConfig, PeftModel  # noqa

        self.pooling = "last"
        self.normalize = True
        self.temperature = 1.0
        self.hidden_size = 4096
        self.device = device

        # Loading the base model
        self.processor = AutoProcessor.from_pretrained(processor_name)
        config = AutoConfig.from_pretrained(model_name)
        model = Qwen2_5ForEmbedding.from_pretrained(
            model_name, config=config, 
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            bidirectional=True,
        )
        model.eval()
        model.eval()
        model.to(device)
        self.mdl = model

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
        # hidden_states = self.mdl(**input, return_dict=True, output_hidden_states=True)
        # hidden_states = hidden_states.hidden_states[-1]
        # pooled_output = self._pooling(hidden_states, input["attention_mask"])
        # return pooled_output
        return self.mdl(**input, return_dict=True, output_hidden_states=True)

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == "last":
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    # reference: https://github.com/TIGER-AI-Lab/MoCa/blob/main/src/collator.py
    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F

        text = "<|vision_start|><|image_pad|><|vision_end|>Represent the given image."
        all_image_embeddings = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch:
                        inputs = self.processor(
                            text=text,
                            images=[F.to_pil_image(b.to("cpu"))],
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_grid_thw.append(inputs["image_grid_thw"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_grid_thw = torch.cat(image_grid_thw, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }

                    image_outputs = self.encode_input(inputs)
                    all_image_embeddings.append(image_outputs.cpu().to(torch.float32))

        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch_images:
                        inputs = self.processor(
                            text=text,
                            images=[b],
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_grid_thw.append(inputs["image_grid_thw"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_grid_thw = torch.cat(image_grid_thw, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }

                    image_outputs = self.encode_input(inputs)
                    all_image_embeddings.append(image_outputs.cpu().to(torch.float32))

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

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
                input_ids = []
                batch_texts = texts[i : i + batch_size]
                for text in batch_texts:
                    inputs = self.processor(
                        text=text,
                        images=None,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))

                input_ids = torch._C._nn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                ).squeeze(2)
                attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                text_outputs = self.encode_input(inputs)
                all_text_embeddings.append(text_outputs.cpu().to(torch.float32))

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

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

        text_embeddings = None
        image_embeddings = None
        kwargs.update(
            task_name=task_name, prompt_type=prompt_type, batch_size=batch_size
        )

        if texts is not None and images is None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)
            return text_embeddings

        if images is not None and texts is None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)
            return image_embeddings

        # text_embeddings is not None and image_embeddings is not None
        texts = iter(texts)
        all_fused_embeddings = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in images:
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch:
                        text = next(texts)
                        inputs = self.processor(
                            text=f"<|vision_start|><|image_pad|><|vision_end|>Represent the given image with the following text: {text}",
                            images=[F.to_pil_image(b.to("cpu"))],
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_grid_thw.append(inputs["image_grid_thw"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_grid_thw = torch.cat(image_grid_thw, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }

                    outputs = self.encode_input(inputs)
                    all_fused_embeddings.append(outputs.cpu().to(torch.float32))
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch_images:
                        text = next(texts)
                        inputs = self.processor(
                            text=f"<|image_1|> Represent the given image with the following question: {text}",
                            images=[b],
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_grid_thw.append(inputs["image_grid_thw"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_grid_thw = torch.cat(image_grid_thw, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }

                    outputs = self.encode_input(inputs)
                    all_fused_embeddings.append(outputs.cpu().to(torch.float32))

        fused_embeddings = torch.cat(all_fused_embeddings, dim=0)
        return fused_embeddings


MoCa_training_datasets = {
    # MMEB-train
}

MoCa_3b = ModelMeta(
    loader=partial(
        MoCaWrapper,
        model_name="moca-embed/MoCa-Qwen25VL-3B",
        processor_name="Qwen/Qwen2.5-VL-3B-Instruct"
    ),
    name="moca-embed/MoCa-Qwen25VL-3B",
    languages=["eng-Latn"],
    revision="89b0e5a9245a95d6df5f54e7ebe1588bc7c7d926",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=3_750_000_000,
    memory_usage_mb=7909,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/MoCa",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    framework=["PyTorch"],
    reference="https://huggingface.co/moca-embed/MoCa-Qwen25VL-3B",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=MoCa_training_datasets,
)

MoCa_8b = ModelMeta(
    loader=partial(
        MoCaWrapper,
        model_name="moca-embed/MoCa-Qwen25VL-7B",
        processor_name="Qwen/Qwen2.5-VL-7B-Instruct"
    ),
    name="moca-embed/MoCa-Qwen25VL-7B",
    languages=["eng-Latn"],
    revision="611a540331631a4e714947e8fa93d14e1ae277d2",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=8_290_000_000,
    memory_usage_mb=15800,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/MoCa",
    public_training_data="https://huggingface.co/moca-embed/MoCa-Qwen25VL-7B",
    framework=["PyTorch"],
    reference="https://huggingface.co/moca-embed/MoCa-Qwen25VL-7B",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=MoCa_training_datasets,
)
