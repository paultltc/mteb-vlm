from __future__ import annotations

import logging
from functools import partial
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    suggest_package,
)

logger = logging.getLogger(__name__)

class VBertWrapper:
    """Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py"""

    TEXT_PROMPT = "{}\nDescribe the sentence."
    IMAGE_PROMPT = "Describe the image."
    FUSED_PROMPT = "Describe the image for the following question: {}."

    def __init__(
        self,
        model_name: str = "SmolVEncoder/vbert-siglip2-slbert-30_210",
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

        self.normalize = True
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = kwargs.pop("pooling", "mean")

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
            **kwargs,
        ).to(self.device)

        model.eval()
        self.mdl = model

        self.processor = AutoProcessor.from_pretrained(model_name)
        # self.processor.tokenizer.padding_side = "left"
        # self.processor.image_processor.size["longest_edge"] = 1024
        # self.processor.image_processor.do_resize = False

    def _pooling(self, last_hidden_states, attention_mask):
        if self.pooling == "last":
            pooled_output = last_hidden_states[:, -1, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling}")
        return pooled_output

    def encode_input(self, input):
        # ff
        outputs = self.mdl(**input, return_dict=True, output_hidden_states=True)

        # pooling 
        pooled_output = self._pooling(
            outputs.hidden_states[-1], 
            input.get("attention_mask", None)
        )
        
        # normalize
        if self.normalize:
            pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=-1)

        return pooled_output

    def _build_prompt(
            self, 
            query: str, 
            images: list[Image.Image] = [],
        ):
        image_placeholders = [{"type": "image"}]* len(images)
        messages = [
            {
                "role": "user",
                "content": [
                    *image_placeholders,
                    {"type": "text", "text": query},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
    
    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        return self.get_text_embeddings(texts=sentences)

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

        # with torch.no_grad():
        #     for i in tqdm(range(0, len(texts), batch_size)):
        #         batch_texts = [
        #             self.TEXT_PROMPT + self._build_prompt(t) 
        #             for t in texts[i : i + batch_size]
        #         ]
        #         inputs = self.processor(
        #             text=batch_texts,
        #             images=None
        #             return_tensors="pt",
        #             padding=True,
        #         ).to(self.device)
        #         text_outputs = self.encode_input(inputs)
        #         all_text_embeddings.append(text_outputs.cpu())

        # all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        # return all_text_embeddings
    
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

        assert not(texts is None and images is None), "Either texts or images must be provided"

        if images is None:
            images = [[]] * len(texts)
        if texts is None:
            texts = [[]] * len(images)

        if not isinstance(texts, DataLoader):
            texts = DataLoader(texts, batch_size=batch_size)
        if not isinstance(images, DataLoader):
            images = DataLoader(images, batch_size=batch_size)
    
        all_fused_embeddings = []

        with torch.no_grad():
            for batch_txt, batch_im in tqdm(zip(texts, images)):
                if len(batch_im) == 0:
                    texts = [self._build_prompt(self.TEXT_PROMPT.format(t)) for t in batch_txt]
                    images = None
                elif len(batch_txt) == 0:
                    texts = [self._build_prompt(self.IMAGE_PROMPT, images=[i]) for i in batch_im]
                    images = [[F.to_pil_image(img.to("cpu"))] for img in batch_im]
                else:
                    texts = [self._build_prompt(self.FUSED_PROMPT.format(t), images=[i]) for (t,i) in zip(batch_txt, batch_im)]
                    images = [[F.to_pil_image(img.to("cpu"))] for img in batch_im]

                inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                outputs = self.encode_input(inputs)
                all_fused_embeddings.append(outputs.cpu())

        fused_embeddings = torch.cat(all_fused_embeddings, dim=0)
        return fused_embeddings
    
    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        return logits.softmax(dim=-1)


vbert_training_datasets = {
    # TODO: Add the training datasets here
}


euro_vbert = ModelMeta(
    loader=partial(
        VBertWrapper,
        model_name="SmolVEncoder/vbert-siglip2-eurobert_210",
    ),
    name="SmolVEncoder/vbert-siglip2-eurobert_210",
    languages=["eng-Latn"],
    revision="9209ff6086d26d226cf7e3ea262cab85b820dfd9",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=576,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/SmolVEncoder/vbert-siglip2-eurobert_210",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vbert_training_datasets,
)

# ------

vbert_sl2_slbert_30 = ModelMeta(
    loader=partial(
        VBertWrapper,
        model_name="SmolVEncoder/vbert-siglip2-slbert-30_210",
    ),
    name="SmolVEncoder/vbert-siglip2-slbert-30_210",
    languages=["eng-Latn"],
    revision="afb7c74174b1db2219ef3ccb10042d18ce311f42",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=576,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/SmolVEncoder/vbert-siglip2-slbert-30_210",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vbert_training_datasets,
)

vbert_sl2_sllm = ModelMeta(
    loader=partial(
        VBertWrapper,
        model_name="SmolVEncoder/vbert-siglip2-sllm_210",
    ),
    name="SmolVEncoder/vbert-siglip2-sllm_210",
    languages=["eng-Latn"],
    revision="6456c5250b065aad1822da1c7829975df33d0160",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/SmolVEncoder/vbert-siglip2-sllm_210",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vbert_training_datasets,
)

vlm_sl2_sllm = ModelMeta(
    loader=partial(
        VBertWrapper,
        model_name="SmolVEncoder/vlm-siglip2-sllm_210",
    ),
    name="SmolVEncoder/vlm-siglip2-sllm_210",
    languages=["eng-Latn"],
    revision="6e4e17db2934c770427d21481df66cfad51fdfe0",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/SmolVEncoder/vlm-siglip2-sllm_210",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vbert_training_datasets,
)

vbert_sl2_sllm_30 = ModelMeta(
    loader=partial(
        VBertWrapper,
        model_name="SmolVEncoder/vbert-siglip2-sllm-30_210",
    ),
    name="SmolVEncoder/vbert-siglip2-sllm-30_210",
    languages=["eng-Latn"],
    revision="6e4e17db2934c770427d21481df66cfad51fdfe0",
    release_date="2025-06-01",
    modalities=["image", "text"],
    n_parameters=413_000_000,
    memory_usage_mb=800,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/huggingface/smollm/tree/main/vision",
    public_training_data="https://huggingface.co/datasets/HuggingFaceM4/Docmatix",
    framework=["PyTorch"],
    reference="https://huggingface.co/SmolVEncoder/vbert-siglip2-sllm-30_210",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vbert_training_datasets,
)

