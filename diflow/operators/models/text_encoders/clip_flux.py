from typing import Any, Dict, List, Union

import torch
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from transformers import CLIPTextModel, CLIPTokenizer

from diflow.operators.base import Operator, has_pretrained_weights
from diflow.operators.operator_ids import CLIP_FLUX_ID
from diflow.operators.utils import (
    require_model_path,
    test_model_memory_allocation,
)


class CLIP_Flux(Operator, TextualInversionLoaderMixin):
    def setup_io(self):
        self.add_input("prompt", Union[str, List[str]])
        self.add_output("prompt_embeds", torch.Tensor)

    @property
    def id(self) -> str:
        return CLIP_FLUX_ID

    def get_encoder_config(self) -> Dict[str, str]:
        return {
            "encoder_path": "text_encoder",
            "tokenizer_path": "tokenizer",
        }

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is None:
            raise ValueError(
                f"{self.id}: model_path is required; dummy initialization is not supported"
            )
        has_pretrained_weights(model_path, self.id)

        config = self.get_encoder_config()
        text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder=config["encoder_path"],
            dtype=torch.bfloat16,
        ).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder=config["tokenizer_path"],
        )

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = kwargs["prompt"]

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, model_components["tokenizer"])

        clip_skip = kwargs.get("clip_skip", None)  # TODO: should add to the input
        num_images_per_prompt = kwargs.get(
            "num_images_per_prompt", 1
        )  # TODO: should add to the input

        text_encoder = model_components["text_encoder"]
        tokenizer = model_components["tokenizer"]
        max_length = getattr(tokenizer, "model_max_length", 77)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return {
            "prompt_embeds": prompt_embeds,
        }


if __name__ == "__main__":
    # text_encoder = CLIP_Flux()
    # model_components = text_encoder.initialize(
    #     require_model_path("FLUX.1-dev"), "cuda"
    # )
    # result = text_encoder.execute(
    #     model_components=model_components,
    #     device="cuda",
    #     prompt="A cat holding a sign that says hello world",
    # )
    test_model_memory_allocation(
        model=CLIP_Flux(),
        model_path=require_model_path("FLUX.1-dev"),
    )
