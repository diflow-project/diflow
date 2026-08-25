from typing import Any, Dict, List, Union

import torch
from transformers import T5EncoderModel, T5TokenizerFast

from diflow.operators.base import Operator, has_pretrained_weights
from diflow.operators.operator_ids import T5_FLUX_ID
from diflow.operators.utils import (
    require_model_path,
    test_model_memory_allocation,
)


class T5_Flux(Operator):
    def setup_io(self):
        self.add_input("prompt", Union[str, List[str]])
        # [batch_size, 512, 4096]
        self.add_output("prompt_embeds", torch.Tensor, [1, 512, 4096])

    @property
    def id(self) -> str:
        return T5_FLUX_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is None:
            raise ValueError(
                f"{self.id}: model_path is required; dummy initialization is not supported"
            )
        has_pretrained_weights(model_path, self.id)

        text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            dtype=torch.bfloat16,
        ).to(device)
        tokenizer = T5TokenizerFast.from_pretrained(
            model_path,
            subfolder="tokenizer_2",
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

        text_encoder = model_components["text_encoder"]
        tokenizer = model_components["tokenizer"]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        num_images_per_prompt = kwargs.get(
            "num_images_per_prompt", 1
        )  # TODO: should add to input
        max_sequence_length = kwargs.get(
            "max_sequence_length", 512
        )  # TODO: should add to input

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
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
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return {"prompt_embeds": prompt_embeds}


if __name__ == "__main__":
    # text_encoder = T5_Flux()
    # model_components = text_encoder.initialize(
    #     require_model_path("FLUX.1-dev"), "cuda"
    # )
    # result = text_encoder.execute(
    #     model_components=model_components,
    #     device="cuda",
    #     prompt="A cat holding a sign that says hello world",
    # )
    test_model_memory_allocation(
        model=T5_Flux(),
        model_path=require_model_path("FLUX.1-dev"),
    )
