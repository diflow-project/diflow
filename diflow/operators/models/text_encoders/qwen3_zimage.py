from typing import Any, Dict, List, Union

import torch
from transformers import AutoModel, AutoTokenizer

from diflow.operators.base import Operator, require_pretrained_weights
from diflow.operators.operator_ids import QWEN3_ZIMAGE_ID


class Qwen3_ZImage(Operator):
    """Encode Z-Image prompts and retain their variable-length token mask."""

    def setup_io(self):
        self.add_input("prompt", Union[str, List[str]])
        self.add_output("prompt_embeds", torch.Tensor)
        self.add_output("encoder_attention_mask", torch.Tensor)

    @property
    def id(self) -> str:
        return QWEN3_ZIMAGE_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        require_pretrained_weights(model_path, self.id)
        text_encoder = AutoModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        text_encoder = model_components["text_encoder"]
        tokenizer = model_components["tokenizer"]
        prompts = kwargs["prompt"]
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)

        formatted = []
        for prompt in prompts:
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )
        text_inputs = tokenizer(
            formatted,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = text_inputs.attention_mask.to(device).bool()
        hidden_states = text_encoder(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]

        # DiFlow transports dense tensors between workers, so pad the reference
        # pipeline's list of variable-length embeddings and send the mask beside it.
        embeddings = [row[mask] for row, mask in zip(hidden_states, attention_mask)]
        max_length = max(row.shape[0] for row in embeddings)
        prompt_embeds = torch.stack(
            [
                torch.cat([row, row.new_zeros(max_length - row.shape[0], row.shape[1])])
                for row in embeddings
            ]
        ).to(dtype=text_encoder.dtype, device=device)
        encoder_attention_mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(row.shape[0], dtype=torch.long, device=device),
                        torch.zeros(
                            max_length - row.shape[0], dtype=torch.long, device=device
                        ),
                    ]
                )
                for row in embeddings
            ]
        )
        return {
            "prompt_embeds": prompt_embeds,
            "encoder_attention_mask": encoder_attention_mask,
        }
