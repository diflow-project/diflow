from typing import Any, Dict, List, Union

import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diflow.operators.base import Operator, require_pretrained_weights
from diflow.operators.operator_ids import QWEN3_FLUX2_KLEIN_ID


class Qwen3_Flux2Klein(Operator):
    """Build FLUX.2 Klein's concatenated Qwen3 intermediate features."""

    HIDDEN_STATE_LAYERS = (9, 18, 27)

    def setup_io(self):
        self.add_input("prompt", Union[str, List[str]])
        self.add_output("prompt_embeds", torch.Tensor)

    @property
    def id(self) -> str:
        return QWEN3_FLUX2_KLEIN_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        require_pretrained_weights(model_path, self.id)
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
        ).to(device)
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_path, subfolder="tokenizer"
        )
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

        input_ids = []
        attention_masks = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            input_ids.append(encoded.input_ids)
            attention_masks.append(encoded.attention_mask)

        output = text_encoder(
            input_ids=torch.cat(input_ids).to(device),
            attention_mask=torch.cat(attention_masks).to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        selected = torch.stack(
            [output.hidden_states[index] for index in self.HIDDEN_STATE_LAYERS],
            dim=1,
        ).to(dtype=text_encoder.dtype, device=device)
        batch, layers, sequence, hidden = selected.shape
        prompt_embeds = selected.permute(0, 2, 1, 3).reshape(
            batch, sequence, layers * hidden
        )
        return {"prompt_embeds": prompt_embeds}
