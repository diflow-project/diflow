from abc import abstractmethod
from typing import Any, Dict, Union

import torch

from diflow.operators.base import Operator
from diflow.operators.execution_modes import PATCH_OFF, PATCH_ON


class BasePatch(Operator):
    @abstractmethod
    def id(self) -> str:
        pass

    def setup_io(self):
        pass

    @abstractmethod
    def _patch_on(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        target_model_id: str,
        target_model_components: Dict[str, Any],
    ) -> None:
        """Apply the patch to the target model."""
        pass

    @abstractmethod
    def _patch_off(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        target_model_id: str,
        target_model_components: Dict[str, Any],
    ) -> None:
        """Remove the patch from the target model."""
        pass

    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        mode: str,
        target_model_id: str,  # the id of the target model
        target_model_components: Dict[str, Any],  # the components of the target model
    ) -> Dict[str, Any]:
        # All patches should implement two execution modes: PATCH_ON and PATCH_OFF.
        if mode == PATCH_ON:
            self._patch_on(
                model_components=model_components,
                device=device,
                target_model_id=target_model_id,
                target_model_components=target_model_components,
            )
        elif mode == PATCH_OFF:
            self._patch_off(
                model_components=model_components,
                device=device,
                target_model_id=target_model_id,
                target_model_components=target_model_components,
            )
        else:
            raise ValueError(
                f"Invalid mode for the patch {self.__class__.__name__}: {mode}. Supported modes are: {PATCH_ON}, {PATCH_OFF}"
            )
        return {}
