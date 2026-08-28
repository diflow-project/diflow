from diflow.operators.models.diffusion_models.zimage import ZImage
from diflow.operators.operator_ids import ZIMAGE_TURBO_ID


class ZImageTurbo(ZImage):
    """Z-Image Turbo checkpoint, kept distinct for placement and model loading."""

    @property
    def id(self) -> str:
        return ZIMAGE_TURBO_ID
