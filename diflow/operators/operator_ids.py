"""
Centralized operator ID constants.

This module contains all operator ID constants used across the codebase.
These constants can be used for comparisons without needing to instantiate operator classes.
"""

# Diffusion Models
FLUX_1_SCHNELL_ID = "Flux1Schnell"
FLUX_1_DEV_ID = "Flux1Dev"

# Text Encoders
CLIP_FLUX_ID = "CLIP_Flux"
T5_FLUX_ID = "T5_Flux"

# Adapters (ControlNet)
FLUX_1_DEV_CONTROLNET_DEPTH_ID = "Flux1DevControlNetDepth"
FLUX_1_DEV_CONTROLNET_CANNY_ID = "Flux1DevControlNetCanny"

# Autoencoders (VAE)
FLUX_1_VAE_ID = "Flux1VAE"

# Patches (LoRA)

# Schedulers
PNDM_SCHEDULER_ID = "PNDMScheduler"
FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID = "FlowMatchEulerDiscreteScheduler"
FLUX_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID = "FluxFlowMatchEulerDiscreteScheduler"
FLUX_SCHNELL_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID = (
    "FluxSchnellFlowMatchEulerDiscreteScheduler"
)

# Custom Operators
FLUX_LATENTS_GENERATOR_ID = "FluxLatentsGenerator"
FLUX_TEXT_ENCODER_ID = "FluxTextEncoder"
GUIDANCE_TENSOR_ID = "GuidanceTensor"
INDEXED_TENSOR_ID = "IndexedTensor"
LATENTS_GENERATOR_ID = "LatentsGenerator"
