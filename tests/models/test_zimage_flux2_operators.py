import pytest
import torch

from diflow.operators import (
    Config,
    Flux2FlowMatchEulerDiscreteScheduler,
    Flux2Klein,
    Flux2LatentsGenerator,
    Flux2VAE,
    Qwen3_Flux2Klein,
    Qwen3_ZImage,
    ZImage,
    ZImageFlowMatchEulerDiscreteScheduler,
    ZImageLatentsGenerator,
    ZImageVAE,
)
from diflow.operators.base import Operator
from diflow.operators.custom.flux2_latents_generator import pack_latents
from diflow.operators.flux2_utils import (
    compute_empirical_mu,
    prepare_latent_ids_4d,
    prepare_text_ids_4d,
)
from diflow.operators.models.autoencoders.flux_2_vae import (
    unpack_latents_with_ids,
    unpatchify_latents,
)


def test_zimage_latents_match_reference_shape_dtype_and_seed():
    generator = ZImageLatentsGenerator()
    first = generator.execute({}, "cpu", height=1024, width=1024, seed=7)["latents"]
    second = generator.execute({}, "cpu", height=1024, width=1024, seed=7)["latents"]
    assert first.shape == (1, 16, 128, 128)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_zimage_rejects_dimensions_the_reference_pipeline_rejects():
    with pytest.raises(ValueError, match="divisible by 16"):
        ZImageLatentsGenerator().execute({}, "cpu", height=1023, width=1024, seed=0)


def test_flux2_latents_match_reference_packed_layout():
    latents = Flux2LatentsGenerator().execute(
        {}, "cpu", height=1024, width=1024, seed=7
    )["latents"]
    assert latents.shape == (1, 4096, 128)
    assert latents.dtype == torch.bfloat16


def test_flux2_position_ids_are_batched_int64_coordinates():
    text_ids = prepare_text_ids_4d(2, 3, "cpu")
    latent_ids = prepare_latent_ids_4d(2, 2, 3, "cpu")
    assert text_ids.shape == (2, 3, 4)
    assert latent_ids.shape == (2, 6, 4)
    assert text_ids.dtype == latent_ids.dtype == torch.int64
    assert text_ids[0, :, 3].tolist() == [0, 1, 2]
    assert latent_ids[0, :, 1:3].tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ]


def test_flux2_pack_scatter_and_unpatchify_are_inverse_layouts():
    spatial = torch.arange(2 * 8 * 4 * 6).reshape(2, 8, 4, 6)
    packed = pack_latents(spatial)
    ids = prepare_latent_ids_4d(2, 4, 6, "cpu")
    assert torch.equal(unpack_latents_with_ids(packed, ids, 4, 6), spatial)

    image_latents = torch.arange(2 * 3 * 8 * 10).reshape(2, 3, 8, 10)
    patched = image_latents.reshape(2, 3, 4, 2, 5, 2)
    patched = patched.permute(0, 1, 3, 5, 2, 4).reshape(2, 12, 4, 5)
    assert torch.equal(unpatchify_latents(patched), image_latents)


def test_flux2_empirical_shift_has_expected_piecewise_behavior():
    assert compute_empirical_mu(4096, 10) > compute_empirical_mu(4096, 200)
    assert compute_empirical_mu(5000, 10) == compute_empirical_mu(5000, 200)


def test_zimage_scheduler_uses_positive_anchored_cfg_and_negates_prediction():
    class FakeScheduler:
        def step(self, noise_pred, timestep, latents, return_dict):
            assert timestep == 500.0
            assert return_dict is False
            return (noise_pred,)

    scheduler = ZImageFlowMatchEulerDiscreteScheduler()
    result = scheduler.execute(
        {"scheduler": FakeScheduler()},
        "cpu",
        mode="step_classifier_free_guidance",
        latents=torch.zeros(1),
        timestep=torch.tensor(500.0),
        noise_pred_uncond=torch.tensor([2.0]),
        noise_pred_text=torch.tensor([5.0]),
        guidance_scale=3.0,
    )
    assert torch.equal(result["output_latents"], torch.tensor([-14.0]))


def test_zimage_scheduler_uses_new_explicit_sigma_schedule_and_begin_index():
    class FakeScheduler:
        config = {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }

        def set_timesteps(self, *, sigmas, device, mu):
            self.received = (sigmas, device, mu)
            self.timesteps = torch.tensor(sigmas) * 1000

        def set_begin_index(self, index):
            self.begin_index = index

    fake = FakeScheduler()
    result = ZImageFlowMatchEulerDiscreteScheduler().execute(
        {"scheduler": fake},
        "cpu",
        mode="init",
        num_inference_steps=4,
        latents=torch.zeros(1, 16, 128, 128),
    )
    expected_sigmas = torch.linspace(1.0, 0.25, 4).tolist()
    assert fake.received[0] == expected_sigmas
    assert fake.received[1] == "cpu"
    assert fake.begin_index == 0
    assert torch.equal(result["timesteps"], torch.tensor(expected_sigmas) * 1000)


def test_new_operators_round_trip_through_registration_serialization():
    config = Config(model_path="/dummy/model")
    operators = [
        ZImageLatentsGenerator(),
        Qwen3_ZImage(config),
        ZImage(config),
        ZImageVAE(config),
        ZImageFlowMatchEulerDiscreteScheduler(config),
        Flux2LatentsGenerator(),
        Qwen3_Flux2Klein(config),
        Flux2Klein(config),
        Flux2VAE(config),
        Flux2FlowMatchEulerDiscreteScheduler(config),
    ]
    for operator in operators:
        restored = Operator.from_dict(operator.to_dict())
        assert restored.id == operator.id
        assert restored.to_dict() == operator.to_dict()


def test_flux2_vae_declares_every_decode_input_it_reads():
    inputs = Flux2VAE().get_execution_modes()["decode_latents"]["inputs"]
    assert set(inputs) == {"latents", "height", "width"}


def test_distilled_flux2_scheduler_does_not_advertise_cfg_mode():
    assert set(Flux2FlowMatchEulerDiscreteScheduler().get_execution_modes()) == {
        "init",
        "step",
    }
