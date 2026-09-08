import torch


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute the FLUX.2 piecewise-linear timestep shift."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    mu_200 = a2 * image_seq_len + b2
    mu_10 = a1 * image_seq_len + b1
    slope = (mu_200 - mu_10) / 190.0
    return float(slope * num_steps + mu_200 - 200.0 * slope)


def prepare_text_ids_4d(
    batch_size: int, seq_len: int, device: torch.device | str
) -> torch.Tensor:
    """Return FLUX.2 text coordinates with shape ``(B, seq_len, 4)``."""
    coords = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(1),
        torch.arange(1),
        torch.arange(seq_len),
    )
    # Position IDs must stay integral: bf16 cannot exactly represent larger
    # token coordinates and subtly changes rotary embeddings.
    return coords.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)


def prepare_latent_ids_4d(
    batch_size: int, height: int, width: int, device: torch.device | str
) -> torch.Tensor:
    """Return FLUX.2 latent coordinates with shape ``(B, H*W, 4)``."""
    coords = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(height),
        torch.arange(width),
        torch.arange(1),
    )
    return coords.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)
