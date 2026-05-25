import torch
import torch.nn.functional as F


def project_weight_gradient(
    module: torch.nn.Module,
    projection: torch.Tensor,
    max_norm: float | None = None,
) -> None:
    """Project a linear weight gradient into the editable null space."""
    grad = getattr(module.weight, "grad", None)
    if grad is None:
        return

    projection = projection.to(device=grad.device, dtype=grad.dtype)
    # Linear layers may expose either output-space or input-space compatible gradients.
    if grad.shape[0] == projection.shape[0]:
        projected = projection @ grad
    elif grad.shape[1] == projection.shape[0]:
        projected = grad @ projection
    else:
        raise ValueError(f"Projection shape {tuple(projection.shape)} is incompatible with gradient {tuple(grad.shape)}")

    if max_norm is not None:
        norm = projected.norm()
        if norm > max_norm:
            # A small norm cap keeps the edit local when the refusal loss is steep.
            projected = projected * (max_norm / norm.clamp_min(1e-8))

    grad.copy_(projected)


def refusal_alignment_loss(
    activation: torch.Tensor,
    refusal_direction: torch.Tensor,
    margin: float = 0.90,
    scale: float = 2.5,
) -> torch.Tensor:
    """Penalize activations until they align with the extracted refusal direction."""
    direction = refusal_direction.to(device=activation.device, dtype=torch.float32)
    sim = F.cosine_similarity(activation.float(), direction, dim=0)
    return torch.exp(scale * F.relu(margin - sim)) - 1.0
