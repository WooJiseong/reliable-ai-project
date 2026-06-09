import torch
import torch.nn.functional as F


def ctc_loss_from_batch(model, batch: dict, blank_id: int) -> torch.Tensor:
    kwargs = {}
    if "language" in batch:
        kwargs["language"] = batch["language"]
    logits, output_lengths = model(
        batch["waveform"],
        batch["waveform_length"],
        **kwargs,
    )
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        batch["tokens"],
        output_lengths,
        batch["token_length"],
        blank=blank_id,
        zero_infinity=True,
    )


def pgd_attack(
    model,
    batch: dict,
    blank_id: int,
    epsilon: float = 0.002,
    alpha: float = 0.0005,
    steps: int = 5,
    random_start: bool = True,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Untargeted PGD that maximizes CTC loss under an L-infinity bound."""
    model.eval()
    clean = batch["waveform"].detach()

    if random_start:
        delta = torch.empty_like(clean).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(clean)

    attacked = (clean + delta).clamp(clamp_min, clamp_max).detach()

    for _ in range(steps):
        attacked.requires_grad_(True)
        attack_batch = dict(batch)
        attack_batch["waveform"] = attacked
        loss = ctc_loss_from_batch(model, attack_batch, blank_id)
        grad = torch.autograd.grad(loss, attacked, only_inputs=True)[0]

        attacked = attacked.detach() + alpha * grad.sign()
        delta = (attacked - clean).clamp(-epsilon, epsilon)
        attacked = (clean + delta).clamp(clamp_min, clamp_max).detach()

    return attacked
