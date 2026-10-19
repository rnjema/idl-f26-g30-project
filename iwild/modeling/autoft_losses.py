"""Learned loss for AutoFT: weighted sum of 9 loss terms (official formulas)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_hinge_loss(logits, targets, margin=1.0):
    """Multi-class hinge loss between logits and ground-truth targets."""
    targets = targets.long()
    correct_scores = logits[torch.arange(logits.shape[0]), targets]
    margins = logits - correct_scores[:, None] + margin
    margins[torch.arange(logits.shape[0]), targets] = 0
    loss = torch.relu(margins)
    return loss.sum(dim=1).mean()


class ClipLoss(nn.Module):
    """CLIP contrastive image-text loss (single-process configuration)."""

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss


class LearnedLoss(nn.Module):
    """Weighted sum of the selected loss terms; regularizers normalized by param count."""

    def __init__(self, losses, loss_weights, initial_params, device=None):
        super().__init__()
        self.losses = losses
        device = device or initial_params[0].device
        self.initial_params = [param.detach().to(device) for param in initial_params]
        if isinstance(loss_weights, list):
            loss_weights = torch.stack(loss_weights)
        self.loss_weights = loss_weights.to(device).float()
        self.param_sum = sum(param.numel() for param in initial_params)
        self.device = device
        if "flyp" in self.losses:
            self.clip_loss_fn = ClipLoss(
                local_loss=False,
                gather_with_grad=False,
                cache_labels=True,
                rank=0,
                world_size=1,
                use_horovod=False,
            )

    def forward(self, model, logits, labels, image_features, text_features=None, logit_scale=None):
        losses = []
        if "ce" in self.losses:
            ce_loss = F.cross_entropy(logits, labels)
            losses.append(ce_loss)
        if "dcm" in self.losses or "entropy" in self.losses:
            entropy_all = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
        if "dcm" in self.losses:
            dcm_loss = ((logits.argmax(dim=1) != labels).float().detach() * entropy_all).mean()
            losses.append(dcm_loss)
        if "entropy" in self.losses:
            entropy = entropy_all.mean()
            losses.append(entropy)
        if "flyp" in self.losses:
            clip_loss = self.clip_loss_fn(image_features, text_features, logit_scale)
            losses.append(clip_loss)
        if "hinge" in self.losses:
            hinge_loss = compute_hinge_loss(logits, labels)
            losses.append(hinge_loss)

        if any(k in self.losses for k in ("l1zero", "l2zero", "l1init", "l2init")):
            l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum = 0, 0, 0, 0
            for param, init_param in zip(model.parameters(), self.initial_params):
                l1_zero_accum = l1_zero_accum + torch.abs(param).sum()
                l2_zero_accum = l2_zero_accum + (param**2).sum()
                diff = param - init_param
                l1_init_accum = l1_init_accum + torch.abs(diff).sum()
                l2_init_accum = l2_init_accum + (diff**2).sum()
            l1_zero_accum = l1_zero_accum / self.param_sum
            l2_zero_accum = l2_zero_accum / self.param_sum
            l1_init_accum = l1_init_accum / self.param_sum
            l2_init_accum = l2_init_accum / self.param_sum
            if "l1init" in self.losses:
                losses.append(l1_init_accum)
            if "l1zero" in self.losses:
                losses.append(l1_zero_accum)
            if "l2init" in self.losses:
                losses.append(l2_init_accum)
            if "l2zero" in self.losses:
                losses.append(l2_zero_accum)
            del l1_zero_accum, l2_zero_accum, l1_init_accum, l2_init_accum

        total = torch.matmul(self.loss_weights, torch.stack(losses))
        return total.mean()


def get_loss_weights(losses, hparams: dict) -> torch.Tensor:
    """Collect the loss weights from hparams; missing weights default to 1.0."""
    loss_weight_keys = [k for k in sorted(hparams.keys()) if "lossw" in k and "_lossw" not in k]
    for loss_type in losses:
        key = f"lossw_{loss_type}"
        if key not in loss_weight_keys:
            hparams[key] = 1.0
            loss_weight_keys.append(key)
    return torch.tensor([hparams[k] for k in sorted(loss_weight_keys)])
