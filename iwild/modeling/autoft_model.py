"""CLIP model wrapper, zero-shot head construction, and checkpoint IO for AutoFT."""

import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from loguru import logger
from tqdm import tqdm

from iwild.modeling.autoft_data import get_iwildcam_classnames

IWILDCAM_TEMPLATES = [
    lambda c: f"a photo of {c}.",
    lambda c: f"{c} in the wild.",
]
DEFAULT_ARCH = "ViT-B-16"
DEFAULT_PRETRAINED = "openai"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(torch.nn.Module):
    """Wraps an open_clip model, keeping the text tower (keep_lang=True)."""

    def __init__(
        self,
        arch: str = DEFAULT_ARCH,
        pretrained: str = DEFAULT_PRETRAINED,
        keep_lang: bool = True,
    ):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        # The OpenAI pretrained tags ship QuickGELU weights; force it so the activations
        # match the official OpenAI/auto-ft CLIP instead of the exact-GELU default.
        self.model, self.train_preprocess, self.val_preprocess = (
            open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, force_quick_gelu=pretrained == "openai"
            )
        )
        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images, text=None):
        if text is None:
            return self.model.encode_image(images)
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(text)
        return image_features, text_features, self.model.logit_scale


class ClassificationHead(torch.nn.Linear):
    """Linear head that L2-normalizes its inputs before applying the weights."""

    def __init__(self, normalize, weights, biases=None, shape=(512, 182)):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class ImageClassifier(torch.nn.Module):
    """Image encoder plus L2-normalizing classification head."""

    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, images, text=None):
        if text is None:
            image_features = self.image_encoder(images)
            text_features, logit_scale = None, None
        else:
            image_features, text_features, logit_scale = self.image_encoder(images, text)
        logits = self.classification_head(image_features)
        return logits, image_features, text_features, logit_scale


@torch.no_grad()
def get_zeroshot_classifier(clip_model, classnames=None, device=None) -> ClassificationHead:
    """Build the zero-shot head: mean of prompt embeddings, L2-normalized, scaled."""
    if classnames is None:
        classnames = get_iwildcam_classnames()
    assert len(classnames) == 182, f"expected 182 classnames, got {len(classnames)}"
    device = device or get_device()
    logit_scale = clip_model.logit_scale
    clip_model.eval()
    clip_model.to(device)

    zeroshot_weights = []
    for classname in tqdm(classnames, desc="Building zero-shot head"):
        texts = [t(classname) for t in IWILDCAM_TEMPLATES]
        embeddings = clip_model.encode_text(open_clip.tokenize(texts).to(device))
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.mean(dim=0, keepdim=True)
        embeddings = embeddings / embeddings.norm()
        zeroshot_weights.append(embeddings)

    zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
    zeroshot_weights = zeroshot_weights * logit_scale.exp()
    return ClassificationHead(normalize=True, weights=zeroshot_weights.float())


def build_zeroshot_model(
    arch: str = DEFAULT_ARCH, pretrained: str = DEFAULT_PRETRAINED, device=None
) -> ImageClassifier:
    """Fresh CLIP encoder plus zero-shot iWildCam head."""
    device = device or get_device()
    image_encoder = ImageEncoder(arch=arch, pretrained=pretrained, keep_lang=True)
    classification_head = get_zeroshot_classifier(image_encoder.model, device=device)
    model = ImageClassifier(image_encoder, classification_head)
    return model.to(device)


def snapshot_initial_params(model) -> list[torch.Tensor]:
    """Detached copies of the current parameters, used by the l1init/l2init losses."""
    return [param.detach().clone() for param in model.parameters()]


def save_classifier(model: ImageClassifier, path: Path) -> None:
    """Save state_dict plus a small JSON config (state keys, arch, num classes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = model.image_encoder
    config = {
        "arch": encoder.arch,
        "pretrained": encoder.pretrained,
        "num_classes": model.classification_head.out_features,
        "normalize": model.classification_head.normalize,
    }
    torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2))
    logger.success(f"Saved classifier to {path} (+ .json config)")


def load_classifier(path: Path, device=None) -> ImageClassifier:
    """Rebuild a classifier from a state_dict checkpoint and its JSON config."""
    path = Path(path)
    device = device or get_device()
    config = json.loads(path.with_suffix(".json").read_text())
    arch = config.get("arch", DEFAULT_ARCH)
    pretrained = config.get("pretrained", DEFAULT_PRETRAINED)
    num_classes = config["num_classes"]
    image_encoder = ImageEncoder(arch=arch, pretrained=pretrained, keep_lang=True)
    embed_dim = image_encoder.model.text_projection.shape[-1]
    classification_head = ClassificationHead(
        normalize=config.get("normalize", True), weights=None, shape=(embed_dim, num_classes)
    )
    model = ImageClassifier(image_encoder, classification_head)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    logger.success(f"Loaded classifier from {path}")
    return model.to(device)


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and torch (CPU + CUDA), matching the official utils."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
