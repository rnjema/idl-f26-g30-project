"""Data utilities for the AutoFT / FLYP pipeline on WILDS iWildCam."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from iwild.config import CSV_DIR, IWILDCAM_LABELS_PATH, RAW_DATA_DIR

IWILDCAM_DIRNAME = "iwildcam_v2.0"
NUM_WILDS_CLASSES = 182
DEFAULT_NUM_OOD_HP_EXAMPLES = 1000


def ft_csv_path() -> Path:
    """Return the path of the FLYP/AutoFT fine-tuning CSV."""
    return CSV_DIR / IWILDCAM_DIRNAME / "iwildcam.csv"


def get_iwildcam_classnames() -> list[str]:
    """Classnames from the official labels.csv, lowercased (all 205 rows)."""
    df = pd.read_csv(IWILDCAM_LABELS_PATH)
    df = df[df["y"] < 99999]
    return [s.lower() for s in list(df["english"])]


class IWildCamSplit:
    """A single WILDS iWildCam split with classnames and post-loop metrics."""

    def __init__(
        self,
        split: str,
        transform,
        root: Path = RAW_DATA_DIR,
        n_examples: int = -1,
        use_class_balanced: bool = False,
    ):
        import wilds

        self.split = split
        base_dataset = wilds.get_dataset(dataset="iwildcam", root_dir=str(root))
        subset = base_dataset.get_subset(split, transform=transform)
        if n_examples > -1:
            if use_class_balanced:
                n_examples_per_class = n_examples // NUM_WILDS_CLASSES
                indices = class_balanced_indices(subset, n_examples_per_class)
            else:
                indices = np.random.choice(len(subset), n_examples, replace=False)
            subset = Subset(subset, indices)
        self.dataset = subset
        self.classnames = get_iwildcam_classnames()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def post_loop_metrics(self, labels, preds, metadata):
        """Runs the official WILDS eval and returns the full metric dict."""
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        dataset = self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset
        results = dataset.eval(preds, labels, metadata)
        return results[0]


def class_balanced_indices(dataset, num_samples_per_class: int, seed: int = 0) -> list[int]:
    """Sample a fixed number of indices per class, with replacement if needed."""
    targets = np.array([int(y) for y in dataset.y_array])
    rng = np.random.default_rng(seed)
    sampled_indices = []
    for cls in np.unique(targets):
        class_indices = np.where(targets == cls)[0]
        if num_samples_per_class > len(class_indices):
            sampled = rng.choice(class_indices, num_samples_per_class, replace=True)
        else:
            sampled = rng.choice(class_indices, num_samples_per_class, replace=False)
        sampled_indices.append(sampled)
    indices = np.concatenate(sampled_indices)
    rng.shuffle(indices)
    return [int(i) for i in indices]


def make_ood_hp_subset(
    transform,
    num_examples: int = DEFAULT_NUM_OOD_HP_EXAMPLES,
    root: Path = RAW_DATA_DIR,
    seed: int = 0,
) -> IWildCamSplit:
    """Class-balanced subset of the OOD val split used as the HP-search objective."""
    ood = IWildCamSplit("val", transform, root=root)
    indices = class_balanced_indices(ood.dataset, num_examples // NUM_WILDS_CLASSES, seed=seed)
    ood.dataset = Subset(ood.dataset, indices)
    return ood


class CsvDataset(Dataset):
    """FLYP image-caption dataset over a tab-separated CSV (filepath/title/label).

    Captions are BPE-tokenized once per unique string at construction (there are only
    ~hundreds of unique prompts) instead of once per row per epoch - tokenization in
    __getitem__ was a major CPU bottleneck that starved the GPU.
    """

    def __init__(
        self,
        input_filename: Path,
        transforms,
        img_key: str = "filepath",
        caption_key: str = "title",
        sep: str = "\t",
        label_key: str = "label",
        max_examples: int | None = None,
    ):
        input_filename = Path(input_filename)
        if not input_filename.exists():
            raise FileNotFoundError(
                f"Fine-tuning CSV not found at {input_filename}. "
                "Generate it with: python iwild/dataset.py make-ft-csv"
            )
        df = pd.read_csv(input_filename, sep=sep, usecols=[img_key, caption_key, label_key])
        if max_examples is not None:
            df = df.iloc[:max_examples].reset_index(drop=True)
        logger.info(f"Loading csv data from {input_filename}: {len(df)} rows.")
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.labels = list(map(int, df[label_key].tolist()))
        self.transforms = transforms

        from open_clip import tokenize

        unique_captions = list(dict.fromkeys(self.captions))
        cache = {
            cap: tokenize([cap])[0] for cap in tqdm(unique_captions, desc="Tokenizing captions")
        }
        self.tokens = [cache[cap] for cap in self.captions]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        return images, self.labels[idx], self.tokens[idx]


def get_dataloaders(
    batch_size: int,
    ft_preprocess,
    val_preprocess,
    num_ood_hp_examples: int = DEFAULT_NUM_OOD_HP_EXAMPLES,
    max_train_examples: int | None = None,
    num_workers: int = 2,
    root: Path = RAW_DATA_DIR,
    seed: int = 0,
) -> dict:
    """Build the FT, ID-val and class-balanced OOD-HP dataloaders."""
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        # A100-class GPUs are dataloader-starved without these
        kwargs.update(persistent_workers=True, prefetch_factor=4)

    ft_dataset = CsvDataset(ft_csv_path(), ft_preprocess, max_examples=max_train_examples)
    ft_loader = DataLoader(
        ft_dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs
    )

    id_val = IWildCamSplit("id_val", val_preprocess, root=root)
    id_val_loader = DataLoader(id_val.dataset, batch_size=batch_size, shuffle=False, **kwargs)

    ood_hp = make_ood_hp_subset(
        val_preprocess, num_examples=num_ood_hp_examples, root=root, seed=seed
    )
    ood_hp_loader = DataLoader(ood_hp.dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return {
        "id": ft_loader,
        "id_val": id_val_loader,
        "ood_hp": ood_hp_loader,
        "datasets": {"id_val": id_val, "ood_hp": ood_hp},
    }


def evaluate_split(net, dataloader, split: IWildCamSplit, device) -> dict:
    """Evaluate on one split; returns accuracy plus the WILDS post-loop metrics."""
    net = net.to(device).eval()
    total_correct, total_samples = 0, 0
    all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {split.split}"):
            images, labels, metadata = batch
            x, y = images.to(device), labels.to(device)
            outputs, _, _, _ = net(x)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)
            all_labels.append(y.cpu())
            all_preds.append(outputs.cpu())
            all_metadata.extend(metadata)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    metrics = split.post_loop_metrics(all_labels, all_preds, all_metadata)
    metrics["acc"] = 100 * total_correct / total_samples
    net.train()
    return metrics
