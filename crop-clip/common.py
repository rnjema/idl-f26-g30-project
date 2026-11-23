"""Shared helpers for the Crop-CLIP notebooks (crop-clip/*.ipynb).

Plain, readable Python on purpose: every notebook imports this file, so the data loading,
metrics and MegaDetector bookkeeping are defined once and reviewed once.

Contents
- constants (classes, splits, dataset mirror, MegaDetector release)
- paths
- dataset download + verification
- metadata loading + protocol checks (split sizes, 182 classes, camera-disjoint splits)
- per-image statistics (size, brightness, colourfulness, infrared flag)
- metrics (WILDS macro F1, accuracy)
- MegaDetector helpers (sharding, result rows, square padded crops)
- Weights & Biases helpers (runs, artifacts)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np
import pandas as pd

# ───────────────────────────────── constants ─────────────────────────────────

N_CLASSES = 182  # WILDS v2.0 classes are y = 0..181. categories.csv has extra catalog rows
EMPTY_Y = 0  # (y up to 99999): never use len(categories) as the class count.
SPLITS = ("train", "id_val", "val", "id_test", "test")
OOD_SPLITS = ("train", "val", "test")  # camera-disjoint slabs
EXPECTED_SPLIT_SIZES = {
    "train": 129_809,
    "val": 14_961,
    "test": 42_791,
    "id_val": 7_314,
    "id_test": 8_154,
}
EXPECTED_CAMERAS = {"train": 243, "val": 32, "test": 48}
SPLIT_LABELS = {
    "train": "Train",
    "id_val": "ID val",
    "val": "OOD val",
    "id_test": "ID test",
    "test": "OOD test",
}

# Team mirror of the official WILDS iWildCam v2.0 archive (unmodified copy), pinned and hashed.
MIRROR_REPO = "emmilly/iwildcam-wilds-v2"
MIRROR_REVISION = "1c01b321d444aa8a15e0bc7a0204d8c6559c8c5b"
ARCHIVE_NAME = "iwildcam_v2.0.tar.gz"
ARCHIVE_SHA256 = "05c872f46eee44315f79bc90f4098c69ae3799a2b95116db02292ea1b4e370d2"
ARCHIVE_BYTES = 11_957_415_473
METADATA_MD5 = "ab4905ee40ac45ab66ddb176d473a4b4"

# MegaDetector v5a, official release (Beery, Morris & Yang 2019; YOLOv5 architecture).
MD_VERSION = "v5a.0.0"
MD_WEIGHTS_URL = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
MD_WEIGHTS_MD5 = "ec1d7603ec8cf642d6e0cd008ba2be8c"  # as listed by the megadetector package
MD_IMAGE_SIZE = 1280  # v5a's native inference size (long side)
MD_CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}
MD_CONF_FLOOR = 0.005  # keep nearly every candidate; real thresholds are tuned on OOD val

# Infrared (night) frames are grayscale: their colour channels are almost identical.
IR_COLOURFULNESS_MAX = 3.0  # mean |max-min| over RGB channels, 0-255 scale; check the histogram

# Local tests only: lets the notebooks run on a tiny fake dataset (skips the exact-size checks).
FAKE_DATA = os.environ.get("CROPCLIP_ALLOW_FAKE_DATA") == "1"

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "team-support-vectors-idlf26")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "iwilds")
WANDB_GROUP = "crop-clip"
# The team W&B moved to CoreWeave Forge; every run and artifact goes there.
WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://api.forge.coreweave.com")

# ─────────────────────────────────── paths ───────────────────────────────────


def data_root() -> Path:
    """Folder holding iwildcam_v2.0/ (default ./data, the same place the YOLO-CLIP notebooks used)."""
    return Path(os.environ.get("CROPCLIP_DATA_ROOT", Path.cwd() / "data")).expanduser().resolve()


def work_root() -> Path:
    """Folder for everything Crop-CLIP writes (figures, boxes, crops, checkpoints)."""
    root = Path(os.environ.get("CROPCLIP_WORK_ROOT", data_root() / "cropclip")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def dataset_dir() -> Path:
    return data_root() / "iwildcam_v2.0"


def image_path(ds_dir: Path, filename: str) -> Path:
    """WILDS keeps every split's images in iwildcam_v2.0/train/ (the split lives in metadata.csv)."""
    return Path(ds_dir) / "train" / filename


# ──────────────────────────── dataset download + checks ────────────────────────


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_looks_complete(ds_dir: Path) -> bool:
    return (
        (ds_dir / "metadata.csv").is_file()
        and (ds_dir / "categories.csv").is_file()
        and (ds_dir / "train").is_dir()
    )


def ensure_dataset(ds_dir: Path | None = None, keep_archive: bool = False) -> Path:
    """Return a verified iwildcam_v2.0/ folder, downloading the pinned mirror archive if needed.

    Reuses an existing extraction (e.g. from the YOLO-CLIP notebooks) when its metadata matches.
    Checks: archive size + SHA-256, metadata.csv MD5, official split sizes, sample image decodes.
    """
    ds_dir = Path(ds_dir or dataset_dir())
    marker = ds_dir / ".cropclip_verified.json"
    if marker.is_file() and _dataset_looks_complete(ds_dir):
        return ds_dir
    if not _dataset_looks_complete(ds_dir):
        from huggingface_hub import hf_hub_download

        ds_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] downloading {MIRROR_REPO}@{MIRROR_REVISION[:8]} / {ARCHIVE_NAME} (~12 GB)")
        archive = Path(
            hf_hub_download(
                repo_id=MIRROR_REPO,
                filename=ARCHIVE_NAME,
                repo_type="dataset",
                revision=MIRROR_REVISION,
                local_dir=str(ds_dir.parent),
            )
        )
        size = archive.stat().st_size
        assert size == ARCHIVE_BYTES, f"archive is {size} bytes, expected {ARCHIVE_BYTES}"
        t0 = time.time()
        sha = file_sha256(archive)
        assert sha == ARCHIVE_SHA256, f"archive SHA-256 {sha} != {ARCHIVE_SHA256}"
        print(f"[data] size and SHA-256 verified ({time.time() - t0:.0f}s); extracting")
        ds_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:  # the archive is flat: ./train, ./metadata.csv
            try:
                tar.extractall(ds_dir, filter="data")
            except TypeError:  # Python without tarfile extraction filters
                tar.extractall(ds_dir)
        if not keep_archive:
            archive.unlink()
    md5 = file_md5(ds_dir / "metadata.csv")
    if FAKE_DATA:
        print("[data] CROPCLIP_ALLOW_FAKE_DATA=1: skipping dataset checks (local tests only)")
        return ds_dir
    assert md5 == METADATA_MD5, f"metadata.csv MD5 {md5} != {METADATA_MD5} (different dataset?)"
    sizes = pd.read_csv(ds_dir / "metadata.csv", usecols=["split"])["split"].value_counts()
    assert sizes.to_dict() == EXPECTED_SPLIT_SIZES, f"unexpected split sizes {sizes.to_dict()}"
    sample = pd.read_csv(ds_dir / "metadata.csv", usecols=["filename"]).sample(200, random_state=0)
    from PIL import Image

    for name in sample["filename"]:
        with Image.open(image_path(ds_dir, name)) as img:
            img.verify()
    marker.write_text(
        json.dumps(
            {
                "mirror": MIRROR_REPO,
                "revision": MIRROR_REVISION,
                "metadata_md5": md5,
                "split_sizes": EXPECTED_SPLIT_SIZES,
                "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    print(f"[data] verified {ds_dir}")
    return ds_dir


# ───────────────────────────── metadata + protocol ─────────────────────────────


def load_metadata(ds_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read metadata.csv and categories.csv; add friendly columns.

    Added columns: camera (= location_remapped), burst (= sequence_remapped), when (datetime),
    hour, is_empty, class_name. Returns (meta, class_names) with len(class_names) == 182.
    """
    ds_dir = Path(ds_dir)
    meta = pd.read_csv(ds_dir / "metadata.csv")
    cats = pd.read_csv(ds_dir / "categories.csv")
    cats = cats[cats["y"].between(0, N_CLASSES - 1)].sort_values("y")
    assert cats["y"].tolist() == list(range(N_CLASSES)), "class map must cover y = 0..181 once"
    name_col = "english" if "english" in cats.columns else "name"
    names = [
        (str(e) if isinstance(e, str) and e.strip() else str(n))
        for e, n in zip(cats[name_col], cats["name"])
    ]
    assert names[EMPTY_Y].strip().lower() == "empty", f"expected y=0 to be empty, got {names[0]}"

    meta["camera"] = meta["location_remapped"].astype(int)
    meta["burst"] = meta["sequence_remapped"].astype(int)
    meta["when"] = pd.to_datetime(meta["datetime"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    meta["hour"] = meta["when"].dt.hour
    meta["is_empty"] = meta["y"] == EMPTY_Y
    meta["class_name"] = [names[y] for y in meta["y"]]
    return meta, names


def verify_protocol(meta: pd.DataFrame) -> dict:
    """Assert the WILDS protocol facts our experiments rely on; return them as a dict."""
    sizes = meta["split"].value_counts().to_dict()
    assert meta["y"].between(0, N_CLASSES - 1).all(), "labels outside 0..181"
    cams = {s: set(meta.loc[meta["split"] == s, "camera"]) for s in SPLITS}
    counts = {s: len(cams[s]) for s in OOD_SPLITS}
    if not FAKE_DATA:
        assert sizes == EXPECTED_SPLIT_SIZES, f"split sizes {sizes}"
        assert meta["y"].nunique() == N_CLASSES, "not all 182 classes present"
        assert counts == EXPECTED_CAMERAS, f"camera counts {counts}"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (cams[a] & cams[b]), f"camera leakage between {a} and {b}"
    for s in ("id_val", "id_test"):
        assert cams[s] <= cams["train"], f"{s} should reuse training cameras"
    assert meta["filename"].is_unique, "duplicate filenames"
    return {"split_sizes": sizes, "ood_camera_counts": counts, "n_classes": N_CLASSES}


def split_overview(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per split: photos, cameras, bursts, classes, % empty, % infrared (if known)."""
    rows = []
    for s in SPLITS:
        part = meta[meta["split"] == s]
        row = {
            "split": SPLIT_LABELS[s],
            "photos": len(part),
            "cameras": part["camera"].nunique(),
            "bursts": part["burst"].nunique(),
            "classes": part["y"].nunique(),
            "% empty": round(100 * part["is_empty"].mean(), 1),
        }
        if "ir" in part.columns:
            row["% infrared"] = round(100 * part["ir"].mean(), 1)
        row["first date"] = str(part["when"].min())[:10]
        row["last date"] = str(part["when"].max())[:10]
        rows.append(row)
    return pd.DataFrame(rows)


def species_camera_table(meta: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Per class: training images, training cameras, and counts in each evaluation split."""
    tr = meta[meta["split"] == "train"]
    out = pd.DataFrame({"y": range(N_CLASSES), "class_name": names})
    out["train_images"] = tr["y"].value_counts().reindex(out["y"], fill_value=0).to_numpy()
    out["train_cameras"] = (
        tr.groupby("y")["camera"].nunique().reindex(out["y"], fill_value=0).to_numpy()
    )
    for s in ("id_val", "val", "id_test", "test"):
        counts = meta.loc[meta["split"] == s, "y"].value_counts()
        out[f"{s}_images"] = counts.reindex(out["y"], fill_value=0).to_numpy()
    return out


# ───────────────────────────── per-image statistics ─────────────────────────────


def one_image_stats(path: str) -> dict:
    """Size, brightness, colourfulness and an infrared flag, read at 1/8 scale (fast JPEG draft)."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            width, height = img.size
            img.draft("RGB", (max(1, width // 8), max(1, height // 8)))
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        colourfulness = float((arr.max(axis=2) - arr.min(axis=2)).mean())
        return {
            "width": width,
            "height": height,
            "brightness": float(arr.mean()),
            "colourfulness": colourfulness,
            "ir": colourfulness <= IR_COLOURFULNESS_MAX,
            "readable": True,
        }
    except Exception:  # unreadable file: keep the row, flag it
        return {
            "width": -1,
            "height": -1,
            "brightness": np.nan,
            "colourfulness": np.nan,
            "ir": False,
            "readable": False,
        }


def compute_image_stats(
    paths: list[str], workers: int = 8, desc: str = "image stats"
) -> pd.DataFrame:
    """Run one_image_stats over many files with a process pool; rows keep the input order."""
    from multiprocessing import Pool

    from tqdm.auto import tqdm

    paths = [str(p) for p in paths]
    if workers <= 1:
        rows = [one_image_stats(p) for p in tqdm(paths, desc=desc, mininterval=5)]
    else:
        with Pool(workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(one_image_stats, paths, chunksize=64),
                    total=len(paths),
                    desc=desc,
                    mininterval=5,
                )
            )
    return pd.DataFrame(rows)


# ─────────────────────────────────── metrics ───────────────────────────────────


def macro_f1(y_true, y_pred) -> float:
    """WILDS iWildCam macro F1: per-class F1 averaged over the classes present in y_true.

    Same as sklearn.f1_score(y_true, y_pred, average='macro', labels=unique(y_true)), which is
    what the official WILDS evaluator calls. Classes with no true and no predicted support give 0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = np.unique(y_true)
    size = int(max(y_true.max(), y_pred.max())) + 1
    tp = np.bincount(y_true[y_true == y_pred], minlength=size)[labels]
    pred_count = np.bincount(y_pred, minlength=size)[labels]
    true_count = np.bincount(y_true, minlength=size)[labels]
    denom = pred_count + true_count  # = 2tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


# ───────────────────────────────── MegaDetector ─────────────────────────────────


def chunk_ranges(n_items: int, chunk_size: int, shard: int = 0, n_shards: int = 1) -> list[tuple]:
    """Split [0, n_items) into numbered chunks; a shard takes every n_shards-th chunk.

    Returns [(chunk_id, start, stop), ...]. Chunk ids are global, so shards never overlap and
    merging is a concat of all chunk files.
    """
    assert 0 <= shard < n_shards, f"shard {shard} not in [0, {n_shards})"
    chunks = [
        (i, start, min(start + chunk_size, n_items))
        for i, start in enumerate(range(0, n_items, chunk_size))
    ]
    return [c for c in chunks if c[0] % n_shards == shard]


def parse_shard(text: str | None) -> tuple[int, int]:
    """'1/3' -> (1, 3); empty -> (0, 1). Shards are 0-based."""
    if not text:
        return 0, 1
    shard, n_shards = (int(x) for x in str(text).split("/"))
    return shard, n_shards


def download_md_weights(dest_dir: Path) -> tuple[Path, dict]:
    """Download MegaDetector v5a from the official release (once) and verify its MD5.

    Returns (path, info) where info records url, md5 and sha256 for the paper / W&B config.
    """
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / Path(MD_WEIGHTS_URL).name
    if not path.is_file() or file_md5(path) != MD_WEIGHTS_MD5:
        print(f"[md] downloading {MD_WEIGHTS_URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MD_WEIGHTS_URL, tmp)
        tmp.rename(path)
    md5 = file_md5(path)
    assert md5 == MD_WEIGHTS_MD5, f"MegaDetector weights MD5 {md5} != {MD_WEIGHTS_MD5}"
    return path, {
        "md_version": MD_VERSION,
        "md_weights_url": MD_WEIGHTS_URL,
        "md_weights_md5": md5,
        "md_weights_sha256": file_sha256(path),
        "md_image_size": MD_IMAGE_SIZE,
        "md_conf_floor": MD_CONF_FLOOR,
    }


def _load_for_md(path: str):
    from megadetector.visualization import visualization_utils as vis_utils

    try:
        return vis_utils.load_image(path)  # handles EXIF rotation, returns RGB PIL image
    except Exception as err:  # keep going; the row records the failure
        return err


def run_md_batches(
    detector,
    paths: list,
    batch_size: int = 16,
    threads: int = 8,
    conf_floor: float = MD_CONF_FLOOR,
    image_size: int = MD_IMAGE_SIZE,
) -> list[dict]:
    """Run MegaDetector over image files in batches; one result dict per path, same order.

    Images are decoded by a thread pool while the GPU works. A file that cannot be read gets
    {'failure': 'image load: ...'} instead of detections.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = [str(p) for p in paths]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        images = pool.map(_load_for_md, paths)
        batch_imgs, batch_ids = [], []

        def flush():
            if not batch_imgs:
                return
            out = detector.generate_detections_one_batch(
                list(batch_imgs),
                image_id=list(batch_ids),
                detection_threshold=conf_floor,
                image_size=image_size,
            )
            got = [str(r.get("file")) for r in out]
            assert got == batch_ids, "MegaDetector returned results out of order"
            results.extend(out)
            batch_imgs.clear()
            batch_ids.clear()

        for path, img in zip(paths, images):
            if isinstance(img, Exception):
                flush()  # keep output order aligned with input order
                results.append({"file": path, "detections": [], "failure": f"image load: {img}"})
                continue
            batch_imgs.append(img)
            batch_ids.append(path)
            if len(batch_imgs) == batch_size:
                flush()
        flush()
    assert len(results) == len(paths), f"{len(results)} results for {len(paths)} images"
    return results


BOX_COLUMNS = ["filename", "det_rank", "category", "conf", "x", "y", "w", "h", "failure"]


def detection_rows(filename: str, result: dict) -> list[dict]:
    """Flatten one MegaDetector result into rows (one per box, sorted by confidence).

    A photo with no boxes keeps one row with det_rank = -1 so every image appears in the table.
    bbox = [x_min, y_min, width, height], normalised to [0, 1] (MegaDetector's format).
    """
    failure = result.get("failure") if isinstance(result, dict) else "no result"
    dets = sorted(
        (result or {}).get("detections") or [], key=lambda d: float(d["conf"]), reverse=True
    )
    if failure or not dets:
        return [
            {
                "filename": filename,
                "det_rank": -1,
                "category": "",
                "conf": np.nan,
                "x": np.nan,
                "y": np.nan,
                "w": np.nan,
                "h": np.nan,
                "failure": str(failure or ""),
            }
        ]
    return [
        {
            "filename": filename,
            "det_rank": rank,
            "category": MD_CATEGORIES.get(str(d["category"]), str(d["category"])),
            "conf": float(d["conf"]),
            "x": float(d["bbox"][0]),
            "y": float(d["bbox"][1]),
            "w": float(d["bbox"][2]),
            "h": float(d["bbox"][3]),
            "failure": "",
        }
        for rank, d in enumerate(dets)
    ]


def read_boxes(path) -> pd.DataFrame:
    """Read a box table (chunk or boxes_all.csv.gz) with text columns kept as text.

    Photos without boxes have empty strings for category/failure and NaN coordinates.
    """
    boxes = pd.read_csv(
        path, dtype={"filename": str, "category": str, "failure": str}, keep_default_na=False
    )
    for col in ("conf", "x", "y", "w", "h"):
        boxes[col] = pd.to_numeric(boxes[col], errors="coerce")
    return boxes


def per_image_box_summary(
    boxes: pd.DataFrame, thresholds=(0.1, 0.2, 0.3, 0.5, 0.8)
) -> pd.DataFrame:
    """One row per image: max animal confidence and number of animal boxes above each threshold."""
    animals = boxes[boxes["category"] == "animal"]
    out = pd.DataFrame({"filename": boxes["filename"].unique()})
    max_conf = animals.groupby("filename")["conf"].max()
    out["max_animal_conf"] = out["filename"].map(max_conf).fillna(0.0)
    for t in thresholds:
        counts = animals[animals["conf"] >= t].groupby("filename").size()
        out[f"n_animal@{t}"] = out["filename"].map(counts).fillna(0).astype(int)
    failures = boxes.loc[boxes["failure"].fillna("") != "", "filename"].unique()
    out["failure"] = out["filename"].isin(failures)
    return out


def square_crop_box(bbox, width: int, height: int, pad: float = 0.1) -> tuple[int, int, int, int]:
    """Normalised [x, y, w, h] box -> square pixel box (left, top, right, bottom) with padding.

    The square side is the longer box side times (1 + 2*pad), centred on the box and shifted (not
    squashed) to stay inside the image. If the square is larger than the image in a direction, it
    is centred there and extends past the edge: PIL's Image.crop fills that area with black, so the
    animal is never cut off (letterboxing).
    """
    x, y, w, h = (float(v) for v in bbox)
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    side = max(2.0, max(w * width, h * height) * (1 + 2 * pad))

    def _start(centre: float, size: int) -> float:
        if side >= size:
            return (size - side) / 2  # larger than the image: centre it, pad both sides
        return min(max(centre - side / 2, 0.0), size - side)

    left, top = _start(cx, width), _start(cy, height)
    return (int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side)))


# ─────────────────────────────── Weights & Biases ───────────────────────────────


def git_commit(repo_dir: Path | None = None) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir or Path(__file__).resolve().parent),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def wandb_run(name: str, job_type: str, config: dict | None = None, enabled: bool = True):
    """Start a W&B run in the team project (group 'crop-clip'); returns None when disabled."""
    if not enabled:
        return None
    import wandb

    os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL
    config = dict(config or {})
    config.setdefault("git_commit", git_commit())
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=name,
        job_type=job_type,
        tags=[WANDB_GROUP, job_type],
        config=config,
        reinit=True,
    )


def log_artifact(run, name: str, kind: str, paths: list[Path], metadata: dict | None = None):
    """Upload files as a versioned W&B artifact (e.g. boxes, image stats, checkpoints)."""
    if run is None:
        print(f"[wandb] disabled: not uploading artifact {name}")
        return None
    import wandb

    art = wandb.Artifact(name, type=kind, metadata=metadata or {})
    for p in paths:
        art.add_file(str(p))
    run.log_artifact(art)
    print(f"[wandb] logged artifact {name} ({len(paths)} file(s))")
    return art


def fetch_artifact(name: str, dest: Path, alias: str = "latest", run=None) -> Path:
    """Download a team artifact (e.g. 'megadetector-v5a-boxes') into dest; returns the folder."""
    import wandb

    os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL
    full = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{name}:{alias}"
    art = run.use_artifact(full) if run is not None else wandb.Api().artifact(full)
    return Path(art.download(root=str(dest)))
