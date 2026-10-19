"""iWildCam (WILDS) dataset utilities.

Commands:
- download: fetch iwildcam_v2.0 from the CodaLab bundle and verify split/leakage invariants.
- make-ft-csv: build the FLYP/AutoFT fine-tuning CSV (one row per image x text prompt).
- location-split: seeded 80/20 camera-trap split of the official train split (for the selector
  lane; the AutoFT baseline trains on the full official train split).
"""

import json
import random
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from iwild.config import (
    CSV_DIR,
    INTERIM_DATA_DIR,
    IWILDCAM_LABELS_PATH,
    RAW_DATA_DIR,
)

app = typer.Typer()

# CodaLab bundle hosting WILDS iWildCam v2.0 (compressed ~12 GB).
# The https endpoint's certificate is currently broken, so the plain-http endpoint is
# tried first and the https fallback disables verification. Manual grab, if needed:
# https://worksheets.codalab.org/bundles/0x6313da2b204647e79a14b468131fcd64
CODALAB_BUNDLE_ID = "0x6313da2b204647e79a14b468131fcd64"
DOWNLOAD_URLS = (
    f"http://worksheets.codalab.org/rest/bundles/{CODALAB_BUNDLE_ID}/contents/blob/",
    f"https://worksheets.codalab.org/rest/bundles/{CODALAB_BUNDLE_ID}/contents/blob/",
)
_COMPRESSED_SIZE = 11_957_420_032  # bytes, per the WILDS 2.0 dataset definition

# WILDS iWildCam v2.0 verified facts (WILDS paper App. E.1 / WILDS 2.0)
EXPECTED_SPLIT_SIZES = {
    "train": 129_809,
    "val": 14_961,
    "test": 42_791,
    "id_val": 7_314,
    "id_test": 8_154,
}
EXPECTED_N_CLASSES = 182
EXPECTED_TRAP_COUNTS = {"train": 243, "val": 32, "test": 48}
N_TRAIN_IMAGES = 129_809

# Official auto-ft iWildCam text prompts (src/templates/iwildcam_template.py)
IWILDCAM_PROMPTS = [lambda c: f"a photo of {c}.", lambda c: f"{c} in the wild."]


def _load_metadata() -> pd.DataFrame:
    metadata_path = RAW_DATA_DIR / "iwildcam_v2.0" / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"metadata.csv not found at {metadata_path}. Run `download` first.")
        raise typer.Exit(code=1)
    return pd.read_csv(metadata_path)


def _verify_metadata(df: pd.DataFrame) -> None:
    sizes = df.groupby("split").size().to_dict()
    for split, expected in EXPECTED_SPLIT_SIZES.items():
        assert (
            sizes.get(split) == expected
        ), f"split {split}: got {sizes.get(split)}, expected {expected}"
    n_classes = df["y"].max() + 1
    assert (
        n_classes == EXPECTED_N_CLASSES
    ), f"got {n_classes} classes, expected {EXPECTED_N_CLASSES}"

    ood_splits = df[df["split"].isin(["train", "val", "test"])]
    traps = ood_splits.groupby("split")["location_remapped"].nunique().to_dict()
    for split, expected in EXPECTED_TRAP_COUNTS.items():
        assert (
            traps.get(split) == expected
        ), f"split {split}: got {traps.get(split)} traps, expected {expected}"

    # Leakage rule from the research protocol: no camera trap crosses split slabs
    train_traps = set(df.loc[df["split"] == "train", "location_remapped"])
    for split in ["val", "test"]:
        overlap = train_traps & set(df.loc[df["split"] == split, "location_remapped"])
        assert not overlap, f"location leakage: {len(overlap)} traps in both train and {split}"

    logger.success(
        f"Verified: {sum(EXPECTED_SPLIT_SIZES.values())} images, {EXPECTED_N_CLASSES} classes, "
        f"trap counts {EXPECTED_TRAP_COUNTS}, no location overlap across splits."
    )


def _download_and_extract(root_dir: Path) -> None:
    """Stream-download the CodaLab tarball (certificate-less endpoint first) and extract it."""
    import tarfile

    import requests
    import urllib3
    from tqdm import tqdm

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    tarball = root_dir / "iwildcam_v2.0.tar.gz"

    last_error = None
    for url in DOWNLOAD_URLS:
        verify = url.startswith("https")
        if verify:
            logger.warning("Certificate-less endpoint failed; retrying without TLS verification.")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        else:
            logger.info(f"Downloading iWildCam v2.0 from {url} (~12 GB compressed)...")
        try:
            with requests.get(url, stream=True, timeout=60, verify=verify) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length") or _COMPRESSED_SIZE)
                with open(tarball, "wb") as handle, tqdm(
                    total=total, unit="B", unit_scale=True, desc="iwildcam_v2.0.tar.gz"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
                        pbar.update(len(chunk))
            break
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(f"Download from {url} failed: {exc}")
            tarball.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"All download endpoints failed. Last error: {last_error}")

    logger.info("Extracting tarball (this takes a few minutes)...")
    with tarfile.open(tarball) as tar:
        tar.extractall(path=root_dir)
    tarball.unlink()


@app.command()
def download(
    root_dir: Path = RAW_DATA_DIR,
):
    """Download iWildCam v2.0 from the CodaLab bundle and verify split/leakage invariants."""
    root_dir = Path(root_dir)
    dataset_dir = root_dir / "iwildcam_v2.0"
    if (dataset_dir / "metadata.csv").exists():
        logger.info(f"{dataset_dir} already exists; skipping download.")
    else:
        try:
            _download_and_extract(root_dir)
        except Exception as exc:
            logger.error(
                f"Download failed: {exc}. Grab the bundle manually at "
                f"https://worksheets.codalab.org/bundles/{CODALAB_BUNDLE_ID} and place the "
                f"extracted `iwildcam_v2.0/` directory under {root_dir}."
            )
            raise typer.Exit(code=1)

    if not (dataset_dir / "metadata.csv").exists():
        logger.error(
            f"{dataset_dir}/metadata.csv not found after download. Extract the bundle manually "
            f"(https://worksheets.codalab.org/bundles/{CODALAB_BUNDLE_ID}) so that "
            f"{dataset_dir}/metadata.csv exists, then re-run `verify`."
        )
        raise typer.Exit(code=1)

    _verify_metadata(pd.read_csv(dataset_dir / "metadata.csv"))


@app.command()
def verify():
    """Re-run split-size, class-count, and location-leakage assertions on downloaded metadata."""
    _verify_metadata(_load_metadata())


@app.command()
def make_ft_csv(
    output_path: Path = CSV_DIR / "iwildcam_v2.0" / "iwildcam.csv",
):
    """Build the FLYP/AutoFT fine-tuning CSV: one row per (train image x text prompt).

    Ports auto-ft/datacreation_scripts/iwildcam.py. Tab-separated `title,filepath,label`,
    2 x 129,809 = 259,618 rows. Titles use the latin-binomial `name` column (as the official
    script does, which differs from the `english` classnames used at eval time).
    """
    df = _load_metadata()
    df = df[(df["split"] == "train") & (df["y"] < 99999)][["filename", "y"]]

    label_to_name = pd.read_csv(IWILDCAM_LABELS_PATH)
    label_to_name = label_to_name[label_to_name["y"] < 99999]
    label_to_name["prompt1"] = label_to_name["name"].map(IWILDCAM_PROMPTS[0])
    label_to_name["prompt2"] = label_to_name["name"].map(IWILDCAM_PROMPTS[1])

    assert len(df) == N_TRAIN_IMAGES, "number of train samples incorrect"

    df1 = pd.merge(df, label_to_name[["y", "prompt1"]], on="y").rename(
        {"prompt1": "title"}, axis=1
    )
    df2 = pd.merge(df, label_to_name[["y", "prompt2"]], on="y").rename(
        {"prompt2": "title"}, axis=1
    )
    assert len(df1) == N_TRAIN_IMAGES and len(df2) == N_TRAIN_IMAGES

    df_final = pd.concat((df1, df2))[["filename", "title", "y"]]
    train_img_dir = RAW_DATA_DIR / "iwildcam_v2.0" / "train"
    df_final["filename"] = df_final["filename"].map(lambda x: str(train_img_dir / x))
    df_final = df_final.rename({"filename": "filepath", "y": "label"}, axis=1)[
        ["title", "filepath", "label"]
    ]

    assert len(df_final) == N_TRAIN_IMAGES * 2, "number of fine-tuning rows incorrect"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, sep="\t", index=False, header=True)
    logger.success(f"Wrote {len(df_final)} rows to {output_path}")


@app.command()
def location_split(
    seed: int = 42,
    frac_a: float = 0.8,
    output_path: Path = INTERIM_DATA_DIR / "location_split.json",
):
    """Split the 243 official-train camera traps into disjoint locations A (80%) and B (20%).

    Locations A/B are the leakage-free slabs for the CF-delta selector lane (B trains selectors,
    never the classifier). The AutoFT baseline itself trains on the full official train split;
    a split-A-only classifier (for the selector lane) is a later milestone and will consume
    this JSON.
    """
    if not 0.0 < frac_a < 1.0:
        raise typer.BadParameter("frac_a must be in (0, 1)")
    df = _load_metadata()
    train_traps = sorted(df.loc[df["split"] == "train", "location_remapped"].unique())
    assert len(train_traps) == EXPECTED_TRAP_COUNTS["train"]

    rng = random.Random(seed)
    shuffled = train_traps[:]
    rng.shuffle(shuffled)
    n_a = int(round(frac_a * len(shuffled)))
    locations_a, locations_b = sorted(shuffled[:n_a]), sorted(shuffled[n_a:])

    assert not (set(locations_a) & set(locations_b)), "location slabs must be disjoint"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "frac_a": frac_a,
        "locations_a": locations_a,
        "locations_b": locations_b,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.success(
        f"Wrote location split to {output_path}: {len(locations_a)} A-traps, {len(locations_b)} B-traps"
    )


if __name__ == "__main__":
    app()
