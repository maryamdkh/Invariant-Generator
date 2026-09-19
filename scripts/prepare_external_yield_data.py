from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from invariant_generator.config import PROJECT_ROOT
from invariant_generator.external_data import (
    DX56D_SOURCE_URL,
    GOLD_SOURCE_URL,
    prepare_dx56d_dataset,
    prepare_gold_dataset,
)


RAW_DIR = PROJECT_ROOT / "data" / "external" / "raw"


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "invariant-generator/0.1"})
    print(f"[INFO] Downloading {url}")
    with urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)
    print(f"[INFO] Downloaded source: {destination}")
    return destination


def _resolve_source(
    source: str | None,
    *,
    download: bool,
    url: str,
    default_path: Path,
) -> Path:
    if source is not None and download:
        raise ValueError("Use either --source or --download, not both.")
    if source is not None:
        path = Path(source)
    elif download:
        path = _download(url, default_path)
    else:
        path = default_path
    if not path.exists():
        raise FileNotFoundError(
            f"Source data not found: {path}. Provide --source or use --download."
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert supported external yield-surface datasets to the existing "
            "six-component HDF input contract."
        )
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    dx56d = subparsers.add_parser("dx56d", help="Prepare the Fraunhofer DX56D dataset.")
    dx56d.add_argument("--source", help="Downloaded data.zip or extracted archive root.")
    dx56d.add_argument("--download", action="store_true", help="Download the official archive.")
    dx56d.add_argument(
        "--sampling",
        default="miller",
        choices=[
            "miller",
            *(f"active_learning_{index}" for index in range(1, 6)),
            *(f"random_{index}" for index in range(1, 6)),
        ],
        help="Published full-stress sampling table to convert.",
    )
    dx56d.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "dx56d_miller.hdf"),
        help="Output HDF path.",
    )

    gold = subparsers.add_parser("gold", help="Prepare the single-crystal gold dataset.")
    gold.add_argument("--source", help="Downloaded MiMeDat JSON file.")
    gold.add_argument("--download", action="store_true", help="Download the official JSON.")
    gold.add_argument(
        "--plastic-strain-threshold",
        type=float,
        default=0.002,
        help="J2-equivalent plastic strain used to define yield (default: 0.002).",
    )
    gold.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "single_crystal_gold_ep002.hdf"),
        help="Output HDF path.",
    )

    args = parser.parse_args()
    if args.dataset == "dx56d":
        source = _resolve_source(
            args.source,
            download=args.download,
            url=DX56D_SOURCE_URL,
            default_path=RAW_DIR / "dx56d_data.zip",
        )
        result = prepare_dx56d_dataset(
            source,
            args.output,
            sampling=args.sampling,
        )
    else:
        source = _resolve_source(
            args.source,
            download=args.download,
            url=GOLD_SOURCE_URL,
            default_path=RAW_DIR / "single_crystal_gold.json",
        )
        result = prepare_gold_dataset(
            source,
            args.output,
            plastic_strain_threshold=args.plastic_strain_threshold,
        )

    print(f"[INFO] Prepared dataset: {result.output_path}")
    print(f"[INFO] Yield points:     {result.n_stress_points}")
    skipped = result.metadata.get("skipped_trajectory_ids", [])
    if skipped:
        print(f"[INFO] Skipped paths:    {len(skipped)} (threshold not reached)")


if __name__ == "__main__":
    main()
