# External Yield-Surface Datasets

This extension prepares two external CPFEM datasets for the existing invariant
generator without changing its training, invariant, sparsification, or symbolic
regression code. The adapters write the same HDF contract used by the original
datasets:

```text
stress: [N, 6]
component order: [s11, s22, s33, s23, s13, s12]
stress format: voigt_3d
```

Downloaded source archives are stored under `data/external/raw/` and are
ignored by Git. The small prepared HDF inputs are included with the project so
cluster runs do not depend on external repository availability. Each HDF also
contains source and preparation provenance metadata.

## DX56D deep-drawing steel

Source: Fraunhofer Fordatis, “Data for Machine learning-based sampling of
virtual experiments within the full stress state,” DOI
`10.24406/fordatis/225.2`.

Landing page:
<https://fordatis.fraunhofer.de/handle/fordatis/292.2>

The archive contains several independently sampled CPFEM descriptions of the
same initial yield surface. The default adapter uses the 402-point Miller
sampling because it provides a deterministic full-stress reference set.

The pipeline-ready `data/dx56d_miller.hdf` is included with the project. To
reproduce it from the original archive:

```bash
uv run python scripts/prepare_external_yield_data.py dx56d --download
```

To use a different published sampling:

```bash
uv run python scripts/prepare_external_yield_data.py dx56d \
  --source data/external/raw/dx56d_data.zip \
  --sampling active_learning_1 \
  --output data/dx56d_active_learning_1.hdf
```

Available sampling names are `miller`, `active_learning_1` through
`active_learning_5`, and `random_1` through `random_5`.

Run the unchanged adaptive pipeline:

```bash
uv run python scripts/adaptive_encoder_pipeline.py \
  --config configs/adaptive_encoder_dx56d.toml
```

## Single-crystal gold

Source: Zenodo, “Micromechanical Simulation Dataset of Single-Crystal Gold
under Multiaxial Loading Using Crystal Plasticity FEM,” DOI
`10.5281/zenodo.21981226`, CC BY 4.0.

Landing page: <https://zenodo.org/records/21981226>

This source contains 295 stress/plastic-strain histories rather than explicit
yield points. The adapter defines yield at a configurable J2-equivalent plastic
strain and linearly interpolates the six stress components at that threshold.
The default is 0.002 (0.2%). A path that never reaches the threshold is omitted
and recorded in the HDF provenance metadata. In the published file, this
normally removes the hydrostatic loading path, which does not activate
pressure-insensitive crystal slip.

The pipeline-ready `data/single_crystal_gold_ep002.hdf` is included with the
project. To reproduce it from the original JSON:

```bash
uv run python scripts/prepare_external_yield_data.py gold --download
```

To study sensitivity to the operational yield definition, create another file
with a different threshold:

```bash
uv run python scripts/prepare_external_yield_data.py gold \
  --source data/external/raw/single_crystal_gold.json \
  --plastic-strain-threshold 0.001 \
  --output data/single_crystal_gold_ep001.hdf
```

Run the unchanged adaptive pipeline:

```bash
uv run python scripts/adaptive_encoder_pipeline.py \
  --config configs/adaptive_encoder_single_crystal_gold.toml
```

## Recommended first runs

Stage 1 is the automatic encoder-dimension search; it does not require choosing
the dimension manually. The supplied configs use `n_min = 1` and `n_max = 0`,
where zero means that the search may continue through all 13 invariant inputs.
Running only Stage 1 is useful when you want to inspect this automatic choice
before committing to the longer sparsification and PySR stages:

```bash
uv run python scripts/adaptive_encoder_pipeline.py \
  --config configs/adaptive_encoder_dx56d.toml \
  --stage stage1

uv run python scripts/adaptive_encoder_pipeline.py \
  --config configs/adaptive_encoder_single_crystal_gold.toml \
  --stage stage1
```

The supplied configurations intentionally mirror the current rotated-hill
adaptive setup. They are baselines, not claims that its PSD constraints,
sparsification strength, or acceptance threshold are optimal for either
material. Tune those only after the baseline results are saved.

## Scientific caveats

- DX56D points use the authors' fixed specific-plastic-work yield definition.
- Gold points use the adapter's explicit equivalent-plastic-strain threshold.
- A random point split tests interpolation across loading directions. It does
  not by itself establish transfer to another material or hardening state.
- Do not combine different yield thresholds or hardening states in one HDF
  surface: the current pipeline assumes all unscaled rows satisfy `f(stress)=1`.
