import json

from invariant_generator.config import Config
from invariant_generator.report import create_adaptive_analysis_notebook


def test_create_adaptive_analysis_notebook_writes_run_notebook(tmp_path):
    config = Config()
    config.train.results_dir = tmp_path / "results"
    config.adaptive.results_subdir = "adaptive_case"

    notebook_path = create_adaptive_analysis_notebook(
        config,
        config_path=tmp_path / "pipeline_config.toml",
    )

    assert notebook_path == tmp_path / "results" / "adaptive_case" / "analysis.ipynb"
    assert notebook_path.exists()

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    sources = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    assert "Stage 1: Adaptive n Sweep" in sources
    assert "Stage 2: Sparse Encoder" in sources
    assert "Stage 3: PySR" in sources
    assert "Synthetic Benchmark Files, If Present" in sources
