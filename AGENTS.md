# Repository Guidelines

## Project Structure & Module Organization
- `src/tagger/`: config handling (`config.py`), logging (`logging_utils.py`), data utilities (`utils.py`, `dataset.py`), model (`network.py`), encoders (`representation/`), CLI (`cli.py`).
- `tagger.ini` optional config; env vars `TAGGER_*` override it. Defaults assume data under `./data`.
- `pyproject.toml` + `uv.lock` (generate via `uv sync`) define deps; `test.py` benchmarks BLAS bindings.

## Build, Test, and Development Commands
- `uv sync` to install dependencies.
- `uv run tagger train --task postwita --config tagger.ini --epochs 5` to train; model saved under `artifacts/` unless overridden.
- `uv run tagger predict --model-path artifacts/tagger.keras --output-ext .pred` to tag tokenized files (uses test presets unless `--inputs` given).
- `python test.py` if you need to sanity-check NumPy/SciPy BLAS bindings.

## Coding Style & Naming Conventions
- Target Python 3.12+; 4-space indentation, snake_case for functions/variables, CamelCase for classes.
- Use type hints; avoid side effects at import time; prefer dependency injection (pass Settings instead of global state).
- Logging via `logging_utils.configure_logging`; avoid ad-hoc prints. Keep line lengths reasonable even though E501 is ignored in `setup.cfg`.
- Keep data roots/config paths parameterized; do not hardcode local absolute paths in code.

## Testing Guidelines
- Add `pytest` cases under `tests/` using tiny fixtures (avoid full corpora/large embeddings).
- Prefer deterministic inputs; mock or stub Word2Vec models where possible to keep tests fast.
- Run `uv run pytest` (or `python -m pytest`) before submitting; note any skipped tests and why.

## Commit & Pull Request Guidelines
- Use short imperative subjects (e.g., `Add uv workflow`, `Update keras`), ≤72 chars.
- Describe dataset sources, config/env overrides, and exact commands used (`uv run tagger ...`); note where artifacts are stored.
- Do not commit corpora, model weights, or generated outputs (`*.done`, `*.keras`); link to issues and include reproducibility notes over screenshots.
