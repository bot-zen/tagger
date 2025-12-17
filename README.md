# bot.zet PoS tagger

Bi-directional LSTM tagger with subword embeddings for EmpiriST/PostWITA datasets. The project now targets Python 3.12+, modern TensorFlow/Keras, and uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Quickstart

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Configure embedding paths in `tagger.ini` or via env vars:
   - `TAGGER_W2V_SMALL=/path/to/small.vec`
   - `TAGGER_W2V_BIG=/path/to/big.vec`
3. Train:
   ```bash
   uv run tagger train --task postwita --config tagger.ini --epochs 5
   ```
4. Predict:
   ```bash
   uv run tagger predict --model-path artifacts/tagger.keras --output-ext .pred
   ```

## Project layout

- `src/tagger/` package with config handling, data utilities, and the BiLSTM model.
- `tagger.ini` optional config; env vars prefixed with `TAGGER_` override values.
- `pyproject.toml` defines runtime/dev dependencies; `uv.lock` produced by `uv sync`.

## Notes

- Default data root is `<repo>/data`; override with `--data-root` or `TAGGER_DATA_ROOT`.
- Saved models use Keras’ `.keras` format containing architecture + weights.
