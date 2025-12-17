from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from tensorflow import keras

from .config import Settings, load_settings
from .logging_utils import configure_logging
from .network import build_model, eval_nn, train_nn
from .representation.postags import PosTagsType
from .utils import (
    default_data_paths,
    process_test_data_tagging,
    training_data_tagging,
    load_tagged_files,
)


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run the PoS tagger.")
    parser.add_argument("--config", type=str, help="Path to tagger.toml/ini for defaults.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", choices=["postwita", "empirist"], help="Task preset (defaults to config).")
    common.add_argument("--data-root", type=str, help="Override data root directory.")
    common.add_argument("--w2v-small", type=str, help="Path to small Word2Vec file.")
    common.add_argument("--w2v-big", type=str, help="Path to big Word2Vec file.")
    common.add_argument("--feature-type", type=str, help="Tagset feature type (e.g., postwita, ibk).")
    common.add_argument("--log-level", type=str, help="Logging level.")

    train = sub.add_parser("train", parents=[common], help="Train the tagger.")
    train.add_argument("--epochs", type=int, help="Number of epochs.")
    train.add_argument("--batch-size", type=int, help="Batch size.")
    train.add_argument("--dropout", type=float, help="Dropout rate.")
    train.add_argument("--lstm-units", type=int, help="Hidden size for BiLSTM.")
    train.add_argument("--validation-split", type=float, help="Validation split fraction.")
    train.add_argument("--sequence-length", type=int, help="Optional fixed sequence length.")
    train.add_argument("--output-dir", type=str, help="Where to save models/artifacts.")
    train.add_argument("--model-filename", type=str, default="tagger.keras", help="Filename for the saved model.")

    predict = sub.add_parser("predict", parents=[common], help="Predict PoS tags for tokenized files.")
    predict.add_argument("--model-path", required=True, type=str, help="Path to a saved Keras model (.keras).")
    predict.add_argument("--inputs", nargs="+", type=str, help="Tokenized test files; defaults to task presets.")
    predict.add_argument("--output-ext", type=str, default=".done", help="Extension suffix for prediction outputs.")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "task": args.task,
        "data_root": Path(args.data_root) if args.data_root else None,
        "w2v_small_path": Path(args.w2v_small) if getattr(args, "w2v_small", None) else None,
        "w2v_big_path": Path(args.w2v_big) if getattr(args, "w2v_big", None) else None,
        "logging_level": args.log_level,
        "epochs": getattr(args, "epochs", None),
        "batch_size": getattr(args, "batch_size", None),
        "dropout": getattr(args, "dropout", None),
        "lstm_units": getattr(args, "lstm_units", None),
        "validation_split": getattr(args, "validation_split", None),
        "sequence_length": getattr(args, "sequence_length", None),
        "output_dir": Path(args.output_dir) if getattr(args, "output_dir", None) else None,
    }
    settings = load_settings(config_file=args.config, overrides=overrides)
    configure_logging(settings.logging_level)
    logger.debug("Loaded settings: %s", settings.to_dict())

    if args.command == "train":
        run_train(args, settings)
    elif args.command == "predict":
        run_predict(args, settings)
    else:
        raise SystemExit(f"Unknown command {args.command}")
    return 0


def resolve_feature_type(args, settings: Settings) -> str:
    if getattr(args, "feature_type", None):
        return args.feature_type
    if settings.task == "postwita":
        return "postwita"
    return "ibk"


def run_train(args, settings: Settings):
    feature_type = resolve_feature_type(args, settings)
    paths = default_data_paths(settings.data_root)

    if settings.task == "postwita":
        train_flocs = paths["all_postwita_tggd_flocs"]
    else:
        train_flocs = paths["all_tggd_flocs"]
    toks, tags = load_tagged_files(train_flocs)

    postagstype = PosTagsType(feature_type=feature_type, data_root=settings.data_root)

    sample_x, _, _, _ = training_data_tagging(
        toks,
        tags,
        sample_size=1,
        seqlen=args.sequence_length,
        postagstype=postagstype,
        settings=settings,
    )
    if not sample_x or not sample_x[0]:
        raise RuntimeError("Could not derive sample input dimension; check training data.")
    input_dim = len(sample_x[0][0])

    model = build_model(
        input_dim=input_dim,
        output_dim=postagstype.feature_length,
        lstm_output_dim=settings.lstm_units,
        dropout=settings.dropout,
    )

    train_nn(
        model,
        toks,
        tags,
        batch_size=settings.batch_size,
        epochs=settings.epochs,
        verbose=1,
        validation_split=settings.validation_split,
        postagstype=postagstype,
        settings=settings,
    )

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.model_filename
    model.save(model_path)
    logger.info("Saved model to %s", model_path)

    if settings.task == "empirist":
        toks_trial, tags_trial = load_tagged_files(paths["all_trial_tggd_flocs"])
        res = eval_nn(model, toks_trial, tags_trial, postagstype=postagstype, settings=settings)
        logger.info("Trial accuracy: %.4f", (sum([r[2][1] for r in res]) / len(res)))


def run_predict(args, settings: Settings):
    model_path = Path(args.model_path)
    model = keras.models.load_model(model_path)
    feature_type = resolve_feature_type(args, settings)
    postagstype = PosTagsType(feature_type=feature_type, data_root=settings.data_root)

    paths = default_data_paths(settings.data_root)
    flocs = [Path(p) for p in (args.inputs or (paths["all_postwita_tst_flocs"] if settings.task == "postwita" else paths["all_tst_tokd_flocs"]))]

    process_test_data_tagging(model, postagstype, flocs, extension=args.output_ext, settings=settings)
    logger.info("Wrote predictions for %d file(s) with extension '%s'", len(flocs), args.output_ext)


if __name__ == "__main__":
    raise SystemExit(main())
