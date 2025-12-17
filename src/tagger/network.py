from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import utils

logger = logging.getLogger(__name__)


def build_model(
    input_dim: int = 2280,
    output_dim: Optional[int] = None,
    lstm_output_dim: int = 512,
    dropout: float = 0.5,
):
    """
    Build a bi-directional LSTM tagger with subword embeddings.
    """
    from .representation.postags import PosTagsType

    if output_dim is None:
        postagstype = PosTagsType(feature_type="ibk")
        output_dim = postagstype.feature_length

    model = keras.Sequential(
        [
            layers.Masking(mask_value=0.0, input_shape=(None, input_dim)),
            layers.Bidirectional(
                layers.LSTM(lstm_output_dim, return_sequences=True, stateful=False)
            ),
            layers.Dropout(dropout),
            layers.TimeDistributed(layers.Dense(output_dim, activation="softmax")),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    logger.info("Built model with input_dim=%s, output_dim=%s", input_dim, output_dim)
    return model


def train_nn(
    model,
    toks,
    tags,
    batch_size: int = 10,
    epochs: int = 10,
    verbose: int = 1,
    postagstype=None,
    validation_split: float = 0.0,
    w2v_small_model=None,
    w2v_big_model=None,
    settings=None,
):
    logger.info("Training model...")
    x, _, _, _ = utils.training_data_tagging(
        toks, tags, postagstype=postagstype, w2v_small_model=w2v_small_model, w2v_big_model=w2v_big_model, settings=settings
    )
    xlens = sorted(set([len(_x) for _x in x]))
    for xlenid, xlen in enumerate(xlens):
        _x, _y, _, _ = utils.training_data_tagging(
            toks,
            tags,
            seqlen=int(xlen),
            postagstype=postagstype,
            w2v_small_model=w2v_small_model,
            w2v_big_model=w2v_big_model,
            settings=settings,
        )
        logger.info("seq_len=%i (%i/%i), samples=%i", xlen, xlenid + 1, len(xlens), len(_x))
        use_batch_size = batch_size if len(_x) >= batch_size else 1

        if xlen > 1:
            model.fit(
                np.array(_x),
                np.array(_y),
                batch_size=use_batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_split=validation_split,
            )
    logger.info("Training finished.")


def eval_nn(model, toks, tags, verbose: int = 1, postagstype=None, w2v_small_model=None, w2v_big_model=None, settings=None):
    retres = []
    xtst, _, _, _ = utils.training_data_tagging(
        toks, tags, postagstype=postagstype, w2v_small_model=w2v_small_model, w2v_big_model=w2v_big_model, settings=settings
    )
    for xtstlen in sorted(list(set([len(_x) for _x in xtst]))):
        xtstseq, ytstseq, _, _ = utils.training_data_tagging(
            toks,
            tags,
            seqlen=xtstlen,
            postagstype=postagstype,
            w2v_small_model=w2v_small_model,
            w2v_big_model=w2v_big_model,
            settings=settings,
        )
        res = model.evaluate(
            np.array(xtstseq),
            np.array(ytstseq),
            batch_size=len(xtstseq),
            verbose=verbose,
        )
        retres.append((xtstlen, len(xtstseq), res))
    return retres


def compact_res(res):
    return sum([r[2][1] for r in res]) / len(res)


def qgrid_search():
    lstm_output_dims = [128, 256, 512, 1024]
    dropouts = [0.1, 0.25, 0.5, 0.75]
    nb_epochs = [5, 10, 20]

    toks, tags = utils.load_tagged_files(utils.all_tggd_flocs)
    toks_trial, tags_trial = utils.load_tagged_files(utils.all_trial_tggd_flocs)

    retres = []
    for lstm_output_dim in lstm_output_dims:
        for dropout in dropouts:
            model = build_model(lstm_output_dim=lstm_output_dim, dropout=dropout)
            model.save_weights('/tmp/tmpmodel.hdf5', overwrite=True)
            for nb_epoch in nb_epochs:
                train_nn(model, toks, tags, epochs=nb_epoch, verbose=0)
                res = eval_nn(model, toks_trial, tags_trial, verbose=0)
                print(
                    "lstm_od:%i, drpt:%0.2f, nb_epc:%i, acc:%f"
                    % (lstm_output_dim, dropout, nb_epoch, compact_res(res))
                )
                retres.append((lstm_output_dim, dropout, nb_epoch, compact_res(res)))
                model.load_weights('/tmp/tmpmodel.hdf5')
    return retres
