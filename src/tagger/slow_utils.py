from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from gensim.models.keyedvectors import KeyedVectors

from .config import Settings, load_settings

logger = logging.getLogger(__name__)


class W2V:
    """
    Lazy wrapper around gensim's Word2Vec that delays loading until needed.
    """

    def __init__(self, floc: Path):
        self.floc = Path(floc)
        self._data = None
        logger.debug("Configured Word2Vec file at %s", self.floc)

    @property
    def data(self):
        if self._data is None:
            self._data = KeyedVectors.load_word2vec_format(self.floc)
        return self._data


def build_embeddings(settings: Optional[Settings] = None) -> Tuple[Optional[W2V], Optional[W2V]]:
    cfg = settings or load_settings()
    if not cfg.w2v_small_path or not cfg.w2v_big_path:
        raise ValueError("Both w2v_small_path and w2v_big_path must be configured.")
    return W2V(cfg.w2v_small_path), W2V(cfg.w2v_big_path)


# Attempt to create lazily-loaded defaults; fine if configuration is missing.
try:
    _default_settings = load_settings()
    w2v_small, w2v_big = build_embeddings(_default_settings)
except Exception as exc:  # pragma: no cover - optional convenience
    logger.debug("Skipping eager embedding configuration: %s", exc)
    w2v_small, w2v_big = None, None
