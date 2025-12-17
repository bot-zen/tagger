import logging
from pathlib import Path
from typing import Optional

import numpy as _np

from .. import utils as _utils
from . import postags as _postags

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


class PosTagsType(object):
    def __init__(self, feature_type: str = "ibk", data_root: Optional[Path] = None):
        self._feature_type = feature_type
        self._feature_names = None
        self._feature_length = None
        self._data_root = data_root or DATA_ROOT

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def feature_names(self):
        if self._feature_names is None:
            self._feature_names = get_ts(self.feature_type, data_root=self._data_root) + [padding_tag]
        return self._feature_names

    @property
    def feature_length(self):
        return len(self.feature_names)


padding_tag = '_padding_'

stts_ibk_floc = DATA_ROOT / "stts_ibk.txt"
stts_1999_floc = DATA_ROOT / "stts_1999.txt"
postwita_floc = DATA_ROOT / "tagset-postwita.txt"

_stts_ibk = None
_stts_1999 = None
_stts_ibk_used = None
_stts_1999_used = None
_postwita = None


def _get_stts_ibk(fileloc=stts_ibk_floc):
    global _stts_ibk
    if _stts_ibk is None:
        _stts_ibk = sorted([line.strip() for line in Path(fileloc).read_text().splitlines() if line])
    return _stts_ibk


def _get_stts_1999(fileloc=stts_1999_floc):
    global _stts_1999
    if _stts_1999 is None:
        _stts_1999 = sorted([line.strip() for line in Path(fileloc).read_text().splitlines() if line])
    return _stts_1999


def _get_postwita(fileloc=postwita_floc):
    global _postwita
    if _postwita is None:
        _postwita = sorted([line.strip() for line in Path(fileloc).read_text().splitlines() if line])
    return _postwita


def _get_stts_ibk_used():
    global _stts_ibk_used
    if _stts_ibk_used is None:
        _, yc = _utils.load_tagged_files(_utils.cmc_tggd_flocs)
        _, yw = _utils.load_tagged_files(_utils.web_tggd_flocs)
        _stts_ibk_used = sorted(set([pos for elem in _utils.filter_elems(yc + yw) for line in elem for pos in line.split('\n')]))
    return _stts_ibk_used


def _get_stts_1999_used():
    global _stts_1999_used
    if _stts_1999_used is None:
        _, tiger_tggs = _utils.load_tiger_vrt_file()
        _stts_1999_used = sorted(set([pos for elem in tiger_tggs for line in elem for pos in line.split('\n')]))
    return _stts_1999_used


def get_ts(type, data_root: Optional[Path] = None):
    """
    Return type of stts.

    Args:
        type: ibk, ibk_used, 1999, 1999_used

    Returns:
        list of postags.
    """
    if data_root:
        _override_paths(data_root)
    return getattr(_postags, '_get_'+type)()


def _override_paths(data_root: Path):
    global stts_ibk_floc, stts_1999_floc, postwita_floc, _stts_ibk, _stts_1999, _postwita
    stts_ibk_floc = Path(data_root) / "stts_ibk.txt"
    stts_1999_floc = Path(data_root) / "stts_1999.txt"
    postwita_floc = Path(data_root) / "tagset-postwita.txt"
    _stts_ibk = None
    _stts_1999 = None
    _postwita = None


def encode(postags, postagstype):
    """
    Converts postags to a one-hot matrix representation.
    """
    matrix = _np.zeros((len(postags), postagstype.feature_length)).astype(bool)
    for rdx, postag in enumerate(postags):
        if postag in postagstype.feature_names:
            index = postagstype.feature_names.index(postag)
        else:
            index = _np.random.randint(1, postagstype.feature_length) - 1
            logger.info("Warning: encoded %s as %s", postag, postagstype.feature_names[index])
        matrix[rdx, index] = True
    return matrix


def decode_oh(onehots, postagstype=None):
    """
    Convert one-hot representations to POS tags.
    """
    postags = list()
    if postagstype is None:
        postagstype = PosTagsType()
    for onehot in onehots:
        if onehot >= postagstype.feature_length:
            postags.append('UNKNOWN')
        else:
            postags.append(postagstype.feature_names[onehot])
    return postags


def decode_m(matrix, postagstype=None):
    """
    Convert matrix to the corresponding postags representation.
    """
    postags = list()
    if postagstype is None:
        postagstype = PosTagsType()
    for row in matrix:
        postags.append(postagstype.feature_names[row.tolist().index(True)])
    return postags
