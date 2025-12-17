import string as _string

import numpy as _np


"""
Quasi-One Hot representation of characters.
"""
# this is what will mark a 'token boundary'
new_token_char = '\n'

# one hot dims for common classes
_string_feature_names = ['ascii_lowercase', 'digits', 'punctuation',
                         'whitespace']

# extra dims: is-uppercase, is-digit, is-punctutation, is-whitespace,
# new token char, unknown char
_other_feature_names = ['is_uppercase', 'is_digit', 'is_punctuation',
                        'is_whitespace', 'is_new_token_char', 'unknown']

_feature_string = ''.join(
    [getattr(_string, feature_name) for feature_name in
        _string_feature_names] +
    [' ' for _ in _other_feature_names])

feature_length = len(_feature_string)

feature_names = (["string."+_str_feat_name+"-"+_str
                  for _str_feat_name in _string_feature_names
                  for _str in getattr(_string, _str_feat_name)]
                 +
                 _other_feature_names)


def _encode_char(char):
    """
    Returns the quasi-one-hot index vector for a character.
        - alpha-characters are mapped to lower-case one-hot + 'is-uppercase'
        - digits are mapped to one-hot + 'is-digit'
        - punctuation marks are mapped to one-hot + 'is-punctuation'
        - whitespace (ecxept '\n') characters are mapped to one-hot +
        'is-whitespace'
        - unknowns have their own one-hot
        * '\n' is treated as new-token-character

    Args:
        char: string
        Character to index

    Returns:
        index : np.ndarray, dtype=bool, shape=(~106,1)
            Index vector of character
    """
    # make sure to process a single character
    if len(char) > 1:
        raise ValueError('can only cope with a single char.')

    index = _np.zeros((1, feature_length)).astype(bool)

    if (char.lower() in _feature_string[0:-len(_other_feature_names)]
            or char in _string.ascii_uppercase):
        index[0, _feature_string.index(char.lower())] = True
    else:
        index[0, feature_length-1] = True

    if char in _string.ascii_uppercase:
        index[0, feature_length-6] = True
    elif char in _string.digits:
        index[0, feature_length-5] = True
    elif char in _string.punctuation:
        index[0, feature_length-4] = True
    elif char in _string.whitespace or char == new_token_char:
        if char == new_token_char:
            index[0, feature_length-2] = True
        else:
            index[0, feature_length-3] = True

    return index


def _decode_matrix(matrix):
    """
    Inverse of _encode().

    Args:
        index: np.ndarray, dtype=bool, shape=(~100,1)
            Index vector of character

    Returns:
        char: string
            Character of index

    """
    chars = ''
    matrix = matrix.T
    for row in matrix:
        if row[feature_length-1]:
            char = '?'
        elif row[feature_length-2]:
            char = '\n'
        else:
            char = _feature_string[row.tolist().index(True)]
            if row[feature_length-6]:
                char = char.upper()
        chars = ''.join(chars+char)
    return chars


def encode(chars):
    """
    Converts chars to the quasi-one-hot matrix representation.

    Returns:
        quasi one-hot matrix of chars
        np.ndarray, dtype=bool, shape=(len(chars),feature_length)
    """
    matrix = _np.zeros((len(chars), feature_length)).astype(bool)
    for rdx in range(len(chars)):
        matrix[rdx, ] = _encode_char(chars[rdx])
    return matrix.T


def decode(matrix):
    """
    Convert matrix to the corresponding character representation.

    Returns:
        chars representation of matrix.
    """
    return _decode_matrix(matrix)
