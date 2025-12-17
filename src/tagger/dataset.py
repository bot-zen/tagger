import logging
from collections import OrderedDict
from typing import Dict, Iterator, List

# import pandas as pd  # type: ignore

from conllu import parse
from conllu.models import TokenList
from gensim.models.fasttext import FastText  # type: ignore
from smart_open import open  # type: ignore

logger = logging.getLogger(__name__)


class Dataset(object):
    """
    Represents a Dataset.
    """
    def __init__(self, url: str, in_memory: bool = True):
        """
        url: The URL of the data
        in_memory: use in-memory operations for small datasets and
            optimize memory footprint for large ones.
        """
        self.url = url
        self.in_memory = in_memory

        self._corpus: Dict[str, list] = None
        self._sentences: List[List[str]] = None

    @property
    def corpus(self):
        """
        """
        # FIXME: DOC

        def populate_corpus() -> None:
            logger.debug("")
            # TODO: Make this more generic
            """
            Populate a UD_Italian-Valico corpus.
            """
            with open(self.url, "r") as f:
                data_conllu = f.read()

                data_docs: Dict[str, list] = OrderedDict()
                for sentence in parse(data_conllu):
                    # logger.debug(sentence)
                    sentence_upos = [token for token in
                                     sentence.filter(id=lambda x: isinstance(x,
                                                                             int))]

                    metadata = sentence.metadata
                    # logger.debug(metadata)
                    if "sent_id" in metadata:
                        """
                        a-bb_xx-c
                        a is the number of the text (from 1 to 36);
                        bb is the number of the sentence in the text;
                        xx is the ISO code for the learner's native language;
                        c is the learner's year of study of Italian.
                        """
                        a_bb, xx_c = metadata["sent_id"].split('_')
                        text_id, sentece_num = a_bb.split('-')
                        data_docs.setdefault(text_id, []).append(sentence_upos)
                self._corpus = data_docs

        if self.in_memory and self._corpus is None:
            populate_corpus()
            # logger.debug(self._corpus)
        elif not self.in_memory:
            # TODO: implement not in_memory logic for large datasets
            raise NotImplementedError

        return self._corpus

    def senteces(self) -> List[List[str]]:
        if self._sentences is None:
            self._sentences = list()
            for doc_id in self.corpus:
                # logger.debug(doc)
                for sentence in self.corpus[doc_id]:
                    # logger.debug(sentence)
                    self._sentences.append([token["form"] for token in sentence])
        return self._sentences

    def __iter__(self) -> Iterator[TokenList]:
        logger.debug(self.corpus)
        for doc_id in self.corpus:
            logger.debug("###")
            logger.debug(doc_id + ":" + str(self.corpus[doc_id]))
            yield self.corpus[doc_id]
            logger.debug("###-###")


def main():
    FASTTEXT_VECTORSIZE = 100
    from os.path import expanduser
    home = expanduser("~")
    ds = Dataset(home + "/phd/src/data/UD_Italian-Valico/it_valico-ud-test.conllu")

    logger.debug(ds.senteces())

    ft_model = FastText(vector_size=FASTTEXT_VECTORSIZE)
    ft_model.build_vocab(corpus_iterable=ds.senteces())
    ft_model.train(corpus_iterable=ds.senteces(),
                   total_examples=len(ds.senteces()), epochs=10)  # train
    for x in ds:
        print(x)


if __name__ == "__main__":
    logger.debug("Foo bar!")
    main()
