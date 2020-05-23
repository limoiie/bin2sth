import torch

from src.models.builder import ModelBuilder
from src.preprocesses.corpus import Corpus
from src.preprocesses.dataset.dataset import SupervisedDataset
from src.preprocesses.vocab import AsmVocab


class SAFELegacyDataset(SupervisedDataset):
    """
    This is the dataset used to support the orginal dataset from the
    original implementation of safe, where the two corpus have already
    been carefully constructed so that two functions at the same
    indice are a negative pair or positive pair.
    As for the dataset created by me, it should be created from the scratch,
    that is, i should construct the negative/positive sample and adjust their
    percentages
    """
    pass


@ModelBuilder.register(SAFELegacyDataset)
class SAFELegacyDatasetBuilder(ModelBuilder):
    vocab: AsmVocab
    corpus1: Corpus
    corpus2: Corpus

    def __init__(self, vocab, corpus1, corpus2):
        self.vocab = vocab
        self.corpus1 = corpus1
        self.corpus2 = corpus2
        self.data = None
        self.label = None

    def build(self):
        if self.corpus1.n_docs != self.corpus2.n_docs:
            raise NotImplementedError(
                f'The two corpus are not in the same shape! '
                f'Hence, these two corpus must not be the '
                f'legacy corpus. You may want to use SAFEDatasetBuilder!')

        self.data, self.label = [], []
        for i, (doc1, doc2) in enumerate(
                zip(self.corpus1.idx2doc, self.corpus2.idx2doc)):
            self.data.append([
                torch.tensor(self.corpus1.idx2ins[i]),
                torch.tensor(self.corpus2.idx2ins[i])
            ])
            self.label.append(float(1. if doc1 == doc2 else 0.))
        self.label = torch.tensor(self.label, dtype=torch.float32)
        return SAFELegacyDataset(self.data, self.label)
