import torch

from src.preprocesses.corpus import Corpus


class NMTInsDataEnd:
    def __init__(self, vocab, corpus1: Corpus, corpus2: Corpus):
        self.vocab = vocab
        self.corpus1 = corpus1
        self.corpus2 = corpus2
        self.data = None
        self.label = None

    def build(self):
        """
        Build training data entries from corpus.
        TODO: now the corpuses are in a delicate state for the conveniency
          of leveraging the nmt official dataset. when manually construct
          the func/block source, try to create the positive/negative samples
          with a more general method
        """
        if self.corpus1.n_docs != self.corpus2.n_docs:
            raise NotImplementedError(
                f'Data end for customized data is not supported yet! '
                f'Now the corpus for nmt-inspired is constructed from '
                f'the official data, which is in the format that each corpus '
                f'has the same size and the functions in separate corpus are '
                f'identified by their index.')

        self.data, self.label = [], []
        for i, (doc1, doc2) in enumerate(
                zip(self.corpus1.idx2doc, self.corpus2.idx2doc)):
            self.data.append([
                torch.tensor(self.corpus1.idx2ins[i]),
                torch.tensor(self.corpus2.idx2ins[i])
            ])
            self.label.append(1 if doc1 == doc2 else 0)
