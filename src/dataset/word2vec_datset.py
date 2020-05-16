import torch

from src.dataset.dataset import ReloadableDataset
from src.dataset.pvdm_dataset import make_one_doc, SubSampleDataMaker
from src.preprocesses.vocab import AsmVocab


class Word2VecDatasetBuilder:

    def __init__(self, vocab: AsmVocab):
        self.vocab = vocab
        self.data = []

    def build(self, corpus, window, ss=None):
        self.data = []
        # convert each func into a sequence of training data
        for func_id, stmts in enumerate(corpus.idx2ins):
            for word, context in make_one_doc(stmts, window):
                self.data.append((
                    torch.tensor(func_id),
                    torch.tensor(word),
                    torch.tensor(context)
                ))

        maker = SubSampleDataMaker(
            self.data, self.vocab.sub_sample_ratio(ss))
        return ReloadableDataset(maker)
