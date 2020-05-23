import torch

from src.models.builder import ModelBuilder
from src.preprocesses.corpus import Corpus
from src.preprocesses.dataset.dataset import ReloadableDataset
from src.preprocesses.dataset.pvdm_dataset import make_one_doc, \
    SubSampleDataMaker
from src.preprocesses.vocab import AsmVocab


class Word2VecDataset(ReloadableDataset):
    pass


@ModelBuilder.register(Word2VecDataset)
class Word2VecDatasetBuilder(ModelBuilder):
    vocab: AsmVocab
    corpus: Corpus

    def __init__(self, vocab, corpus, window, ss=None):
        self.vocab = vocab
        self.corpus = corpus
        self.window = window
        self.ss = ss
        self.data = []

    def build(self):
        self.data = []
        # convert each func into a sequence of training data
        for func_id, stmts in enumerate(self.corpus.idx2ins):
            per_doc = []
            for word, context in make_one_doc(stmts, self.window):
                per_doc.append((
                    torch.tensor(func_id),
                    torch.tensor(word),
                    torch.tensor(context)
                ))
            if per_doc:
                self.data.append(per_doc)

        maker = SubSampleDataMaker(
            self.data, self.vocab.sub_sample_ratio(self.ss))
        return Word2VecDataset(maker)
