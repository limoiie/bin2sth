from dataclasses import dataclass

import rx

import src.preprocesses.preprocess as pp
from src.database.repository import Repository
from src.models.builder import ModelBuilder, WholeModelKeeper, ModelKeeper
from src.preprocesses.corpus import CorpusMaker, Corpus
from src.preprocesses.vocab import AsmVocab
from src.training.args.train_args import BinArgs
from src.utils.auto_json import auto_json


class CorpusRecipeBase:
    def make(self, vocab):
        raise NotImplementedError()


@auto_json
@dataclass
class CorpusRecipe(CorpusRecipeBase):
    min_fun_len: int = 5

    def make(self, vocab):
        return [
            pp.PpFullLabel(),
            pp.PpMergeBlocks(),
            pp.PpFilterFunc(minlen=self.min_fun_len),
            pp.PpTokenizer(),
            pp.PpOneHotEncoder(vocab),
            pp.PpMergeFuncs(),
            pp.PpCorpus(CorpusMaker())
        ]


@auto_json
@dataclass
class CorpusSeqRecipe(CorpusRecipeBase):
    min_fun_len: int = 5
    max_seq_len: int = 150

    def make(self, vocab):
        return [
            pp.PpFullLabel(),
            pp.PpMergeBlocks(),
            pp.PpFilterFunc(minlen=self.min_fun_len),
            pp.PpOneHotEncoder(vocab),
            pp.PpPadding(self.max_seq_len, 0),
            pp.PpMergeFuncs(),
            pp.PpCorpus(CorpusMaker())
        ]


@ModelBuilder.register(Corpus)
class CorpusBuilder(ModelBuilder):
    vocab: AsmVocab

    def __init__(self, data, recipe, vocab):
        self.data: BinArgs = data
        self.recipe: CorpusRecipeBase = recipe
        self.vocab: AsmVocab = vocab

    def build(self):
        data = pp.BinBag(Repository.find_prog(self.data))
        recipe = map(lambda x: x.as_map(), self.recipe.make(self.vocab))
        return rx.just(data).pipe(*recipe).run().corpus.run()


@ModelKeeper.register(Corpus)
class CorpusKeeper(WholeModelKeeper):
    pass
