import rx

import src.preprocesses.preprocess as pp
from src.database.repository import Repository
from src.models.builder import ModelBuilder, WholeModelKeeper, ModelKeeper
from src.preprocesses.corpus import CorpusMaker, Corpus
from src.preprocesses.vocab import AsmVocab
from src.training.args.train_args import BinArgs
from src.utils.auto_json import auto_json


@auto_json
class CorpusRecipe:
    def __init__(self, min_fun_len=5):
        self.min_fun_len = min_fun_len

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


@ModelBuilder.register(Corpus)
class CorpusBuilder(ModelBuilder):
    vocab: AsmVocab

    def __init__(self, data, recipe, vocab):
        self.data: BinArgs = data
        self.recipe: CorpusRecipe = recipe
        self.vocab: AsmVocab = vocab

    def build(self):
        data = pp.BinBag(Repository.find_prog(self.data))
        recipe = map(lambda x: x.as_map(), self.recipe.make(self.vocab))
        return rx.just(data).pipe(*recipe).run().corpus.run()


@ModelKeeper.register(Corpus)
class CorpusKeeper(WholeModelKeeper):
    pass
