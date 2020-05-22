import rx

import src.preprocesses.preprocess as pp
from src.database.factory import Repository
from src.preprocesses.builder import ModelBuilder, ModelKeeper, WholeModelKeeper
from src.preprocesses.vocab import AsmVocab
from src.training.args.train_args import BinArgs
from src.utils.auto_json import auto_json


@auto_json
class VocabRecipe:
    def __init__(self, min_word_freq=0):
        self.min_word_freq = min_word_freq

    def make(self):
        return [
            pp.PpMergeBlocks(),
            pp.PpTokenizer(),
            pp.PpMergeFuncs(),
            pp.PpOutStmts(),
            pp.PpVocab(min_freq=self.min_word_freq)
        ]


@ModelBuilder.register(AsmVocab)
class VocabBuilder(ModelBuilder):

    def __init__(self, data, recipe):
        self.data: BinArgs = data
        self.recipe: VocabRecipe = recipe

    def build(self) -> AsmVocab:
        data = pp.BinBag(Repository.find_prog(self.data))
        recipe = map(lambda x: x.as_map(), self.recipe.make())
        return rx.just(data).pipe(*recipe).run().vocab.run()


@ModelKeeper.register(AsmVocab)
class VocabKeeper(WholeModelKeeper):
    pass
