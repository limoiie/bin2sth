from src.models.modules.word2vec import Word2Vec
from src.models.pvdm import CBowPVDM
from src.models.builder import ModelKeeper


@ModelKeeper.register(CBowPVDM)
class CBowPVDMKeeper(ModelKeeper):
    def from_state(self, state):
        model = CBowPVDM()
        keeper = ModelKeeper.instance(Word2Vec)
        model.args = state['args']
        model.vocab_size = state['vocab_size']
        model.corpus_size = state['corpus_size']
        model.neg_samp_dist = state['neg_samp_dist']
        model.init_param()
        model.f2v.load_state_dict(state['f2v'])
        model.w2v = keeper.from_state(state['w2v'])
        return model

    def state(self, model: CBowPVDM):
        keeper = ModelKeeper.instance(Word2Vec)
        return {
            'args': model.args,
            'vocab_size': model.vocab_size,
            'corpus_size': model.corpus_size,
            'f2v': model.f2v.state_dict(),
            'neg_samp_dist': model.neg_samp_dist,
            'w2v': keeper.state(model.w2v)
        }
