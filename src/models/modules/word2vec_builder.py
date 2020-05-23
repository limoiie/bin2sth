from src.models.modules.word2vec import Word2Vec
from src.models.builder import ModelKeeper


@ModelKeeper.register(Word2Vec)
class Word2VecKeeper(ModelKeeper):
    def from_state(self, state):
        model = Word2Vec()
        model.cfg = state['cfg']
        model.vocab_size = state['vocab_size']
        model.init_param()
        model.idx2vec.load_state_dict(state['idx2vec'])
        model.idx2hdn.load_state_dict(state['idx2hdn'])
        if model.cfg.no_hdn:
            model.idx2hdn = model.idx2vec
        return model

    def state(self, model: Word2Vec):
        return {
            'vocab_size': model.vocab_size,
            'cfg': model.cfg,
            'idx2vec': model.idx2vec.state_dict(),
            'idx2hdn': model.idx2hdn.state_dict()
        }
