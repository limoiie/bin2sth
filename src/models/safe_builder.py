from src.models.builder import ModelKeeper
from src.models.safe import SAFE


@ModelKeeper.register(SAFE)
class SAFEKeeper(ModelKeeper):
    def from_state(self, state):
        model = SAFE(state['conf'], vocab_size=state['vocab_size'])
        model.load_state_dict(state['state'])
        return model

    def state(self, model: SAFE):
        return {
            'conf': model.conf,
            'vocab_size': model.vocab_size,
            'state': model.state_dict()
        }
