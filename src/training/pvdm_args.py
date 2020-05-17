from src.ida.as_json import AsJson
from src.utils.auto_json import auto_json


@auto_json
class PVDMArgs(AsJson):
    def __init__(self, n_emb, n_negs, no_hdn, ss, window):
        # network configs
        self.n_emb = n_emb
        self.n_negs = n_negs
        self.no_hdn = no_hdn

        # dataset configs
        self.ss = ss
        self.window = window
