from src.utils.auto_json import auto_json


@auto_json
class GENNArgs:
    def __init__(self, n_emb=None, n_negs=None, no_hdn=None,
                 ss=None, window=None, requires=None):
        # network configs
        self.n_emb = n_emb
        self.n_negs = n_negs
        self.no_hdn = no_hdn

        # dataset configs
        self.ss = ss
        self.window = window
        self.requires = requires
