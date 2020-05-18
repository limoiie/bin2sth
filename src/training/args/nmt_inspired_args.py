from src.utils.auto_json import auto_json


@auto_json
class NMTInspiredArgs:
    def __init__(self, n_emb, max_seq_len, n_lstm_hidden):
        # network configs
        self.n_emb = n_emb

        # dataset configs
        self.max_seq_len = max_seq_len
        self.n_lstm_hidden = n_lstm_hidden
