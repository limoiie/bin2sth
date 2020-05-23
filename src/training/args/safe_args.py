from src.utils.auto_json import auto_json


@auto_json
class SAFEArgs:
    def __init__(self, n_emb=None, n_rnn_state=None, rnn_depth=None,
                 max_seq_len=None, atten_hops=None, atten_depth=None,
                 n_dense_layer=None):
        # network configs
        self.n_emb = n_emb  # dimension of the function embedding

        # RNN PARAMETERS, these parameters are only used for RNN model.
        self.n_rnn_state = n_rnn_state  # dimesion of the rnn state
        self.rnn_depth = rnn_depth  # depth of the rnn
        self.max_seq_len = max_seq_len  # number of instructions

        # ATTENTION PARAMETERS
        self.atten_hops = atten_hops
        self.atten_depth = atten_depth

        # RNN SINGLE PARAMETER
        self.n_dense_layer = n_dense_layer
