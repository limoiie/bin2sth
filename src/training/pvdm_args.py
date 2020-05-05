import copy

from src.database.program_dao import BinArgs

from src.ida.as_json import AsJson, load_json_file


class ModelArgs(AsJson):
    def __init__(self, epochs, n_batch, n_emb, n_negs, init_lr,
                 no_hdn, ss, window):
        self.epochs = epochs
        self.n_batch = n_batch
        self.n_emb = n_emb
        self.n_negs = n_negs
        self.init_lr = init_lr
        self.no_hdn = no_hdn
        self.ss, self.window = ss, window


class TrainArgs(AsJson):
    rt: ModelArgs
    vocab: BinArgs
    corpus: BinArgs

    def __init__(self, model, rt, vocab: BinArgs, corpus: BinArgs):
        self.model = model
        self.rt = rt
        self.vocab = vocab
        self.corpus = corpus


class QueryArgs(TrainArgs):
    # TODO: consider take train args as a foreign key¬
    rt: ModelArgs
    vocab: BinArgs
    corpus: BinArgs
    query: BinArgs

    def __init__(self, model, rt, vocab: BinArgs, corpus: BinArgs,
                 query: BinArgs):
        """ 
        :param query: should be a json object which represents the
        control condition 
        """
        super().__init__(model, rt, vocab, corpus)
        self.query = query


# TODO: now the model args are passed through command line while the
#   the corpus/vocab is setted in a separate conf file. Consider merge
#   them.
def parse_data_file(data_args_file):
    args = load_json_file(data_args_file)
    vocab_args = AsJson.from_dict(BinArgs, args['vocab'])
    train_corpus_args = AsJson.from_dict(BinArgs, args['corpus'])
    return vocab_args, train_corpus_args


def json_update(src, delta):
    src = copy.copy(src)
    src.update(delta)
    return src


def parse_eval_file(data_args_file):
    args = load_json_file(data_args_file)
    vocab_args = AsJson.from_dict(BinArgs, args['vocab'])
    train_corpus_args = AsJson.from_dict(BinArgs, args['corpus'])
    query_corpus_args = AsJson.from_dict(
        BinArgs, json_update(args['corpus'], args['query']))
    return vocab_args, train_corpus_args, query_corpus_args
