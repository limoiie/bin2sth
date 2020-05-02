import copy

from src.database import BinArgs, load_json_file

from src.ida.code_elements import Serializable


class ModelArgs(Serializable):
    def __init__(self, epochs, n_batch, n_emb, n_negs, init_lr,
                 no_hdn, ss, window):
        self.epochs = epochs
        self.n_batch = n_batch
        self.n_emb = n_emb
        self.n_negs = n_negs
        self.init_lr = init_lr
        self.no_hdn = no_hdn
        self.ss, self.window = ss, window


class TrainArgs(Serializable):
    def __init__(self, model, rt, vocab: BinArgs, corpus: BinArgs):
        self.model = model
        self.rt = rt
        self.vocab = vocab
        self.corpus = corpus

    def serialize(self):
        dic = super().serialize()
        dic['rt'] = self.to_json(self.rt)
        dic['vocab'] = self.to_json(self.vocab)
        dic['corpus'] = self.to_json(self.corpus)
        return dic

    def deserialize(self, data):
        super().deserialize(data)
        self.rt = self.from_json(ModelArgs, data['rt'])
        self.vocab = self.from_json(BinArgs, data['vocab'])
        self.corpus = self.from_json(BinArgs, data['corpus'])
        return self


class QueryArgs(TrainArgs):
    def __init__(self, model, rt, vocab: BinArgs, corpus: BinArgs, query):
        """ 
        :param query: should be a json object which represents the
        control condition 
        """
        super().__init__(model, rt, vocab, corpus)
        self.query = query

    def serialize(self):
        dic = super().serialize()
        dic['query'] = self.to_json(self.query)
        return dic

    def deserialize(self, data):
        super().deserialize(data)
        self.query = self.from_json(BinArgs, data['query'])
        return self


def parse_data_file(data_args_file):
    args = load_json_file(data_args_file)
    vocab_args = BinArgs().deserialize(args['vocab'])
    train_corpus_args = BinArgs().deserialize(args['corpus'])
    return vocab_args, train_corpus_args


def json_extend(src, delta):
    src = copy.deepcopy(src)
    for key in delta:
        src[key] = delta[key]
    return src


def parse_eval_file(data_args_file):
    args = load_json_file(data_args_file)
    vocab_args = BinArgs().deserialize(args['vocab'])
    train_corpus_args = BinArgs().deserialize(args['corpus'])
    query_corpus_args = BinArgs().deserialize(
        json_extend(args['corpus'], args['query']))
    return vocab_args, train_corpus_args, query_corpus_args
