from src.database.program_dao import BinArgs
from src.ida.as_json import AsJson, load_json_file
from src.utils.json_utils import json_update


class DatasetArgs(AsJson):
    vocab: BinArgs
    base_corpus: BinArgs
    find_corpus: BinArgs

    def __init__(self, vocab, base_corpus, find_corpus):
        self.vocab = vocab
        self.base_corpus = base_corpus
        self.find_corpus = find_corpus


class RuntimeArgs(AsJson):

    def __init__(self, epochs, n_batch, init_lr):
        self.epochs = epochs
        self.n_batch = n_batch
        self.init_lr = init_lr


class TrainArgs(AsJson):
    """
    Used to identify model and training results, such as loss and metrics
    """
    ds: DatasetArgs
    rt: RuntimeArgs

    def __init__(self, dataset_args, runtime_args, model_args):
        self.ds: DatasetArgs = dataset_args
        self.rt: RuntimeArgs = runtime_args
        self.m = model_args


def parse_dataset_args_from_file(file) -> DatasetArgs:
    args = load_json_file(file)
    args['find_corpus'] = json_update(args['base_corpus'], args['find_corpus'])
    vocab = AsJson.from_dict(BinArgs, args['vocab'])
    base_corpus = AsJson.from_dict(BinArgs, args['base_corpus'])
    find_corpus = AsJson.from_dict(BinArgs, args['find_corpus'])
    return DatasetArgs(vocab, base_corpus, find_corpus)


def prepare_args(data_args, epochs, n_batch, init_lr, ModelArg, **model_args):
    # dataset args are loaded from file since they are too complex
    # to be passed through command line
    ds_args = parse_dataset_args_from_file(data_args)
    rt_args = RuntimeArgs(epochs, n_batch, init_lr)
    m_args = ModelArg(**model_args)
    return TrainArgs(ds_args, rt_args, m_args)
