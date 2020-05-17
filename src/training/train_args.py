from src.database.program_dao import BinArgs
from src.utils.auto_json import auto_json, AutoJson
from src.utils.json_utils import obj_update


@auto_json
class DatasetArgs:
    vocab: BinArgs
    base_corpus: BinArgs
    find_corpus: BinArgs

    def __init__(self, vocab=None, base_corpus=None, find_corpus=None):
        self.vocab = vocab
        self.base_corpus = base_corpus
        self.find_corpus = find_corpus


@auto_json
class RuntimeArgs:

    def __init__(self, epochs=None, n_batch=None, init_lr=None):
        self.epochs = epochs
        self.n_batch = n_batch
        self.init_lr = init_lr


@auto_json
class TrainArgs:
    """
    Used to identify model and training results, such as loss and metrics
    """
    rt: RuntimeArgs

    def __init__(self, dataset_args=None, runtime_args=None, model_args=None):
        self.ds = dataset_args
        self.rt: RuntimeArgs = runtime_args
        self.m = model_args


def parse_dataset_args_from_file(file):
    args: DatasetArgs = AutoJson.load(file)
    if not args.find_corpus:
        args.find_corpus = args.base_corpus
    obj_update(args.base_corpus, args.find_corpus)
    return args


def prepare_args(data_args, epochs, n_batch, init_lr, ModelArg, **model_args):
    # dataset args are loaded from file since they are too complex
    # to be passed through command line
    ds_args = parse_dataset_args_from_file(data_args)
    rt_args = RuntimeArgs(epochs, n_batch, init_lr)
    m_args = ModelArg(**model_args)
    return TrainArgs(ds_args, rt_args, m_args)
