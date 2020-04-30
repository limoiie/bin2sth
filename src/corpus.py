from src.utils.logger import get_logger



class Corpus:
    """
    Contains function names and ids
    """
    logger = get_logger('Corpus')

    def __init__(self):
        # doc is binary#function
        self.doc2idx = {}
        self.idx2doc = []
        self.n_docs = 0

    def build(self, docs):
        """ 
        Build the corpus.
        @param docs: a set of function labels
        """
        self.__prepare()
        self.__scan(docs)
    
    def __prepare(self):
        self.doc2idx = {}
        self.idx2doc = []
        self.n_docs = 0
    
    def __scan(self, docs):
        self.idx2doc = list(docs)
        for i, doc in enumerate(self.idx2doc):
            self.doc2idx[doc] = i
        self.n_docs = len(self.doc2idx)

    def onehot_encode(self, fun_name):
        doc = fun_name
        if doc not in self.doc2idx:
            raise RuntimeError(f'No doc with name {doc} in corpus! Consider rebuilding it.')
        return self.doc2idx[doc]
