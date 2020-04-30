import codecs, pickle, os, logging


class AsmVocab:
    logger = logging.getLogger('AsmVocab')
    
    def __init__(self, unk='<unk>', max_vocab=10000):
        self.unk = unk
        self.size = 0
        self.max_vocab = max_vocab
        self.words = set([])
        self.idx2word = []
        self.word2idx = {}
    
    def build(self, doc):
        """ 
        Do the convertion
        @param doc: iterator of sentences
        """     
        self.logger.debug('building vocab...')
        
        self.wc = {self.unk: 1}
        
        for step, line in enumerate(doc):
            if not step % 1000:
                self.logger.debug(f'working on {step // 1000}kth line')
            for word in line.strip().split():
                self.wc[word] = self.wc.get(word, 0) + 1
        
        sorted_wc = sorted(self.wc, key=self.wc.get, reverse=True)
        self.idx2word = [self.unk] + sorted_wc[:self.max_vocab - 1]
        self.size, self.words = len(self.idx2word), set(self.idx2word)
        self.word2idx = {self.idx2word[i]: i for i in range(self.size)}

        self.logger.debug('building vocab done')


class CBowPreprocessor:
    """ 
    Convert documents into the sequence of (center_word, words_window)
    for the support of training CBow model. The words therein have 
    been substituted with the one-hot encoding by using the given vocab   
    """ 
    logger = logging.getLogger('CBowPreprocessor')    

    def __init__(self, window, vocab: AsmVocab):
        self.window = window
        self.vocab = vocab
        self.data = []
    
    def __unk_list(self, l):
        return [self.vocab.unk] * l

    def __window(self, sentence, i):
        cw = sentence[i]
        l, r = max(i - self.window, 0), i + 1 + self.window
        lw, rw = sentence[l:i], sentence[i+1:r]
        return cw, self.__unk_list(self.window-len(lw)) + lw + \
            rw + self.__unk_list(self.window-len(rw))

    def run(self, doc):
        """ 
        Do the convertion
        @param doc: iterator of sentences
        """ 
        self.logger.debug('building training data...')
        
        data = []
        for step, line in enumerate(doc):
            if not step % 1000:
                self.logger.debug(f'working on {step//1000}kth line')
            line = line.strip()
            if not line:
                continue
            sent = [w if w else self.vocab.unk for w in line.split()]
            for i in range(len(sent)):
                iword, owords = self.__window(sent, i)
                iword = self.vocab.word2idx[iword]
                owords = [self.vocab.word2idx[oword] for oword in owords]
                data.append((iword, owords))
        self.data = data

        self.logger.debug('building training data done')
