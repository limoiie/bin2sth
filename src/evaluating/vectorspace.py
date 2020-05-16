import torch

from src.preprocesses.corpus import Corpus


class VocabWrapper:
    def __init__(self, vocab, ws, cuda):
        self.vocab = vocab
        self.word_subsam = torch.tensor(ws, dtype=torch.float).unsqueeze(1)

        if cuda >= 0:
            self.word_subsam = self.word_subsam.cuda(device=cuda)


class CorpusWrapper:
    def __init__(self, token_vec_space, corpus: Corpus, cuda):
        self.vw = token_vec_space

        self.idx2doc = corpus.idx2doc
        self.doc2idx = corpus.doc2idx
        self.idx2ins = corpus.idx2ins

        self.idx2hist = torch.stack([self.__hist(ins) for ins in self.idx2ins])
        self.embedding = None

        if cuda >= 0:
            self.idx2hist = self.idx2hist.cuda(device=cuda)

    def __hist(self, insts):
        hist = torch.zeros_like(self.vw.word_subsam)
        for inst in insts:
            for tkn in inst:
                hist[tkn][0] += 1
        # return hist / hist.sum()
        return hist

    def update_embedding_by_accum(self, idx2vec, t=1):
        # normalize and 
        idx2vec = idx2vec - idx2vec.mean()
        emb = idx2vec * self.vw.word_subsam.pow(t)

        # sum up code embeddings as the function embedding
        ins_embedding = [(emb * hist).mean(0) for hist in self.idx2hist]
        self.embedding = torch.stack(ins_embedding)
        return self.embedding

    def update_embedding(self, embedding):
        self.embedding = embedding
        return self.embedding

    def code_idx(self, doc):
        return self.idx2ins[self.doc2idx[doc]]

    def code(self, doc):
        idxs = self.code_idx(doc)
        return [[self.vw.vocab.idx2tkn[i] for i in ins] for ins in idxs]
