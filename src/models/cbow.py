import torch
import numpy as np

from logging import getLogger


g_i = 0

def describe_loss(loss):
    global  g_i
    if (g_i % 100) == 0:
        np.set_printoptions(precision=3)
        x = np.copy(loss.cpu().detach())
        p, n = x[:,0], x[:,1:].mean(axis=1)
        print(f'loss mean is \t{p.mean()},\t{n.mean()}')
        print(f'loss median is \t{np.median(p)},\t{np.median(n)}')
    g_i += 1


class WordEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, padding_idx=0, no_hdn=False):
        super(WordEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.padding_idx = padding_idx

        self.idx2vec = self.__create_embedding()
        self.idx2vec.weight = self.__create_param()

        if no_hdn:
            self.idx2hdn = self.idx2vec
        else:
            self.idx2hdn = self.__create_embedding()
            self.idx2hdn.weight = self.__create_param()

    def forward(self, idx):
        return self.forward_vec(idx)

    def forward_vec(self, idx):
        return self.idx2vec(self.__cuda_wrap(idx))

    def forward_hdn(self, idx):
        return self.idx2hdn(self.__cuda_wrap(idx))

    def freeze(self):
        for p in self.parameters():
            p.required_grad = False

    def active(self):
        for p in self.parameters():
            p.required_grad = True

    def __create_embedding(self):
        return torch.nn.Embedding(self.vocab_size, self.embed_size,
                                  padding_idx=self.padding_idx)

    def __create_param(self):
        t = torch.FloatTensor(self.vocab_size, self.embed_size)
        t.uniform_(-0.5 / self.embed_size, 0.5 / self.embed_size)
        t[self.padding_idx] = 0
        return torch.nn.Parameter(t, requires_grad=True)

    def __cuda_wrap(self, data):
        v = torch.LongTensor(data)
        return v.cuda(self.idx2vec.weight.device) if self.idx2vec.weight.is_cuda else v


class FuncEmbedding(torch.nn.Module):
    def __init__(self, corpus_size, embed_size):
        super(FuncEmbedding, self).__init__()

        self.corpus_size = corpus_size
        self.embed_size = embed_size

        self.idx2vec = self.__create_embedding()
        self.idx2vec.weight = self.__create_param()

    def forward(self, idx):
        return self.idx2vec(self.__cuda_wrap(idx))

    def __create_embedding(self):
        return torch.nn.Embedding(self.corpus_size, self.embed_size)

    def __create_param(self):
        t = torch.FloatTensor(self.corpus_size, self.embed_size)
        t.uniform_(-0.5 / self.embed_size, 0.5 / self.embed_size)
        return torch.nn.Parameter(t, requires_grad=True)

    def __cuda_wrap(self, data):
        v = torch.LongTensor(data)
        return v.cuda(self.idx2vec.weight.device) if self.idx2vec.weight.is_cuda else v


class CBowNegSample(torch.nn.Module):
    logger = getLogger('CBowNegSample')

    def __init__(self, embedding, doc_embedding, vocab_size, n_negs, word_freq=None):
        super(CBowNegSample, self).__init__()
        
        self.embedding = embedding
        self.doc_embedding = doc_embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.neg_samp_dist = None

        if word_freq is not None:
            t = np.power(word_freq, 0.75)
            self.neg_samp_dist = torch.FloatTensor(t / t.sum())

    def forward(self, fun, word, context):
        batch_size = word.size()[0]
        neg_words = self.__neg_sample(batch_size)

        # batch_size * 1 * embed_size
        cur_vectors = self.embedding.forward_hdn(word).unsqueeze(1)
        # batch_size * n_negs * embed_size
        neg_vectors = self.embedding.forward_hdn(neg_words).neg()
        # batch_size * (1 + n_negs) * embed_size
        sam_vectors = torch.cat([cur_vectors, neg_vectors], dim=1)

        # batch_size * 1 * embed_size
        fun_vectors = self.doc_embedding.forward(fun).unsqueeze(1)
        # batch_size * ctx_size * embed_size
        ctx_vectors = self.embedding.forward_vec(context)
        # batch_size * embed_size * 1
        prd_vectors = torch.cat([fun_vectors, ctx_vectors], dim=1).mean(dim=1).unsqueeze(2)

        sim = torch.bmm(sam_vectors, prd_vectors).squeeze(2).sigmoid()
        # if (sim == 0).any():
        #     self.logger.warning('There is a 0 in sim!')
        #     import IPython
        #     IPython.embed()
        # describe_loss(sim)

        loss = -sim.masked_fill(sim == 0, 1).log().sum(1).mean()
        # loss = loss.log().sum(1).mean()
        return loss, sam_vectors.abs().mean()

        # loss = -torch.bmm(sam_vectors, prd_vectors).squeeze(2).sigmoid().log().sum(1).mean()
        # return loss, cur_vectors.abs().mean()

    def __neg_sample(self, batch_size):
        if self.neg_samp_dist is not None:
            return self.__neg_sample_in_dist(batch_size)
        else:
            return self.__neg_sample_in_uniform(batch_size)

    def __neg_sample_in_dist(self, batch_size):
        return torch.multinomial(self.neg_samp_dist, batch_size * self.n_negs,
                                 replacement=True).view(batch_size, -1)

    def __neg_sample_in_uniform(self, batch_size):
        # NOTE: I modify [0, vocab_size-1] to [1, vocab_size]
        return torch.FloatTensor(batch_size, self.n_negs)\
            .uniform_(1, self.vocab_size).long()
