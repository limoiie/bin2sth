from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import Embedding

from src.models.builder import ModelBuilder
from src.preprocesses.vocab import AsmVocab
from src.utils.auto_json import auto_json


@auto_json
class Word2VecRecipe:
    def __init__(self, n_emb=0, no_hdn=False, padding_idx=0):
        self.n_emb = n_emb
        self.no_hdn = no_hdn
        self.padding_idx = padding_idx


@ModelBuilder.register_cls
class Word2Vec(torch.nn.Module):
    vocab: AsmVocab

    def __init__(self, vocab=None, cfg=None):
        super(Word2Vec, self).__init__()
        self.idx2vec: Optional[Embedding] = None
        self.idx2hdn: Optional[Embedding] = None
        self.cfg: Word2VecRecipe = cfg

        if vocab is None:
            self.vocab_size = 0
        else:
            self.vocab_size = vocab.size
            self.init_param()

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

    def init_param(self):
        self.idx2vec = self.__create_embedding(
            self.__create_param())
        self.idx2hdn = self.idx2vec
        if not self.cfg.no_hdn:
            self.idx2hdn = self.__create_embedding(
                self.__create_param())

    def __create_embedding(self, weight):
        return torch.nn.Embedding(
            self.vocab_size, self.cfg.n_emb,
            padding_idx=self.cfg.padding_idx, _weight=weight)

    def __create_param(self):
        t = torch.zeros((self.vocab_size, self.cfg.n_emb), dtype=torch.float32)
        t.uniform_(-0.5 / self.cfg.n_emb, 0.5 / self.cfg.n_emb)
        t[self.cfg.padding_idx] = .0
        return t

    def __cuda_wrap(self, data):
        return data.to(self.idx2vec.weight.device)


@auto_json
@dataclass
class NegSampleArgs:
    n_negs: int = 0
    use_wr: bool = False


@ModelBuilder.register_cls
class NegSample(torch.nn.Module):
    vocab: AsmVocab

    def __init__(self, vocab=None, args=None):
        super().__init__()
        self.args: Optional[NegSampleArgs] = args
        self.vocab_size = 0
        self.__neg_sample_dist = None

        if vocab:
            self.vocab_size = vocab.size
            if args.use_wr:
                t = torch.tensor(vocab.word_freq_ratio()).pow(0.75)
                self.__neg_sample_dist = t / t.sum()

    def neg_sample(self, batch_size):
        # note: I did avoid the positive word during negatively sampling
        # words because, in my opinion, the only downside of no avoiding
        # is the loss cancelling which may slow down the training process
        # slightly but has very little possibility to take place
        if self.__neg_sample_dist is not None:
            return self.__neg_sample_in_dist(batch_size, self.args.n_negs)
        return self.__neg_sample_in_uniform(batch_size, self.args.n_negs)

    def __neg_sample_in_dist(self, batch_size, n_negs):
        """Negative sampling in a given distribution"""
        return torch.multinomial(
            self.__neg_sample_dist, batch_size * n_negs,
            replacement=True).view(batch_size, -1)

    def __neg_sample_in_uniform(self, batch_size, n_negs):
        """Uniformly negative sampling"""
        return torch.zeros(batch_size, n_negs) \
            .uniform_(1, self.vocab_size).long()


@ModelBuilder.register_cls
class CBow(torch.nn.Module):
    w2v: Word2Vec
    sampler: NegSample

    def __init__(self, w2v=None, sampler=None):
        super().__init__()
        self.w2v = w2v
        self.sampler = sampler

    def forward(self, input_batch):
        # ctx: batch_size x n_ctx  :pre
        # cur: batch_size          :hdn
        cur, ctx = input_batch
        batch_size = cur.size()[0]

        # neg: batch_size x n_neg  :hdn
        neg = self.sampler.neg_sample(batch_size)
        # -> batch_size x n_neg

        cur_vec = self.w2v.forward_hdn(cur).unsqueeze(1)
        # -> batch_size x 1 x n_emb
        neg_vec = self.w2v.forward_hdn(neg).neg()
        # -> batch_size x n_neg x n_emb
        sam_vec = torch.cat([cur_vec, neg_vec], dim=1)
        # -> batch_size x (1 + n_nge) x n_emb
        prd_vec = self.w2v.forward_vec(ctx).mean(dim=1).unsqueeze(2)
        # -> batch_size x n_emb x 1
        sim_mat = torch.bmm(sam_vec, prd_vec).squeeze(2)
        # -> batch_size x (1 + n_neg)
        return -sim_mat.sigmoid().log().sum(dim=1).mean()


class SkipGram(torch.nn.Module):

    def __init__(self, w2v, sampler):
        super().__init__()
        self.w2v = w2v
        self.sampler = sampler

    def forward(self, input_batch):
        # ctx: batch_size x n_ctx :hdn
        # cur: batch_size         :pre
        cur, ctx = input_batch
        batch_size, n_ctx = ctx.size()[0], ctx.size()[1]

        # neg: batch_size x n_ctx x n_neg :hdn
        neg = self.sampler.neg_sample(batch_size, self.sampler.n_negs * n_ctx)

        ctx_vec = self.w2v.forward_hdn(ctx)
        # -> batch_size x n_ctx x n_emb
        neg_vec = self.w2v.forward_hdn(neg).neg()
        # -> batch_size x (n_ctx * n_neg) x n_emb
        sam_vec = torch.cat([ctx_vec, neg_vec], dim=1)
        # -> batch_size x (n_ctx * (n_neg + 1)) x n_emb
        cur_vec = self.w2v.forward_vec(cur).unsqueeze(2)
        # -> batch_size x n_emb x 1
        sim_mat = torch.bmm(sam_vec, cur_vec).squeeze(2)
        # -> batch_size x (n_ctx * (n_neg + 1))
        return -sim_mat.sigmoid().log().sum(dim=1).mean()
