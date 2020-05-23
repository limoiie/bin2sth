from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn import Embedding

from src.models.model import UnSupervisedModule
from src.models.modules.word2vec import Word2Vec
from src.models.builder import ModelBuilder
from src.preprocesses.corpus import Corpus
from src.preprocesses.vocab import AsmVocab
from src.utils.auto_json import auto_json
from src.utils.logger import get_logger


@auto_json
@dataclass
class CBowPVDMArgs:
    n_emb: int = 0
    n_negs: int = 0
    use_wr: bool = False


@ModelBuilder.register_cls
class CBowPVDM(UnSupervisedModule):
    w2v: Word2Vec
    vocab: AsmVocab
    corpus: Corpus

    logger = get_logger('CBowPVDM')

    def __init__(self, w2v=None, vocab=None, corpus=None, args=None):
        super(CBowPVDM, self).__init__()

        self.args = args
        self.w2v = w2v
        self.vocab_size = 0
        self.corpus_size = 0
        self.neg_samp_dist = None
        self.f2v: Optional[Embedding] = None

        if vocab is not None:
            self.vocab_size = vocab.size
            self.corpus_size = corpus.n_docs
            self.init_param()
            if args.use_wr:
                t = np.power(self.vocab.word_freq_ratio(), 0.75)
                self.neg_samp_dist = torch.from_numpy(t / t.sum())

    def forward(self, input_batch):
        fun, word, context = input_batch
        batch_size = word.size()[0]
        neg_words = self.__neg_sample(batch_size)

        # batch_size * 1 * embed_size
        cur_vec = self.w2v.forward_hdn(word).unsqueeze(1)
        # batch_size * n_negs * embed_size
        neg_vec = self.w2v.forward_hdn(neg_words).neg()
        # batch_size * (1 + n_negs) * embed_size
        sam_vec = torch.cat([cur_vec, neg_vec], dim=1)

        # batch_size * 1 * embed_size
        fun_vec = self.f2v.forward(fun).unsqueeze(1)
        # batch_size * ctx_size * embed_size
        ctx_vec = self.w2v.forward_vec(context)
        # batch_size * embed_size * 1
        prd_vec = torch.cat([fun_vec, ctx_vec], dim=1).mean(dim=1).unsqueeze(2)

        sim = torch.bmm(sam_vec, prd_vec).squeeze(2).sigmoid()

        loss = -sim.masked_fill(sim == 0, 1).log().sum(1).mean()
        # loss = loss.log().sum(1).mean()
        # return loss, sam_vec.abs().mean()
        return loss

    def init_param(self):
        self.f2v = torch.nn.Embedding(self.corpus_size, self.args.n_emb,
                                      _weight=self.__init_weights())

    def __init_weights(self):
        t = torch.zeros(self.corpus_size, self.args.n_emb)
        t.uniform_(-0.5 / self.args.n_emb, 0.5 / self.args.n_emb)
        return t

    def interested_out(self):
        return self.w2v, self.f2v

    def __neg_sample(self, batch_size):
        if self.neg_samp_dist is not None:
            return self.__neg_sample_in_dist(batch_size)
        else:
            return self.__neg_sample_in_uniform(batch_size)

    def __neg_sample_in_dist(self, batch_size):
        return torch.multinomial(
            self.neg_samp_dist, batch_size * self.args.n_negs,
            replacement=True).view(batch_size, -1)

    def __neg_sample_in_uniform(self, batch_size):
        # NOTE: I modify [0, vocab_size-1] to [1, vocab_size]
        return torch.zeros(batch_size, self.args.n_negs) \
            .uniform_(1, self.vocab_size).long()


def doc_eval_transform(output):
    """
    Transform the output of :class CBowPVDM to the form that
    could be accepted the :class ignite.metrics.TopKCategoricalAccuracy
    """
    doc_ids, (_, base_doc_embedding), (_, doc_embedding) = output
    true_embedding_w = normalize(base_doc_embedding.weight)
    pred_embedding_w = normalize(doc_embedding(doc_ids))

    y_pred = torch.matmul(pred_embedding_w, true_embedding_w.T)
    y = doc_ids

    return y_pred, y.T


def doc_eval_flatten_transform(output):
    """
    Transform the output of :class CBowPVDM to the form that
    could be accepted by the subclasses of :class ignite.mertics.Mertic
    that are based on pair-wise losses. The finall output will be flatten
    into 1-dimension
    """
    doc_ids, (_, base_doc_embedding), (_, doc_embedding) = output
    true_embedding_w = normalize(base_doc_embedding.weight)
    pred_embedding_w = normalize(doc_embedding(doc_ids))

    y_pred = torch.matmul(true_embedding_w, pred_embedding_w.T)
    y = torch.zeros_like(y_pred, dtype=torch.int32)
    y[doc_ids] = torch.eye(len(doc_ids), dtype=torch.int32, device=y.device)

    return y_pred.reshape(-1), y.reshape(-1)


def normalize(m):
    return m / m.norm(dim=1, keepdim=True)
