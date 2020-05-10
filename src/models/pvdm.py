import torch
import numpy as np

from src.models.model import UnSupervisedModule
from src.utils.logger import get_logger


# noinspection PyArgumentList
class FuncEmbedding(torch.nn.Module):
    def __init__(self, corpus_size, embed_size):
        super(FuncEmbedding, self).__init__()

        self.corpus_size = corpus_size
        self.embed_size = embed_size

        self.idx2vec = self.__create_embedding(
            self.__create_param())

    def forward(self, idx):
        return self.idx2vec(self.__cuda_wrap(idx))

    def __create_embedding(self, weight):
        return torch.nn.Embedding(self.corpus_size, self.embed_size,
                                  _weight=weight)

    def __create_param(self):
        t = torch.FloatTensor(self.corpus_size, self.embed_size)
        t.uniform_(-0.5 / self.embed_size, 0.5 / self.embed_size)
        return t

    def __cuda_wrap(self, data):
        return data.to(self.idx2vec.weight.device)


# noinspection PyArgumentList
class CBowPVDM(UnSupervisedModule):

    logger = get_logger('CBowPVDM')

    def __init__(self, embedding, doc_embedding, vocab_size, n_negs, wr=None):
        super(CBowPVDM, self).__init__()

        self.w2v = embedding
        self.f2v = doc_embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.neg_samp_dist = None

        if wr is not None:
            t = np.power(wr, 0.75)
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

    def interested_out(self):
        return self.w2v, self.f2v

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
        return torch.FloatTensor(batch_size, self.n_negs) \
            .uniform_(1, self.vocab_size).long()


def doc_eval_transform(output):
    """
    Transform the output of :class CBowPVDM to the form that
    could be accepted the :class ignite.metrics.TopKCategoricalAccuracy
    """
    doc_ids, (_, base_doc_embedding), (_, doc_embedding) = output
    true_embedding_w = normalize(base_doc_embedding.idx2vec.weight)
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
    true_embedding_w = normalize(base_doc_embedding.idx2vec.weight)
    pred_embedding_w = normalize(doc_embedding(doc_ids))

    y_pred = torch.matmul(true_embedding_w, pred_embedding_w.T)
    y = torch.zeros_like(y_pred, dtype=torch.int32)
    y[doc_ids] = torch.eye(len(doc_ids), dtype=torch.int32, device=y.device)

    return y_pred.reshape(-1), y.reshape(-1)


def normalize(m):
    return m / m.norm(dim=1, keepdim=True)
