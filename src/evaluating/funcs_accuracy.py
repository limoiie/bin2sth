import torch
import numpy as np

from src.utils.logger import get_logger
from src.corpus import Corpus
from src.evaluating.vectorspace import CorpusWrapper, VocabWrapper


def score_sim_mat(sim, topks=None, coe=1.):
    """ Count the examples under topk """
    if topks is None:
        topks = [1, 5, 10]
    keep_topk = max(topks)
    sim = sim * coe
    top_pred = torch.zeros(sim.shape[0], keep_topk)
    for i in range(sim.shape[0]):
        top_pred[i] = sim[i].sort().values[-keep_topk:].cpu()

    pred = sim.diagonal().cpu()
    topk_nums = np.zeros((len(topks),))
    for i, k in enumerate(topks):
        topk_nums[i] = (pred >= top_pred[:, keep_topk - k]).sum()
    return topk_nums


def normalize(tv, qv):
    tv = tv / tv.norm(dim=1, keepdim=True)
    qv = qv / qv.norm(dim=1, keepdim=True)
    return tv, qv


class QueryMatrix:
    def __init__(self, eva, tv, qv):
        self.eva = eva
        self.tv = tv
        self.qv = qv
        self.sim = torch.matmul(tv, qv.T).cpu()
        self.topsim = self.sim.sort(dim=1, descending=True).values
        self.toparg = self.sim.argsort(dim=1, descending=True)
        self.pred = self.sim.diagonal()

    def code(self, doc):
        return self.eva.code(doc, True), self.eva.code(doc, False)

    def topk_by_doc(self, doc, k=1):
        i = self.eva.doc2idx[doc]
        topk = self.toparg[i, :k]

        topk_accs = self.sim[i, topk]
        topk_docs = self.eva.idx2doc[topk]
        print(f'Compare with {doc} (true {self.pred[i]:.3f}):')
        for i, (a, d) in enumerate(zip(topk_accs, topk_docs)):
            print(f'\t{i:02}: {a:.3f} -> {d}')
        return topk_accs, topk_docs

    def bad_docs(self, k=1):
        out_idxs = self.pred < self.topsim[:, k-1]
        out_docs = self.eva.idx2doc[out_idxs]
        return out_docs


# noinspection PyCallingNonCallable
class Evaluation:
    logger = get_logger('evalutaion')

    def __init__(self, cuda, vocab, word_subsam, train_corpus: Corpus,
                 query_corpus: Corpus):
        self.vw = VocabWrapper(vocab, word_subsam, cuda)

        self.tc = CorpusWrapper(self.vw, train_corpus, cuda)
        self.qc = CorpusWrapper(self.vw, query_corpus, cuda)

        t_docs = set(self.tc.idx2doc)
        q_docs = set(self.qc.idx2doc)

        self.idx2doc = np.array(list(t_docs.intersection(q_docs)))
        self.doc2idx = {doc: idx for idx, doc in enumerate(self.idx2doc)}
        self.n_docs = len(self.idx2doc)

        self.t_idx = torch.tensor([self.tc.doc2idx[f] for f in self.idx2doc])
        self.q_idx = torch.tensor([self.qc.doc2idx[f] for f in self.idx2doc])

        self.truth = torch.tensor(np.arange(self.n_docs))

        # self.ct_ins = torch.tensor([len(self.tc.idx2ins[i]) for i in self.t_idx])
        # self.cq_ins = torch.tensor([len(self.qc.idx2ins[i]) for i in self.q_idx])

        # self.ct_ins = self.ct_ins.unsqueeze(dim=0)
        # self.cq_ins = self.cq_ins.unsqueeze(dim=1)

        if cuda >= 0:
            self.t_idx = self.t_idx.cuda(device=cuda)
            self.q_idx = self.q_idx.cuda(device=cuda)
            self.truth = self.truth.cuda(device=cuda)
            # self.ct_ins = self.ct_ins.cuda(device=cuda)
            # self.cq_ins = self.cq_ins.cuda(device=cuda)

    def code(self, doc, is_train):
        return self.tc.code(doc) if is_train else self.qc.code(doc)

    def evaluate1(self, train_fun_emb, query_fun_emb):
        tv, qv = self.__make_common_matrix1(train_fun_emb, query_fun_emb)
        sim = torch.matmul(tv, qv.T)
        return self.score_and_print('sim1', sim)

    def evaluate2(self, embedding, t=1):
        tv, qv = self.__make_common_matrix2(embedding.idx2vec.weight, t)
        sim = torch.matmul(tv, qv.T)
        return self.score_and_print('sim2', sim)

    def matrix1(self, train_fun_emb, query_fun_emb):
        tv, qv = self.__make_common_matrix1(train_fun_emb, query_fun_emb)
        return QueryMatrix(self, tv, qv)

    def matrix2(self, embedding, t=1):
        tv, qv = self.__make_common_matrix2(embedding.idx2vec.weight, t)
        return QueryMatrix(self, tv, qv)

    def __make_common_matrix1(self, t_emb, q_emb):
        tv = t_emb.idx2vec.forward(self.t_idx)
        qv = q_emb.idx2vec.forward(self.q_idx)

        return normalize(tv, qv)

    def __make_common_matrix2(self, idx2vec, t):
        tv = self.tc.update_embedding_by_accum(idx2vec, t)[self.t_idx]
        qv = self.qc.update_embedding_by_accum(idx2vec, t)[self.q_idx]

        return normalize(tv, qv)

    # def length_coefficient(self, e=1.):
    #     len_gap = torch.abs(self.ct_ins - self.cq_ins)
    #     len_sum = self.ct_ins + self.cq_ins
    #     return 1.0 - len_gap / len_sum / e

    def score_and_print(self, prefix, sim, topks=None):
        total = sim.shape[0]
        topk_nums = score_sim_mat(sim, topks)
        topk_accs = topk_nums / total
        self.logger.info(f'{prefix} -> {topk_nums}/{total} ({topk_accs})')
        return topk_nums, total

    def score_partially_and_print(self, sim, docs):
        s = []
        for doc in docs:
            for i, doc2 in enumerate(self.idx2doc):
                if doc2.endswith(doc):
                    s.append(i)
                    break
        sim = sim[s][:, s]
        return self.score_and_print('part sim', sim)


def make_evaluation(train_data, query_data, cuda, ws):
    vocab = train_data.vocab
    train_corpus = train_data.corpus
    query_corpus = query_data.corpus
    return Evaluation(cuda, vocab, ws, train_corpus, query_corpus)
