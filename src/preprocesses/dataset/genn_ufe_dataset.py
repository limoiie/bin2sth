import random

from src.preprocesses.dataset.dataset import SupervisedDataset

random.seed(10)


class GENNDatasetBuilder:

    def __init__(self):
        self.data, self.label = [], []

    def build(self, corpus1, corpus2):
        t_samples = make_true_samples(corpus1, corpus2)
        f_samples = make_false_samples(corpus1, corpus2, len(t_samples))

        samples = t_samples + f_samples
        random.shuffle(samples)

        self.data, self.label = [], []
        for (i, j), l in samples:
            self.data.append((
                (corpus1.idx2ins[i], corpus1.cfg_adj[i]),
                (corpus2.idx2ins[j], corpus2.cfg_adj[j])
            ))
            self.label.append(l)

        return SupervisedDataset(self.data, self.label)


def make_true_samples(corpus1, corpus2):
    shared_docs = set(corpus1.idx2doc) & set(corpus2.idx2doc)
    samples = []
    for doc in shared_docs:
        x, y = corpus1.doc2idx[doc], corpus2.doc2idx[doc]
        samples.append(((x, y), 1))
    return samples


def make_false_samples(corpus1, corpus2, n):
    l1, l2 = len(corpus1.idx2doc), len(corpus2.idx2doc)
    samples = []
    for i in range(n):
        x = random.randint(0, l1-1)
        while True:
            y = random.randint(0, l2-1)
            if corpus1.idx2doc[x] != corpus2.idx2doc[y]:
                break
        samples.append(((x, y), 0))
    return samples
