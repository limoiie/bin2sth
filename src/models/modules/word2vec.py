import torch


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, padding_idx=0, no_hdn=False):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.padding_idx = padding_idx

        self.idx2vec = self.__create_embedding(
            self.__create_param())

        if no_hdn:
            self.idx2hdn = self.idx2vec
        else:
            self.idx2hdn = self.__create_embedding(
                self.__create_param())

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

    def __create_embedding(self, weight):
        return torch.nn.Embedding(
            self.vocab_size, self.embed_size,
            padding_idx=self.padding_idx, _weight=weight)

    def __create_param(self):
        t = torch.zeros((self.vocab_size, self.embed_size), dtype=torch.float32)
        t.uniform_(-0.5 / self.embed_size, 0.5 / self.embed_size)
        t[self.padding_idx] = .0
        return t

    def __cuda_wrap(self, data):
        return data.to(self.idx2vec.weight.device)
