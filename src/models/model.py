from torch.nn import Module


def layer_trainable(m, trainable=True):
    for p in m.parameters():
        p.requires_grad = trainable


class UnSupervisedModule(Module):
    def interested_out(self):
        raise NotImplementedError(
            'Un-supervised model should implement this to to expose the '
            'interested part of the whole model.')
