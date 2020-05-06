def layer_trainable(m, trainable=True):
    for p in m.parameters():
        p.requires_grad = trainable
