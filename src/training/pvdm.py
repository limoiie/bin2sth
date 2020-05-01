from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database import ArgsBag
from src.database import load_cbow_data_end, get_database_client, load_json_file
from src.dataset import CBowDataset
from src.models.pvdm import CBowPVDM, WordEmbedding, FuncEmbedding
from src.utils.progress_bar import ProgressBar
from src.vocab import compute_word_freq_ratio, compute_sub_sample_ratio


def parse_data_file(data_args_file):
    args = load_json_file(data_args_file)
    vocab_args = ArgsBag().deserialize(args['vocab'])
    train_args = ArgsBag().deserialize(args['train'])
    return vocab_args, train_args


def train_one_epoch(epoch, model, dataset, optimizer, n_batch):
    data_loader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
    progress_bar = ProgressBar(data_loader, f'[Epoch {epoch}]', 100)

    # TODO: use average loss instead
    step_loss = 0
    for i, (fun, word, ctx) in enumerate(progress_bar.bar):
        # print(f'data entry: ({fun}, {word}, {ctx})')
        loss, mean_vec = model(fun, word, ctx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss = loss.item()
        progress_bar.step(loss=step_loss)
    return step_loss


def train(epochs, n_batch, n_emb, n_negs, init_lr, no_hdn,
          cuda, ss, window, data_args):
    # prepare database
    client = get_database_client()
    db = client.test_database

    # parser data configuration and load data
    vocab_args, train_args = parse_data_file(data_args)
    data_end = load_cbow_data_end(db, vocab_args, train_args, window)

    wf = compute_word_freq_ratio(data_end.vocab)
    ws = compute_sub_sample_ratio(wf, ss)

    # init embedding matrix
    vocab_size = data_end.vocab.total_unique_tkn
    corpus_size = data_end.corpus.n_docs
    embedding = WordEmbedding(vocab_size, n_emb, no_hdn=no_hdn)
    doc_embedding = FuncEmbedding(corpus_size, n_emb)

    # create model
    model = CBowPVDM(embedding, doc_embedding, vocab_size, n_negs, wf)
    optimizer = Adam(model.parameters(), lr=init_lr)

    loss_epochs = []
    for epoch in range(epochs):
        dataset = CBowDataset(data_end, ws)
        loss = train_one_epoch(epoch, model, dataset, optimizer, n_batch)
        loss_epochs.append(loss)

    client.close()


# if __name__ == "__main__":
#     fire.Fire(train)
