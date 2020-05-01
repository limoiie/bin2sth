from tqdm import tqdm


class ProgressBar:
    def __init__(self, dataloader, description, update_gap):
        self.bar = tqdm(dataloader)
        self.bar.set_description(description)
        self.update_gap = update_gap

    def step(self, **args):
        self.bar.set_postfix(**args)
