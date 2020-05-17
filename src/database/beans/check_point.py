from src.utils.auto_json import auto_json


@auto_json
class CheckPoint:
    def __init__(self, train_process_id, checkpoints):
        self.train_process_id = train_process_id
        self.checkpoints = checkpoints
