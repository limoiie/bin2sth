from src.utils.auto_json import auto_json


@auto_json
class CheckPoint:
    def __init__(self, training_id=None, checkpoints=None):
        self.training_id = training_id
        self.checkpoints = checkpoints
