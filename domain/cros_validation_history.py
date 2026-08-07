from .history_model import HistoryModel
class CrossValidationHistory:

    def __init__(self):

        self.history_model:HistoryModel = HistoryModel()

        self.acc:list = []

        self.loss:list = []

        self.f1:list = []

    def add_fold(self, loss, acc, f1):

        self.loss.append(loss)

        self.acc.append(acc)

        self.f1.append(f1)