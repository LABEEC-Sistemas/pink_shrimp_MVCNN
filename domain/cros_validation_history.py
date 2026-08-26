from .history_model import HistoryModel
from .mvcnn_model import MVCNNModel
import copy
class CrossValidationHistory:

    def __init__(self):

        self.history_model: dict[int, HistoryModel] = {}

        self.fold_metrics:list[dict] = []

        self.models:list[MVCNNModel] = []
        self.weigths_models:list = []

    def get_metrics(self)->list[dict]:
        return self.fold_metrics

    def add_model(self, model:MVCNNModel)->None:
        self.models.append(model)
        self.weigths_models.append(copy.deepcopy(model.state_dict()))

    def get_models(self)->list[MVCNNModel]:
        return self.models

    def get_model(self, index)->MVCNNModel:
        if (index-1) < len(self.models):
            return self.models[index-1]

        return None

    def add_fold(self, loss, acc, f1)->None:
        metrics = {
            "loss": loss,
            "acc": acc,
            "f1": f1
        }

        self.fold_metrics.append(metrics)

    def add_train_acc_fold(self, acc, fold)->None:
        if fold not in self.history_model:
            self.history_model[fold] = HistoryModel()

        self.history_model[fold].add_train_acc(acc)

    def get_train_acc_fold(self, fold)->list:
        return self.history_model[fold].get_val_acc()

    def add_train_loss_fold(self, loss, fold)->None:
        if fold not in self.history_model:
            self.history_model[fold] = HistoryModel()

        self.history_model[fold].add_train_loss(loss)

    def get_val_acc_fold(self, fold)->list:
        return self.history_model[fold].get_val_acc()

    def add_val_acc_fold(self, acc, fold)->None:
        if fold not in self.history_model:
            self.history_model[fold] = HistoryModel()

        self.history_model[fold].add_val_acc(acc)

    def get_train_loss_fold(self, fold)->list:
        return self.history_model[fold].get_train_loss()

    def add_val_loss_fold(self, loss, fold)->None:
        if fold not in self.history_model:
            self.history_model[fold] = HistoryModel()

        self.history_model[fold].add_val_loss(loss)

    def get_val_loss_fold(self, fold)->list:
        return self.history_model[fold].get_val_loss()
    
    def add_all_pred_fold(self, all_pred, fold)->None:
        if fold not in self.history_model:
            self.history_model[fold] = HistoryModel()

        self.history_model[fold].add_all_preds(all_pred)

    def get_all_pred_fold(self, fold)->list:
        return self.history_model[fold].get_all_preds()

    def add_all_prob_fold(self, all_prob, fold)->None:
        if fold not in self.history_model:
            self.history_model[fold] = HistoryModel()

        self.history_model[fold].add_all_probs(all_prob)

    def get_all_prob_fold(self, fold)->list:
        return self.history_model[fold].get_all_probs()
    
    def add_all_label_fold(self, all_label, fold)->None:
        if fold not in self.history_model:
            self.history_model[fold] = HistoryModel()

        self.history_model[fold].add_all_labels(all_label)

    def get_all_label_fold(self, fold)->list:
        return self.history_model[fold].get_all_labels()
