class HistoryModel:
    def __init__(self)->None:
        self.train_loss:list = []
        self.val_loss:list = []
        self.train_acc:list = []
        self.val_acc:list  = []

    def add_train_loss(self, train_loss)->None:
        self.train_loss.append(train_loss)

    def add_val_loss(self, val_loss)->None:
        self.val_loss.append(val_loss)

    def add_train_acc(self, train_acc)->None:
        self.train_acc.append(train_acc)

    def add_val_acc(self, val_acc)->None:
        self.val_acc.append(val_acc)

    def get_train_loss(self)->list:
        return self.train_loss

    def get_val_loss(self)->list:
        return self.val_loss

    def get_train_acc(self)->list:
        return self.train_acc

    def get_val_acc(self)->list:
        return self.val_acc