from PIL import Image
import torch
from torch.utils.data import Dataset

class MVCNNDataset(Dataset):

    def __init__(self, views_list, labels, classes, transform=None):

        self.views_list = views_list
        self.labels = labels
        self.transform = transform
        self.classes = classes

        self.class_to_idx = {
            cls: i for i, cls in enumerate(classes)
        }

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        image_paths = self.views_list[idx]
        label_str = self.labels[idx]

        views = []

        for path in image_paths:

            img = Image.open(path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            views.append(img)

        views = torch.stack(views)

        label = self.class_to_idx[label_str]
        label = torch.tensor(label, dtype=torch.long)

        return views, label