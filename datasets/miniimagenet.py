import torchvision as tv
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset
import os
import csv
from tqdm import tqdm

class MiniImagenet84(Dataset):
    # including hard labels & soft labels
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data, self.train_labels =  data, labels
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
            
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        sample = {'image':img, 'target':target, 'index':index}
        return sample


    def __len__(self):
        return len(self.train_data)


def make_dataset(root):
    np.random.seed(42)
    csv_files = ["train.csv", "val.csv", "test.csv"]
    img_paths = []
    labels = []
    for split in csv_files:
        in_csv_path = os.path.join(root, split)
        in_images_path = os.path.join(root, "images")

        with open(in_csv_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i,row in enumerate(csvreader):
                img_paths.append(os.path.join(in_images_path,row[0]))
                labels.append(row[1])

    mapping = {y: x for x, y in enumerate(np.unique(labels))}
    label_mapped = [mapping[i] for i in labels]

    # labels

    # split in train and validation:
    train_num = 50000
    val_num = 10000
    idxes = np.random.permutation(len(img_paths))
    img_paths = np.asarray(img_paths)[idxes]
    label_mapped = np.asarray(label_mapped)[idxes]

    train_paths = img_paths[:train_num]
    train_labels = label_mapped[:train_num]
    val_paths = img_paths[train_num:train_num+val_num]
    val_labels = label_mapped[train_num:train_num+val_num]
    test_paths = img_paths[train_num+val_num:]
    test_labels = label_mapped[train_num+val_num:]

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
