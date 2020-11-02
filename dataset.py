import csv
import numpy as np
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class TrainingSet(Dataset):
    def __init__(self, args, transform):
        self.train_dir = args.dataset_path
        self.transform = transform
        self.imgs, self.indices = load_csv(args.label_path)

    def __getitem__(self, index):
        image_path = os.path.join(self.train_dir, self.imgs[index]) + '.jpg'
        image = self.transform(Image.open(image_path).convert('RGB'))
        index = self.indices[index]
        return image, index

    def __len__(self):
        return len(self.imgs)


def load_csv(path):
    imgs = []
    labels = []
    indices = []
    with open(path, newline='') as csv_file:
        # read csv file
        rows = csv.DictReader(csv_file)
        for row in rows:
            imgs.append(row['id'])
            labels.append(row['label'])

        # transform labels to index
        idx2label = sorted(set(labels))
        label2idx = dict()
        for i in range(len(idx2label)):
            label2idx[idx2label[i]] = i
        for label in labels:
            indices.append(label2idx[label])

        # write idx2label to np file
        np.save('class_name.npy', np.array(idx2label))
    return imgs, indices


def create_dataloader(args):
    # define the transform for preprocessing and data augmentation
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(args.input_size),
         transforms.RandomHorizontalFlip(0.5),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # create training dataset
    training_set = TrainingSet(args, transform)
    dataloader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader
