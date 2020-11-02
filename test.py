import torch
import argparse
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

from src.models import create_model


parser = argparse.ArgumentParser(description='PyTorch TResNet Fine-Grained Test')
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--prediction_path', type=str, default='./result.csv')
parser.add_argument('--class_name_path', type=str, default='./class_name.npy')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=196)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--test_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--remove_aa_jit', action='store_true', default=False)


class TestDataset(Dataset):
    def __init__(self, args):
        self.img_paths = [os.path.join(args.test_dir, f)
                          for f in os.listdir(args.test_dir) if os.path.isfile(os.path.join(args.test_dir, f))]
        self.transform = transforms.Compose(
                [transforms.Resize(int(args.input_size / args.test_zoom_factor)),
                 transforms.CenterCrop(args.input_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = self.transform(Image.open(self.img_paths[index]).convert('RGB'))
        return image

    def __len__(self):
        return len(self.img_paths)


def create_dataloader(args):
    testset = TestDataset(args)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return testloader


def write_csv(file_name, results, dataset):
    # result size must be the same as data size
    assert len(results) == len(dataset), 'the predicted result size should be consistent with data size'

    # start write csv file (result.csv)
    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        # write fields
        writer.writerow(['id', 'label'])

        for i in range(len(results)):
            img_name = os.path.split(dataset.img_paths[i])[-1].split('.')[0]
            result = results[i]
            writer.writerow([img_name, result])


def infer(model, test_loader, args):
    # the predicted class_name of whole datasets
    results = []

    # read the class name file (class_name.npy)
    idx2class = np.load(args.class_name_path)

    # testing: not need gradient
    with torch.no_grad():
        # go through the dataset
        for batch_idx, inputs in enumerate(test_loader):
            # infer the predicted labels
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicts = torch.max(outputs, 1)

            # from idx to class name
            predicted_names = [idx2class[predict] for predict in predicts]
            results += predicted_names
        # write the csv file
        print('write the prediction to {}'.format(args.prediction_path))
        write_csv(args.prediction_path, results, test_loader.dataset)


def main():
    # parsing args
    args = parser.parse_args()

    # setup model
    print('creating model...')
    model = create_model(args).cuda()
    state = torch.load(args.model_path, map_location='cpu')['model']
    model.load_state_dict(state, strict=False)
    model.eval()
    print('done\n')

    # setup data loader
    print('creating data loader...')
    test_loader = create_dataloader(args)
    print('done\n')

    # actual validation process
    print('doing testing...')
    infer(model, test_loader, args)
    print('done\n')


if __name__ == '__main__':
    main()
