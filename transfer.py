import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time
import os

from src.models import create_model
from dataset import create_dataloader

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch TResNet Transfer Training')
# data loader settings
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--zoom_factor', type=int, default=0.875)
# pretrained model setting
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--remove_aa_jit', action='store_true', default=False)
# transfer learning settings
parser.add_argument('--transfer_num_classes', type=int, default=196)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--transfer_model_dir', type=str, default='./checkpoints')


def train(model, dataloader, args, device, stage='only_fc'):
    start = time.time()
    # training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if stage == 'only_fc':
        num_epochs = args.epochs // 4
    elif stage == 'full_net':
        num_epochs = args.epochs - args.epochs // 4
    else:
        raise ValueError('wrong training stage!')

    # training statistics
    for epoch in range(num_epochs):
        total_loss = 0
        corrects = 0
        print('Epoch {}/{}'.format(epoch, num_epochs-1))

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # clear the gradient for each iteration
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # update the parameters
            loss.backward()
            optimizer.step()

            # compute the training performance for this iteration
            total_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(predicted == labels.data)

        scheduler.step()

        # print the performance for this epoch
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = torch.true_divide(corrects, len(dataloader.dataset))
        print('Average loss: {}, Accuracy: {}'.format(avg_loss, accuracy))

        # may save the best model
        # do this...
    time_elapsed = time.time() - start
    print('Training finish: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # save model
    state = {
        'model': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': num_epochs-1,
        'stage': stage
    }
    ckpts_name = os.path.join(args.transfer_model_dir, 'transfer_model.pth')
    torch.save(state, ckpts_name)
    return model


def main():
    # parsing args
    args = parser.parse_args()

    # gpu setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setup model
    print('load pretrained model...')
    model = create_model(args)
    state = torch.load(args.model_path, map_location='cpu')['model']
    model.load_state_dict(state, strict=False)

    print('set up the model for fine-grained task...')
    for param in model.parameters():
        param.requires_grad = False    # freeze the convnet
    num_in_fc = model.num_features
    model.head.fc = nn.Linear(num_in_fc, args.transfer_num_classes)
    model = model.to(device)
    print('done\n')

    # setup data loader
    print('creating data loader...')
    train_loader = create_dataloader(args)
    print('done\n')

    # training process
    print('start training...')
    # only train the fc
    print('train only fc layer:')
    model = train(model, train_loader, args, device, stage='only_fc')
    # train the entire network
    print('train entire network:')
    for param in model.parameters():
        param.requires_grad = True
    train(model, train_loader, args, device, stage='full_net')
    print('finish training\n')


if __name__ == '__main__':
    main()
