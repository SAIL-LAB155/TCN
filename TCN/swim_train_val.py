import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import sys
sys.path.append("../../")
from TCN.swim_classify.utils import TCNData, train_val_split, get_data
from TCN.swim_classify.model import TCN
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size (default: 8)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=6,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
if use_cuda:
    print('use CUDA')
else:
    print('use CPU')

root = './data'
#data_pth = 'data.txt'
batch_size = args.batch_size
n_classes = 2
input_channels = 30
seq_length = 34
epochs = args.epochs
steps = 0

print(args)
#load data
data = get_data(root)
train_data, test_data = train_val_split(data)
#print(len(train_data))
train_set = TCNData(train_data)
vaild_set = TCNData(test_data)
train_loader = DataLoader(train_set, batch_size=8, shuffle = True)
valid_loader = DataLoader(vaild_set, batch_size=8, shuffle = True)

permute = torch.Tensor(np.random.permutation(360).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

criterion = nn.CrossEntropyLoss().to(device) 
optimizer = optim.Adam(model.parameters(), lr= args.lr, betas=(0.9,0.999),eps=1e-8)

def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data = data.to(device=device)
        target = target.to(device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.max(target,1)[1])
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            #data = data.view(-1,seq_length, input_channels)
            if args.permute:
                data = data[:, :, permute]
            data, target = data.to(device=device), target.to(device=device)
            #print('target:', target)
            total += target.size(0) 
            output = model(data)
            #print('output shape:', output.shape)
            test_loss += criterion(output, torch.max(target,1)[1]).item()
            pred = output.data.max(1, keepdim=True)[1]
            #print('pred:', pred)
            correct +=pred.eq(torch.max(target,1, keepdim=True)[1]).sum().item()
            

        test_loss /= len(valid_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total,
            100. * correct / total))
        return test_loss

if __name__ == "__main__":
    min_loss = 1000
    for epoch in range(1, epochs+1):
        train(epoch)
        val_loss = test()
        if val_loss < min_loss:
            min_loss = val_loss
            print("save model")
            torch.save(model, 'checkpoints_nd_pth/model'+str(epoch)+'.pth')
        """
        if epoch % 10 == 0:
            #lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        """