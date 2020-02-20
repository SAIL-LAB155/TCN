import torch
import sys
sys.path.append("../../")
from TCN.swim_classify.model import TCN
import numpy as np

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

if use_cuda:
    print('use CUDA')
else:
    print('use CPU')


model_pth = '/home/hkuit104/Desktop/TCN-master/TCN/swim_classify/checkpoints_nd/model198.pt'
model = torch.load(model_pth)
input_pth = '/home/hkuit104/Desktop/TCN-master/TCN/swim_classify/test/carol1.txt'

def get_input_data(input_pth):
    keypoint = np.loadtxt(input_pth).reshape(34, -1).astype(np.float32)#.reshape(34, -1).astype(np.float32)[:,:30]
    print('keypoint shape:', keypoint.shape)
    seq = torch.from_numpy(keypoint)#.to(device=device)
    #seq = seq.permute(1,0)
    seq = seq.unsqueeze(0).to(device=device)
    
    return seq

def predict(seq):
    model.eval()
    output = model(seq)
    pred = output.data.max(1, keepdim=True)[1]
    return pred


if __name__ == '__main__':
    seqs = get_input_data(input_pth)
    i = 0
    print('seqs shape',seqs.shape)
    while seqs.size(2):
        seq = seqs[:,:,i:i+30]
        i = i + 30
        seq_true = seq
        if seq_true.size(2) < 30:
            break
        if seq.shape[2] < 30:
            tmp = torch.zeros(1, 34, 30 - seq.shape[2] , device=device)
            seq = torch.cat(((seq), tmp), dim=2)
        print('seq shape:', seq.shape)
        prediction = predict(seq)
        print(prediction)

