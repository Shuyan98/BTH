import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from bert import BERT
from args import nbits,hidden_size

class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = 1-(ctx.input==0)
        mask = Variable(mask).cuda().float()
        grad_output = grad_output*mask
        return grad_output, None, None

class BERTLM(nn.Module):
    """
    BERT Language Model
    """

    def __init__(self, frame_size):

        super(BERTLM,self).__init__()
        self.bert1 = BERT(frame_size, hidden=hidden_size, n_layers=1, attn_heads=1, dropout=0.1)
        self.bert2 = BERT(frame_size, hidden=hidden_size, n_layers=1, attn_heads=1, dropout=0)
        self.project = nn.Linear(hidden_size,frame_size)
        self.restore = nn.Linear(nbits, frame_size)
        self.binary = nn.Linear(hidden_size, nbits)
        self.activation = self.binary_tanh_unit

    def hard_sigmoid(self,x):
        y = (x+1.)/2.
        y[y>1] = 1
        y[y<0] = 0
        return y

    def binary_tanh_unit(self,x):
        y = self.hard_sigmoid(x)
        out = 2.*Round3.apply(y)-1.
        return out
    def forward(self, x):

        hid1 = self.bert1(x)
        hid2 = self.bert2(x)
        hid = hid1
        z = self.binary(hid)
        bb = self.activation(z)
        frame = self.restore(bb)

        return bb, frame, hid
