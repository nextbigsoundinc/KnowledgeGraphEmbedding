from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

x = torch.randn(2,2)
print(x)
y = x.view(-1)
print(y)

x = torch.tensor([[1.0,2,3],[4,5,6]])
y = torch.tensor([[1,3,5], [2,4,6]])
print(torch.stack([x,y], dim=2))

print(x.norm(dim=1))

print('mean: ', x.mean())

z = x.repeat(3,2)
print(z)

a = torch.tensor([1,2,3])
print(a.repeat(2,1))

b = y.T
print(b)
b[:,1] = a
print(b)


print('\n\n\n')

a = torch.nn.Embedding(3,5)
print('Emb: ', a.weight)
print(a.weight.detach().cpu().numpy())


print('\n\n\n')
a1, a2, a3 = torch.tensor([1,2,3]), torch.tensor([4,5,6]), torch.tensor([7,8,9])
a = torch.cat([a1,a2,a3], dim= 0)
print(a)

print('\n\n\n')
a = torch.tensor([1.0])
b = torch.randn((3,2))
# a = 1.0
print(a.expand_as(b))
print(b)