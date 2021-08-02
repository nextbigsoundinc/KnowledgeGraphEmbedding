#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

from torch.nn.init import xavier_normal_, xavier_uniform_

from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


class ComplExDeep(nn.Module):

    def __init__(self,
                 input_neurons):

        super(ComplExDeep, self).__init__()
        self.input_neurons = int(input_neurons * 0.5)
        self.hidden_drop = torch.nn.Dropout(0.5)
        self.input_drop = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(self.input_neurons, 256)
        self.fc2 = torch.nn.Linear(256, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, head, relation,  tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # print('re_head.shape=', re_head.shape)
        # print('im_head.shape=', im_head.shape)
        # print('re_relation.shape=', re_relation.shape)
        # print('im_relation.shape=', im_relation.shape)
        # print('re_tail.shape=', re_tail.shape)
        # print('im_tail.shape=', im_tail.shape)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_head * re_score
            im_score = im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_tail * re_score
            im_score = im_tail * im_score
        score1 = re_score * im_score
        score1 = score1.sum(dim=2)
        # print('re_score.shape=', re_score.shape)
        # print('im_score.shape=', im_score.shape)
        score = torch.stack([re_score, im_score], dim=0)  # # 2 * 1024 * 256 * hid_dim
        # print('score.shape=', score.shape)
        score = score.norm(dim=0)  # 1024 * 256 * hid_dim
        # print('score.shape=', score.shape)

        x = F.relu(self.fc1(score))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        score = x.sum(dim=2)
        score += score1
        # print('score1.shape=', score1.shape)
        return score


class ConvELayer(nn.Module):

    def __init__(self,
                 entity_embedding,
                 img_entity_embedding,
                 relation_embedding,
                 img_relation_embedding,
                 embedding_dim,
                 nentity,
                 embedd_dim_fold=20,
                 input_drop=0.2,
                 hidden_drop=0.3,
                 feat_drop=0.3,
                 hidden_size=58368):

        super(ConvELayer, self).__init__()
        self.input_neurons = int(embedding_dim)
        self.entity_embedding = entity_embedding
        self.img_entity_embedding = img_entity_embedding
        self.relation_embedding = relation_embedding
        self.img_relation_embedding = img_relation_embedding
        self.nentity = nentity
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)
        self.loss = torch.nn.BCEWithLogitsLoss()  # modify: cosine embedding loss / triplet loss
        self.embedding_dim = embedding_dim
        self.entity_embedding = entity_embedding
        self.img_entity_embedding = img_entity_embedding
        self.relation_embedding = relation_embedding
        self.img_relation_embedding = img_relation_embedding

        self.emb_dim1 = embedding_dim // embedd_dim_fold  # this is from the original configuration in ConvE
        self.emb_dim2 = embedd_dim_fold

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.input_neurons)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.input_neurons)))
        self.fc = torch.nn.Linear(hidden_size, self.input_neurons)
        # self.fc_real_reduction = torch.nn.Linear(self.input_neurons, 256)
        # self.fc_img_reduction = torch.nn.Linear(self.input_neurons, 256)
        # self.fc_combine = torch.nn.Bilinear(256, 256, self.input_neurons)
        # self.fc_combine_reduce = torch.nn.Linear(self.input_neurons, 256)
        self.fc_score = torch.nn.Linear(256, 32)

    def init(self):
        xavier_normal_(self.entity_embedding.weight.data)
        xavier_normal_(self.relation_embedding.weight.data)
        xavier_normal_(self.img_entity_embedding.weight.data)
        xavier_normal_(self.img_relation_embedding.weight.data)

    def forward(self, head, relation, tail, mode, batch_size, negative_sample_size):
        re_head = self.entity_embedding(head)
        im_head = self.img_entity_embedding(head)
        re_relation = self.relation_embedding(relation)
        im_relation = self.img_relation_embedding(relation)
        re_tail = self.entity_embedding(tail)
        im_tail = self.img_entity_embedding(tail)

        # print('re_head.shape=', re_head.shape)
        # print('im_head.shape=', im_head.shape)
        # print('re_relation.shape=', re_relation.shape)
        # print('im_relation.shape=', im_relation.shape)
        # print('re_tail.shape=', re_tail.shape)
        # print('im_tail.shape=', im_tail.shape)

        if mode == 'head-batch':
            re_head = re_head.view(batch_size, negative_sample_size, -1)
            im_head = im_head.view(batch_size, negative_sample_size, -1)
            re_relation = re_relation.view(batch_size, 1, -1)
            im_relation = im_relation.view(batch_size, 1, -1)
            re_tail = re_tail.view(batch_size, 1, -1)
            im_tail = im_tail.view(batch_size, 1, -1)

            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score.view(batch_size, 1, self.emb_dim2, self.emb_dim1)
            im_score = im_score.view(batch_size, 1, self.emb_dim2, self.emb_dim1)
            # print("re_score_shape=", re_score.shape)
            # print("im_score_shape=", im_score.shape)
            re_entity = re_head
            im_entity = im_head

            # re_score = re_head * re_score
            # im_score = im_head * im_score
        else:
            re_head = re_head.view(batch_size, 1, -1)
            im_head = im_head.view(batch_size, 1, -1)
            re_relation = re_relation.view(batch_size, 1, -1)
            im_relation = im_relation.view(batch_size, 1, -1)
            re_tail = re_tail.view(batch_size, negative_sample_size, -1)
            im_tail = im_tail.view(batch_size, negative_sample_size, -1)
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score.view(batch_size, 1, self.emb_dim2, self.emb_dim1)
            im_score = im_score.view(batch_size, 1, self.emb_dim2, self.emb_dim1)
            # print("re_score_shape=", re_score.shape)
            # print("im_score_shape=", im_score.shape)
            re_entity = re_tail
            im_entity = im_tail

            # re_score = re_tail * re_score
            # im_score = im_tail * im_score

        stacked_inputs = torch.cat([re_score, im_score], 2)

        # print("stacked_inputs=", stacked_inputs.shape)

        x = self.inp_drop(stacked_inputs)
        #print("inp_drop x.shape=", x.shape)
        x = self.conv1(x)
        #print("conv1 x.shape=", x.shape)
        x = self.bn1(x)
        #print("bn1 x.shape=", x.shape)
        x = F.relu(x)
        #print("relu x.shape=", x.shape)
        x = self.feature_map_drop(x)
        #print("feature_map x.shape=", x.shape)
        x = x.view(x.shape[0], -1)  # len * 1152
        #print("x.view x.shape=", x.shape)
        x = self.fc(x)
        #print("fc x.shape=", x.shape)
        # len * 200
        x = self.hidden_drop(x)
        #print("hidden_drop x.shape=", x.shape)
        x = self.bn2(x)
        #print("bn2 x.shape=", x.shape)
        x = F.relu(x)  # bs * 200
        #print("relu2 x.shape=", x.shape)
        # x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))  # len * 200  @ (200 * # ent)  => len *  # ent
        x += self.b.expand_as(x)
        x = x.view(x.shape[0], 1, -1)
        #print("x expand x.shape=", x.shape)
        #print("re_entity.shape=", re_entity.shape)
        re_score = re_entity * x
        #print("re_score = re_entity * x=", re_score.shape)
        #print("im_entity.shape=", im_entity.shape)
        im_score = im_entity * x
        x = torch.stack([re_score, im_score], dim=0)
        x = x.norm(dim=0)
        #print("im_score = im_entity * x=", im_score.shape)
        score = F.relu(self.hidden_drop(self.fc(x)))
        # #print("fc real reduction", re_score.shape)
        # im_score = F.relu(self.hidden_drop(self.fc_img_reduction(im_score)))
        # #print("fc img reduction", im_score.shape)
        # x = F.relu(self.hidden_drop(self.fc_combine(re_score, im_score)))
        # #print("bilinear layer", x.shape)
        # x = F.relu(self.hidden_drop(self.fc_combine_reduce(x)))
        x = self.fc_score(x)
        score = x.sum(dim=2)
        # print('score1.shape=', score1.shape)
        return score


class CoCoELayer(nn.Module):
    def __init__(self, ent_real, ent_img, rel_real, rel_img, negative_sample_size):
        super(CoCoELayer, self).__init__()

        self.ent_real = ent_real
        self.ent_img = ent_img
        self.rel_real = rel_real
        self.rel_img = rel_img

        self.conve_layer_rr = ConvELayer(self.ent_real, self.rel_real)
        self.conve_layer_ri = ConvELayer(self.ent_real, self.rel_img)
        self.conve_layer_ir = ConvELayer(self.ent_img, self.rel_real)
        self.conve_layer_ii = ConvELayer(self.ent_img, self.rel_img)

        #self.last_fc = torch.nn.Linear

        '''
        self.conv1 = torch.nn.Conv2d(2, 32, (3, 3), 1, 0, bias=True)
        self.conv2 = torch.nn.Conv2d(2, 32, (3, 3), 1, 0, bias=True)
        self.conv3 = torch.nn.Conv2d(2, 32, (3, 3), 1, 0, bias=True)
        self.conv4 = torch.nn.Conv2d(2, 32, (3, 3), 1, 0, bias=True)
        self.conv = [self.conv1, self.conv2, self.conv3, self.conv4]

        self.bn00 = torch.nn.BatchNorm2d(1)
        self.bn01 = torch.nn.BatchNorm2d(1)
        self.bn02 = torch.nn.BatchNorm2d(1)
        self.bn03 = torch.nn.BatchNorm2d(1)

            # = [torch.nn.BatchNorm2d(1),torch.nn.BatchNorm2d(1),torch.nn.BatchNorm2d(1),torch.nn.BatchNorm2d(1)]
        self.bn1 = [torch.nn.BatchNorm2d(32), torch.nn.BatchNorm2d(32),torch.nn.BatchNorm2d(32),torch.nn.BatchNorm2d(32)]
        self.bn2 = [torch.nn.BatchNorm1d(self.embedding_dim),torch.nn.BatchNorm1d(self.embedding_dim),torch.nn.BatchNorm1d(self.embedding_dim),torch.nn.BatchNorm1d(self.embedding_dim)]
        self.register_parameter('b', nn.Parameter(torch.zeros(self.nentity)))

        self.fc1 = torch.nn.Linear(hidden_size, self.embedding_dim)
        self.fc2 = torch.nn.Linear(hidden_size, self.embedding_dim)
        self.fc3 = torch.nn.Linear(hidden_size, self.embedding_dim)
        self.fc4 = torch.nn.Linear(hidden_size, self.embedding_dim)
        self.fc = [self.fc1, self.fc2, self.fc3, self.fc4]
        '''
        self.drop_layer = nn.Dropout(0.3)

    def init(self):
        xavier_normal_(self.ent_real.weight.data)
        xavier_normal_(self.ent_img.weight.data)
        xavier_normal_(self.rel_real.weight.data)
        xavier_normal_(self.rel_img.weight.data)

    def forward(self, e1, rel, batch_size, negative_sample_size):
        # e1_real = self.ent_real(e1)         # bs * 200
        # e1_img = self.ent_img(e1)
        # rel_real = self.rel_real(rel)
        # rel_img = self.rel_img(rel)

        '''
        e1_real = self.ent_real(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)  # bs * 1 * 20 * 10
        e1_img = self.ent_img(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)  # bs * 1 * 20 * 10
        rel_real = self.rel_real(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)  # bs * 1 * 20 * 10
        rel_img = self.rel_img(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)  # bs * 1 * 20 * 10

        er = self.inp_drop(self.bn0[0](e1_real))  # bs * 1 * 20 * 10
        ei = self.inp_drop(self.bn0[1](e1_img))
        rr = self.inp_drop(self.bn0[2](rel_real))
        ri = self.inp_drop(self.bn0[3](rel_img))

        r_r = torch.cat([er,rr], dim=1)      # bs * 2 * 20 * 10   real_real
        r_i = torch.cat([er,ri], dim=1)                         # real_img
        i_r = torch.cat([ei,rr], dim=1)                         # img_real
        i_i = torch.cat([ei,ri], dim=1)                         # img_img

        for i, fm in enumerate([r_r, r_i, i_r, i_i]):
            fm = self.feature_map_drop(F.relu(self.bn1[i](self.conv[i](fm))))      # bs * 32 * 18 * 8
            fm = fm.view(fm.shape[0], -1)       # bs * 4608
            fm = F.relu(self.bn2[i](self.hidden_drop(self.fc[i](fm))))              # bs * 200

        '''

        rr = self.conv_layer_rr(e1, rel, batch_size, negative_sample_size) # bs * nentity
        ri = self.conv_layer_ri(e1, rel, batch_size, negative_sample_size) # bs * nentity
        ir = self.conv_layer_ir(e1, rel, batch_size, negative_sample_size) # bs * nentity
        ii = self.conv_layer_ii(e1, rel, batch_size, negative_sample_size) # bs * nentity

        rrr = torch.mm(rr, self.conv_layer_rr.entity_embedding.weight.transpose(1, 0))  # rr: bs * 200, tail...': 200 * (1024*256) =>
        rii = torch.mm(ri, self.conv_layer_ii.entity_embedding.weight.transpose(1, 0))
        iri = torch.mm(ir, self.conv_layer_ri.entity_embedding.weight.transpose(1, 0))
        iir = torch.mm(ii, self.conv_layer_ir.entity_embedding.weight.transpose(1, 0))
        score = rrr + rii + iri - iir

        return score




# bash run.sh train RotatE FB15k    0       0      1024        256               1000         24.0    1.0   0.0001 150000         16               -de
#               1     2      3       4      5        6          7                   8          9       10     11     12           13
#              mode model  dataset  GPU  saveid    batchsize   neg_sample_size  hidden_dim    gamma   alpha   lr    Max_steps  test_batchsize

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=True, double_relation_embedding=True, neg_sample_size=256):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )


        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'CoCoE':
            self.cocoe_layer = ComplExDeep(self.entity_dim)

        elif model_name == 'ConvE':
            self.conve_layer = ConvELayer(self.entity_embedding,
                                          self.img_entity_embedding,
                                          self.relation_embedding,
                                          self.img_relation_embedding,
                                          self.hidden_dim,
                                          self.nentity)
            self.conve_layer.init()

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'ConvE', 'CoCoE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample           # tail part: 1024 * 3 (1024 positive triples)
                                                    # head part: 1024 * 256 (each row represent neg sample ids of the corresponding positive triple)
                                                    # in other words, each positive triplet have 256 negetive triplets
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)     # 1024 256

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)                # indexes * entity_dim: (1024 * 256) * entity_dim
                                                                        # corrupted head

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)                                              # 1024 * 1 * entity_dim

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)                                              # 1024 * 1 * entity_dim

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'ConvE': self.ConvE,
            'CoCoE': self.CoCoE
        }
        
        if self.model_name in model_func:
            
            score = model_func[self.model_name](head, relation, tail, mode, batch_size, negative_sample_size)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def ConvE(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        # TransE
        #print(mode)
        # print(head.shape)
        # print(relation.shape)
        # print(tail.shape)

        #       mode:       single
        #       head.shape: torch.Size([512, 1, 1000])
        #   relation.shape: torch.Size([512, 1, 1000])
        #       tail.shape: torch.Size([1, 1000])
        #      score.shape: torch.Size([512, 1, 1000])
        # score.sum(dim=2): torch.Size([512, 1])
        #
        #
        #             mode: tail-batch
        #       head.shape: torch.Size([512, 1, 1000])
        #   relation.shape: torch.Size([512, 1, 1000])
        #       tail.shape: torch.Size([512, 1024, 1000])
        #      score.shape: torch.Size([512, 1024, 1000])
        # score.sum(dim=2): torch.Size([512, 1024)
        #
        #
        #             mode: head-batch
        #       head.shape: torch.Size([8, 40943, 1000])
        #   relation.shape: torch.Size([8, 1, 1000])
        #       tail.shape: torch.Size([8, 1, 1000])
        #      score.shape: torch.Size([8, 40943, 1000])
        # score.sum(dim=2): torch.Size([8, 40943])
        #

        score = self.conve_layer(head, relation, tail, mode, batch_size, negative_sample_size)

        return score  # len * # ent

    def CoCoE(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        score = self.cocoe_layer(head, relation, tail, mode)
        return score

    def TransE(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        #print(mode)
        #print(head.shape)
        #print(relation.shape)
        #print(tail.shape)
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        #print(score.shape)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        #print(score.shape)
        return score

    def DistMult(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    # def ConvE(self, head, relation, tail, mode):

    def ComplEx(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        # head (if corrupted): 1024 * 256 * ent_dim         (ent_dim = hidden_dim * 2)
        # relation: 1024 * 1 * hidden_dim
        # tail: 1024 * 1 * ent_dim
        # mode: (here assume to be 'head_batch')

        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)          # both 1024 * 256 * hid_dim
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)          # both 1024 * 1 * hid_dim

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)             # 1024 * 1 * hid_dim
        im_relation = torch.sin(phase_relation)             # 1024 * 1 * hid_dim

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail       # 1024 * 1 * hid_dim
            im_score = re_relation * im_tail - im_relation * re_tail        # 1024 * 1 * hid_dim
            re_score = re_score - re_head                                   # 1024 * 256 * hid_dim
            im_score = im_score - im_head                                   # 1024 * 256 * hid_dim
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)          # # 2 * 1024 * 256 * hid_dim
        score = score.norm(dim = 0)                                 # 1024 * 256 * hid_dim

        score = self.gamma.item() - score.sum(dim = 2)              # 1024 * 256
        return score

    def pRotatE(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    @staticmethod
    def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method

        """
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        label_shape = torch.Size((true_labels.size(0), classes))
        indices = torch.zeros(true_labels.size(0), 1).long()
        # print("indices=", indices)
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=true_labels.device)
            true_dist.fill_(smoothing / (classes - 1))
            true_dist.scatter_(1, indices, confidence)
        return true_dist


    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)

        if model.model_name not in ['ConvE', 'CoCoE']:

            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (
                        F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                        * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)

            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            if args.uni_weight:
                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()
            else:
                positive_sample_loss = - (
                            subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = - (
                            subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2

            if args.regularization != 0.0:
                # Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                        model.entity_embedding.norm(p=3) ** 3 +
                        model.relation_embedding.norm(p=3).norm(p=3) ** 3
                )
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}

            loss.backward()
            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }

        else:
            batch_size = positive_sample.size(0)
            pred = torch.cat([positive_score, negative_score], dim=1)
            #print("pred=.shape", pred.shape)
            target = torch.zeros(batch_size, dtype=torch.int64)
            #print('target.shape=', target.shape)
            # for batch in range(batch_size):
            #     target[batch][0] = 1

            # smooth_target = KGEModel.smooth_one_hot(target, pred.size(1), 0.001)
            #print("pred=", pred)
            #print('targets=', target)
            if args.cuda:
                pred = pred.cuda()
                target = target.cuda()
            loss = model.loss(pred, target)
            #print("loss=", loss)
            loss.backward()
            log = {
                'positive_sample_loss': 0,
                'negative_sample_loss': 0,
                'loss': loss.item()
            }

        optimizer.step()

        return log


    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:                # pos_sample: bs * 3
                        if args.cuda:                                                                       # neg_sample: bs * 256
                                                                             # bs * (1 good trip, 256 bad trip)
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        #print("score=[{}]".format(score))
                        score += filter_bias
                        #print("score+filter bias=[{}]".format(score))

                        # print('\n**************************\nScore_dim: ', score.size(), '\n**************************\n')


                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)


                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)


                        for i in range(batch_size):
                            #print("negative sample shape=[{}]".format(negative_sample.shape))
                            #Notice that argsort is not ranking

                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            #print(argsort[i, :])
                            assert ranking.size(0) == 1
                            #print('ranking=[{}]'.format(ranking))
                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                'HITS@1000': 1.0 if ranking <= 1000 else 0.0
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))


                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
