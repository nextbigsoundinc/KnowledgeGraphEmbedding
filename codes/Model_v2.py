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


class ConvELayer(nn.Module):

    def __init__(self, entity_embedding, relation_embedding,
                 input_drop=0.2, hidden_drop=0.3,
                 feat_drop=0.2,
                 emb_dim1=20,
                 test_batch_size1=512,
                 test_batch_size2=8,
                 hidden_size=9728):

        super(ConvELayer, self).__init__()
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)
        # self.loss = torch.nn.BCELoss()  # modify: cosine embedding loss / triplet loss
        self.emb_dim1 = emb_dim1             # this is from the original configuration in ConvE

        self.nentity = self.entity_embedding.weight.shape[0]
        self.embedding_dim = self.entity_embedding.weight.shape[1]
        self.emb_dim2 = self.embedding_dim // self.emb_dim1

        self.adavgpool1 = torch.nn.AdaptiveAvgPool2d((8,1))
        self.conv1 = torch.nn.Conv2d(2, 32, (3, 3), 1, 0, bias=True)
        self.mpool = torch.nn.MaxPool2d(2, stride=2)

        self.bn0 = torch.nn.BatchNorm2d(2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc = torch.nn.Linear(6912, self.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.nentity)))

    def init(self):
        xavier_normal_(self.entity_embedding.weight.data)
        xavier_normal_(self.relation_embedding.weight.data)

    def forward(self, head, rel,  batch_size, negative_sample_size):
        head_embedding = self.entity_embedding(head).view(batch_size, negative_sample_size,
                                                          self.emb_dim1, self.emb_dim2)
        rel_embedding = self.relation_embedding(rel).view(batch_size, 1,
                                                          self.emb_dim1, self.emb_dim2)  # bs * 1 * 200       len(e1) = len(rel)

        # print("head embedding=[", head_embedding.shape, "]")
        # print("rel embedding=[", rel_embedding.shape, "]")
        stacked_inputs = torch.cat([head_embedding, rel_embedding], 1)                                  # len * 2 * 20 * 10
        # print("stacked=[", stacked_inputs.shape, "]")


        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)                                  # len * 1152
        x = self.fc(x)                   # len * 200
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # bs * 200
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))  # len * 200  @ (200 * # ent)  => len *  # ent
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred

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
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
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

        if model_name in ['ConvE', 'CoCoE']:
            self.entity_embedding = nn.Embedding(self.nentity, self.entity_dim, padding_idx=0)
            self.relation_embedding = nn.Embedding(self.nrelation, self.relation_dim, padding_idx=0)
            self.conve_layer = ConvELayer(self.entity_embedding, self.relation_embedding)
            self.conve_layer.init()
            self.loss = torch.nn.BCELoss()
            if model_name == 'CoCoE':
                self.img_entity_embedding = nn.Embedding(self.nentity, self.entity_dim, padding_idx=0)
                self.img_relation_embedding = nn.Embedding(self.nrelation, self.relation_dim, padding_idx=0)
                self.cocoe_layer = CoCoELayer(self.entity_embedding,
                                              self.img_entity_embedding,
                                              self.relation_embedding,
                                              self.img_relation_embedding, hidden_dim)
                self.cocoe_layer.init()

        else:
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

        batch_size=0
        negative_sample_size=0

        if self.model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'ComplEx']:

            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                head = sample[:, 0]
                relation = sample[:, 1]
                tail = sample[:, 2]

            elif mode == 'head-batch':
                tail_part, head_part = sample  # tail part: 1024 * 3 (1024 positive triples)
                # head part: 1024 * 256 (each row represent neg sample ids of the corresponding positive triple)
                # in other words, each positive triplet have 256 negetive triplets
                #print("head part=[", head_part,"]")
                #print("tail part=[", tail_part,"]")
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)  # 1024 256
                head = head_part.view(-1)
                relation = tail_part[:, 1]
                tail = tail_part[:, 2]

            elif mode == 'tail-batch':
                head_part, tail_part = sample      # tail_part: bs * neg_sample_size        #1024*256
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                head = head_part[:, 0]          # 1024
                relation = head_part[:, 1]      # 1024
                tail = tail_part.view(-1)       # 1024 * 256

            else:
                raise ValueError('mode %s not supported' % mode)

            # print("mode=[",mode,"]")

        else:

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

        if mode == 'head-batch':
            # head_rel_embeddings = self.conve_layer(head, relation, batch_size, negative_sample_size)
            # score = torch.mm(head_rel_embeddings,
            #                  self.conve_layer.entity_embedding.weight.transpose(1,
            #                                                                     0))  # len * 200  @ (200 * # ent)  => len *  # ent
            # print(score.shape)
            # score = score[:, tail]
            # print(score.shape)
            # # score = score.sum(dim=1).view(batch_size, -1)
            multi_head = list(torch.tensor_split(head, negative_sample_size))
            a_head = multi_head.pop(0)
            scores = list()
            single_score_all = self.conve_layer(a_head, relation, -1, 1).view(-1)
            single_score_tail = torch.index_select(input=single_score_all, dim=0, index=tail)
            single_score_tail = single_score_tail.view(batch_size, 1)
            scores.append(single_score_tail)
            del a_head
            while (len(multi_head) > 0):
                a_head = multi_head.pop(0)
                single_score_all = self.conve_layer(a_head, relation, -1, 1).view(-1)
                single_score_tail = torch.index_select(input=single_score_all, dim=0, index=tail)
                single_score_tail = single_score_tail.view(batch_size, 1)
                # print("single_score=[", single_score_tail.shape, "]")
                scores.append(single_score_tail)
                score_stack = torch.cat(scores, dim=1)
                # print("score_stack=[", score_stack.shape, "]")
                del single_score_all
                del single_score_tail
                del scores
                del a_head
                scores = list()
                scores.append(score_stack)
                if (len(multi_head) % 1000) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            del multi_head
            score = torch.cat(scores, dim=1)


            # print("score=[", score.shape, "]")
        else:
            score = self.conve_layer(head, relation, -1, 1)

        return score  # len * # ent

    def CoCoE(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        if mode == 'head-batch':
            multi_head = list(torch.tensor_split(head, negative_sample_size))
            a_head = multi_head.pop(0)
            scores = list()
            single_score_all = self.cocoe_layer(a_head, relation, -1, 1)
            single_score_tail = single_score_all[:, tail]
            single_score_tail = single_score_tail.sum(dim=1).view(batch_size, -1)
            scores.append(single_score_tail)
            del a_head
            while (len(multi_head) > 0):
                a_head = multi_head.pop(0)
                single_score_all = self.conve_layer(a_head, relation, -1, 1)
                single_score_tail = single_score_all[:, tail]
                single_score_tail = single_score_tail.sum(dim=1).view(batch_size, -1)
                # print("single_score=[", single_score_tail.shape, "]")
                scores.append(single_score_tail)
                score_stack = torch.cat(scores, dim=1)
                # print("score_stack=[", score_stack.shape, "]")
                del single_score_all
                del single_score_tail
                del scores
                scores = list()
                scores.append(score_stack)
                del a_head
                if (len(multi_head) % 1000) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            del multi_head
            score = torch.cat(scores, dim=1)
            #print("score=[", score.shape, "]")
        else:
            score = self.concoe_layer(head, relation, -1, 1)
            #print("score=[", score.shape, "]")
            score = score[:, tail]
            score = score.sum(dim=1).view(batch_size, -1)
            #print("score=[", score.shape, "]")

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

        if model.model_name not in ['ConvE', 'CoCoE']:
            negative_score = model((positive_sample, negative_sample), mode=mode)
            positive_score = model(positive_sample)

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
            pred = model(positive_sample)
            batch_size = pred.size(0)  # e.g., 1024
            targets = torch.zeros(batch_size, pred.size(1))
            for batch in range(batch_size):
                targets[batch][positive_sample[batch][2]] = 1.0

            if args.cuda:
                pred = pred.cuda()
                targets = targets.cuda()

            loss = model.loss(pred, targets)
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
            # test_dataloader_head = DataLoader(
            #     TestDataset(
            #         test_triples,
            #         all_true_triples,
            #         args.nentity,
            #         args.nrelation,
            #         'head-batch'
            #     ),
            #     batch_size=args.test_batch_size,
            #     num_workers=max(1, args.cpu_num//2),
            #     collate_fn=TestDataset.collate_fn
            # )

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
            
            # test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            test_dataset_list = [test_dataloader_tail]
            
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

                        if model.model_name not in ['ConvE', 'CoCoE']:

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
                        else:
                            score = model((positive_sample, negative_sample), mode)
                            #print("score test=[{}]".format(score.shape))

                            # Explicitly sort all the entities to ensure that there is no test exposure bias
                            argsort = torch.argsort(score, dim=1, descending=True)

                            if mode == 'head-batch':
                                positive_arg = positive_sample[:, 0]
                            elif mode == 'tail-batch':
                                positive_arg = positive_sample[:, 2]
                            else:
                                raise ValueError('mode %s not supported' % mode)

                            for i in range(batch_size):
                                # print("negative sample shape=[{}]".format(negative_sample.shape))
                                # Notice that argsort is not ranking

                                ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                                print("entity_id sort=[{}]".format(argsort[i, :]))
                                assert ranking.size(0) == 1
                                print("true tail id=[{}]".format(positive_arg[i]))
                                # ranking + 1 is the true ranking used in evaluation metrics
                                ranking = 1 + ranking.item()

                                stats_dict = {
                                    'MRR': 1.0 / ranking,
                                    'MR': float(ranking),
                                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                    'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                    'HITS@1000': 1.0 if ranking <= 1000 else 0.0
                                }
                                print(stats_dict)
                                logs.append(stats_dict)

                            if step % args.test_log_steps == 0:
                                logging.info(
                                    'Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
