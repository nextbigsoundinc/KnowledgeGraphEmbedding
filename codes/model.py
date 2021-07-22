#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

from torch.nn.init import xavier_normal_, xavier_uniform_


class ConvLayer(nn.Module):

    def __init__(self, entity_embedding, relation_embedding,
                 input_drop=0.2, hidden_drop=0.3,
                 feat_drop=0.2,
                 emb_dim1=20,
                 hidden_size=9728):

        super(ConvLayer, self).__init__()
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

        self.conv1 = torch.nn.Conv2d(2, 32, (3, 3), 1, 0, bias=True)
        self.mpool = nn.MaxPool2d(2, stride=2)

        self.bn0 = torch.nn.BatchNorm2d(2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc = torch.nn.Linear(hidden_size, self.embedding_dim)


    def init(self):
        xavier_normal_(self.entity_embedding.weight.data)
        xavier_normal_(self.relation_embedding.weight.data)


    def forward(self, e1, rel, batch_size, negative_sample_size):

        e1_embedded = self.entity_embedding(e1).view(batch_size,
                                                     negative_sample_size,
                                                     self.emb_dim1,
                                                     self.emb_dim2)           # len(e1) *  1 * 20 * 10
        rel_embedded = self.relation_embedding(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)       # len(rel) * 1 * 20 * 10       len(e1) = len(rel)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)                                  # len * 2 * 20 * 10

        stacked_inputs = self.bn0(stacked_inputs)                   # len * 2 * 20 * 10
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)                                           # len * 32 * 18 * 8

        x = self.mpool(x)                                           # len * 32 * 9 * 4

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)                                  # len * 1152
        x = self.fc(x)                                              # len * 200
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # bs * 200

        return x


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
            self.conve_layer_rr = ConvLayer(self.entity_embedding, self.relation_embedding)
            self.register_parameter('b', nn.Parameter(torch.zeros(self.nentity)))

            if model_name == 'CoCoE':
                self.img_entity_embedding = nn.Embedding(self.nentity, self.entity_dim, padding_idx=0)
                self.img_relation_embedding = nn.Embedding(self.nrelation, self.relation_dim, padding_idx=0)
                self.conv_layer_ri = ConvLayer(self.entity_embedding, self.img_relation_embedding)
                self.conv_layer_ir = ConvLayer(self.img_entity_embedding, self.relation_embedding)
                self.conv_layer_ii = ConvLayer(self.img_entity_embedding, self.img_relation_embedding)

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
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
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

        batch_size=0, negative_sample_size=0

        if self.model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'ComplEx']:
            batch_size = sample.size(0)

            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                head = sample[:, 0]
                relation = sample[:, 1]
                tail = sample[:, 2]

            elif mode == 'head-batch':
                tail_part, head_part = sample  # tail part: 1024 * 3 (1024 positive triples)
                # head part: 1024 * 256 (each row represent neg sample ids of the corresponding positive triple)
                # in other words, each positive triplet have 256 negetive triplets
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)  # 1024 256
                head = head_part.view(-1)
                relation = tail_part[:, 1]
                tail = tail_part[:, 2]

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                head = head_part[:, 0]
                relation = head_part[:, 1]
                tail = tail_part.view(-1)

            else:
                raise ValueError('mode %s not supported' % mode)

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

        if mode=='head_batch':
            x = self.conve_layer_rr(head, relation, batch_size, negative_sample_size)
            x = torch.mm(x, self.entity_embedding(tail).weight.transpose(1, 0))  # len * 200  @ (200 * # ent)  => len *  # ent
            x += self.b.expand_as(x)
            score = torch.sigmoid(x)
        else:
            x = self.conve_layer_rr(head, relation, -1, 1)
            x = torch.mm(x, self.entity_embedding(tail).weight.transpose(1, 0))  # len * 200  @ (200 * # ent)  => len *  # ent
            x += self.b.expand_as(x)
            score = torch.sigmoid(x)

        return score  # len * # ent

    def TransE(self, head, relation, tail, mode, batch_size=0, negative_sample_size=0):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
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

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

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
                        score += filter_bias

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
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

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
