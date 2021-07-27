
#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_normal_, xavier_uniform_

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class ConvModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, input_drop, hidden_drop, feat_drop, emb_dim1, hidden_size):
        super(ConvModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.embedding_dim = hidden_dim

        self.entity_embedding = nn.Embedding(self.nentity, self.entity_dim, padding_idx=0 )
        self.relation_embedding = nn.Embedding(self.nrelation, self.relation_dim, padding_idx=0 )

        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)
        self.loss = torch.nn.BCELoss()  # modify: cosine embedding loss / triplet loss
        self.emb_dim1 = emb_dim1             # this is from the original configuration in ConvE
        self.emb_dim2 = self.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.nentity)))
        self.fc = torch.nn.Linear(hidden_size, self.embedding_dim)


    def init(self):
        xavier_normal_(self.entity_embedding.weight.data)
        xavier_normal_(self.relation_embedding.weight.data)


    def forward(self, e1, rel):

        e1_embedded = self.entity_embedding(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)           # len(e1) *  1 * 20 * 10
        rel_embedded = self.relation_embedding(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)       # len(rel) * 1 * 20 * 10       len(e1) = len(rel)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)                                  # len * 1 * 40 * 10

        stacked_inputs = self.bn0(stacked_inputs)                   # len * 1 * 40 * 10
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)                                           # len * 32 * 38 * 8
                                                                    # pooling ... len * 32 * 19 * 4
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)                                  # len * 9728
        x = self.fc(x)                                              # len * 200
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # 1 * 200
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))  # len * 200  @ (200 * # ent)  => len *  # ent
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred         # len * # ent


    @staticmethod
    def train_step( model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()


        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if mode == 'head-batch': return None
        # mode = 'single'
        bs = positive_sample.size(0)        # e.g., 1024
        ns = negative_sample.size(1)        # e.g., 256

        #  positive_sample: 1024 * 3
        #  negative_sample: 1024 * 256

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        e1 = positive_sample[:,0]
        rel = positive_sample[:, 1]
        pred = model(e1, rel)           # bs * #ent
        input = torch.empty((bs, (1 + ns)))
        for i in range(bs):
            input[i,0] = pred[i,positive_sample[i,2]]           # input: 1st column is score of the true entity
            input[i,1:] = pred[i, negative_sample[i]]           # input: subsequent columns are scores of negative entities
        target = torch.tensor([1.0] + [0.] * ns)
        target = target.repeat(bs, 1)

        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = model.loss(input, target)

        # print('\n**************************\npos_score_dim: ', positive_score.size(),
        #       '\t\tneg_score_dim: ', negative_score.size(),
        #       '\n**************************\n')


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

        optimizer.step()

        log = {
            **regularization_log,
            # 'positive_sample_loss': positive_sample_loss.item(),
            # 'negative_sample_loss': negative_sample_loss.item(),
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
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
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

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
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
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if mode != 'tail-batch':
                            continue
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        e1, rel = positive_sample[:,0], positive_sample[:,1]
                        score = model(e1, rel)                              # bs * #ent



                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        # print('\n**************************\nScore_dim: ', score.size(), '\n**************************\n')

                        argsort = torch.argsort(score, dim=1, descending=True)

                        # 16 * 14951 , each row : [234, 24, 0, 190 ..., ]

                        positive_arg = positive_sample[:, 2]

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                'HITS@1000': 1.0 if ranking <= 1000 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
