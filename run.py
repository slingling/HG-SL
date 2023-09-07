# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:42:32 2021

@author: Ling Sun
"""

import argparse
import time
import numpy as np 
import Constants
import torch
from dataLoader import DataReader, GraphReader, DataLoader
from Metrics import Metrics
from HGSL import HGSL
from Optim import ScheduledOptim
import torch.nn.functional as F


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()


parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='poli')
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-early_time', type=int, default=10)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-save_path', default= "./checkpoint/fake_detection.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')

opt = parser.parse_args()


def train_epoch(model, training_data, hypergraph_list, optimizer):
    # train
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(training_data):
        # data preparing
        tgt, labels = (item.to(Constants.device) for item in batch)
        # training
        optimizer.zero_grad()
        pred= model(tgt, hypergraph_list)
        
        # loss
        loss = F.nll_loss(pred, labels.squeeze())
        loss.backward()

        # parameter update
        optimizer.step()
        optimizer.update_learning_rate()

        total_loss += loss.item()

    return total_loss

def train_model(HGSL, data_name):
    # ========= Preparing DataLoader =========#
    train, valid, test, news_size,  = DataReader(data_name)
    hypergraph_list, user_size = GraphReader(data_name)

    train_data = DataLoader(train, batch_size=opt.batch_size, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, cuda=False)


    opt.user_size = user_size
    opt.edge_size = news_size+1

    # ========= Preparing Model =========#
    model = HGSL(opt)
    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.to(Constants.device)

    validation_history = 0.0
    best_scores = {}
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(model, train_data, hypergraph_list, optimizer)
        print('  - (Training) loss: {loss: 8.5f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss,
            elapse=(time.time() - start) / 60))

        if epoch_i > 5:
            #start = time.time()
            scores = test_epoch(model, valid_data, hypergraph_list)
            print('  - (Validation) ')
            for metric in scores.keys():
                print(metric + ': ' + "%.5f"%(scores[metric]*100) +"%")

            print('  - (Test) ')
            scores = test_epoch(model, test_data, hypergraph_list)
            for metric in scores.keys():
                print(metric + ': ' + "%.5f"%(scores[metric]*100) +"%")

            if validation_history <= sum(scores.values()):
                print("Best Test Accuracy:{}% at Epoch:{}".format(round(scores["Acc"]*100,5), epoch_i))
                validation_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)
                
    print(" - (Finished!!) \n Best scores: ")
    for metric in best_scores.keys():
        print(metric + ': ' + "%.5f"%(best_scores[metric]*100) +"%")
                
def test_epoch(model, validation_data, hypergraph_list):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    k_list = ['Acc', 'F1', 'Pre', 'Recall']
    for k in k_list:
        scores[k] = 0

    n_total_words = 0
    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            tgt, labels = (item.to(Constants.device) for item in batch)
            y_labels = labels.detach().cpu().numpy()
            # forward
            pred = model(tgt, hypergraph_list)
            y_pred = pred.detach().cpu().numpy()
            n_total_words += len(tgt)

            scores_batch= metric.compute_metric(y_pred, y_labels)
            for k in k_list:
                scores[k] += scores_batch[k]

    for k in k_list:
        scores[k] = scores[k] / n_total_words
    return scores

if __name__ == "__main__": 
    model = HGSL
    train_model(model, opt.data_name)



