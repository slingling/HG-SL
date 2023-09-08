
"""
Created on Nov 1 22:28:02 2021

@author: Ling Sun
"""
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle

class Options(object):

    def __init__(self, data_name='poli'):
        self.news_centered = 'data/' + data_name + '/Processed/news_centered.pickle'
        self.user_centered = 'data/' + data_name + '/Processed/user_centered.pickle'

        #self.user_features = 'data/' + data_name + '/user_features.pickle'
        self.test_data = 'data/' + data_name + '/Processed/test_processed.pickle'
        self.valid_data = 'data/' + data_name + '/Processed/valid_processed.pickle'
        self.train_data = 'data/' + data_name + '/Processed/train_processed.pickle'
        self.news_features = 'data/' + data_name + '/struct_temp.pkl'
        self.news_mapping = 'data/' + data_name + '/news_mapping.pickle'

        self.save_path = ''

def DataReader(data_name):
    options = Options(data_name)
    with open(options.train_data, 'rb') as f:
        train_data = pickle.load(f)
    with open(options.valid_data, 'rb') as f:
        valid_data = pickle.load(f)
    with open(options.test_data, 'rb') as f:
        test_data = pickle.load(f)

    #print(train_data)

    total_size = len(train_data[0])+len(test_data[0])+len(valid_data[0])

    print("news cascades size:%d " % (total_size))
    print("train size:%d " % (len(train_data[0])))
    print("test and valid size:%d " % (len(test_data[0])+len(valid_data[0])))

    return train_data, valid_data, test_data, total_size

def FeatureReader(data_name):
    options = Options(data_name)
    with open(options.news_mapping, 'rb') as handle:
        n2idx = pickle.load(handle)
    '''Spread status: S1, S2, T1, T2 
    Structural：(S1)number of sub-cascades, (S2)proportion of non-isolated cascades;
    Temporal:  (T1) duration of spread,(T2) the average response time from tweet to retweet'''
    with open(options.news_features, 'rb') as f:
        features = np.array(pickle.load(f))
        news_size = len(features)
        spread_status = np.zeros((news_size + 1, 5))
        for news in features:
            #print(news)
            spread_status[n2idx[news[0]]]=np.array(news[1:])
            #print(spread_status[n2idx[news[0]]])
    return spread_status

def GraphReader(data_name):
    options = Options(data_name)
    with open(options.news_centered, 'rb') as f:
        news_centered_graph = pickle.load(f)

    with open(options.user_centered, 'rb') as f:
        user_centered_graph = pickle.load(f)

    useq, user_inf = (item for item in user_centered_graph)
    seq, timestamps, user_level, news_inf = (item for item in news_centered_graph)
    spread_status = FeatureReader(data_name)

    user_size = len(useq)
    user_inf[user_inf>0] = 1
    act_level = user_inf[1:].sum(1)
    avg_inf = np.append([0],act_level)

    news_centered_graph = [seq, timestamps, user_level]
    user_centered_graph = [useq, news_inf, avg_inf]


    return [[torch.LongTensor(i).to(Constants.device) for i in news_centered_graph], [torch.LongTensor(i).to(Constants.device) for i in user_centered_graph],
            torch.LongTensor(spread_status).to(Constants.device)], user_size

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
        self, data, batch_size=64, cuda=True, test=False):
        self._batch_size = batch_size
        self.idx = data[0]
        self.label = data[1]
        self.test = test
        self.cuda = cuda


        self._n_batch = int(np.ceil(len(self.idx) / self._batch_size))
        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def seq_to_tensor(insts):

            inst_data_tensor = Variable(
                torch.LongTensor(insts), volatile=self.test)

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            idx = self.idx[start_idx:end_idx]
            labels = self.label[start_idx:end_idx]
            idx = seq_to_tensor(idx)
            labels = seq_to_tensor(labels)

            return idx, labels
        else:

            self._iter_count = 0
            raise StopIteration()
