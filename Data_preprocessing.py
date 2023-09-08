import numpy as np
import torch
import Constants
import pickle
import os


class Options(object):

    def __init__(self, data_name='poli'):
        self.nretweet = 'data/' + data_name + '/news_centered_data.txt'
        self.uretweet = 'data/' + data_name + '/user_centered_data.txt'
        self.label = 'data/' + data_name + '/label.txt'
        self.news_list = 'data/' + data_name + '/' + data_name + '_news_list.txt'

        self.news_centered = 'data/' + data_name + '/Processed/news_centered.pickle'
        self.user_centered = 'data/' + data_name + '/Processed/user_centered.pickle'

        self.train_idx = torch.from_numpy(np.load('data/' + data_name +'/train_idx.npy'))
        self.valid_idx = torch.from_numpy(np.load('data/' + data_name +'/val_idx.npy'))
        self.test_idx = torch.from_numpy(np.load('data/' + data_name +'/test_idx.npy'))

        self.train = 'data/' + data_name + '/Processed/train_processed.pickle'
        self.valid = 'data/' + data_name + '/Processed/valid_processed.pickle'
        self.test = 'data/' + data_name + '/Processed/test_processed.pickle'

        self.user_mapping = 'data/' + data_name + '/user_mapping.pickle'
        self.news_mapping = 'data/' + data_name + '/news_mapping.pickle'
        self.save_path = ''
        self.embed_dim = 64


def buildIndex(user_set, news_set):
    n2idx = {}
    u2idx = {}

    pos = 0
    u2idx['<blank>'] = pos
    pos += 1
    for user in user_set:
        u2idx[user] = pos
        pos += 1

    pos = 0
    n2idx['<blank>'] = pos
    pos += 1
    for news in news_set:
        n2idx[news] = pos
        pos += 1

    user_size = len(user_set)
    news_size = len(news_set)
    return user_size, news_size, u2idx, n2idx

def Pre_data(data_name, early_type, early, max_len=200):
    options = Options(data_name)
    cascades = {}

    '''load news-centered retweet data'''
    for line in open(options.nretweet):
        userlist = []
        timestamps = []
        levels = []
        infs = []

        chunks = line.strip().split(',')
        cascades[chunks[0]] = []

        for chunk in chunks[1:]:
            try:
                user, timestamp, level, inf = chunk.split()
                userlist.append(user)
                timestamps.append(float(timestamp)/3600/24)
                levels.append(int(level)+1)
                infs.append(inf)
            except:
                user = chunk
                userlist.append(user)
                timestamps.append(float(0.0))
                infs.append(1)
                levels.append(1)
                print('tweet root', chunk)
        cascades[chunks[0]] = [userlist, timestamps, levels, infs]

    news_list = []
    for line in open(options.news_list):
            news_list.append(line.strip())
    cascades = {key: value for key, value in cascades.items() if key in news_list}

    if early:
        if early_type == 'engage':
            max_len = early
        elif early_type == 'time':
            mint = []
            for times in np.array(list(cascades.values()))[:,1]:
                if max(times)-min(times) < early:
                    mint.append(len(times))
                else:
                    for t in times:
                        if t - min(times) >= early:
                            mint.append(times.index(t))
                            break


    '''ordered by timestamps'''
    for idx, cas in enumerate(cascades.keys()):
        max_ = mint[idx] if early and early_type == 'time' and mint[idx] < max_len else max_len
        cascades[cas] = [i[:max_] for i in cascades[cas]]

        order = [i[0] for i in sorted(enumerate(cascades[cas][1]), key=lambda x: float(x[1]))]
        #print(cascades[cas].shape)
        cascades[cas] = [[x[i] for i in order] for x in cascades[cas]]
        #cascades[cas] = cascades[cas][:,order]
        #cascades[cas][1][:] = [cascades[cas][1][i] for i in order]
        #cascades[cas][0][:] = [cascades[cas][0][i] for i in order]
        #cascades[cas][2][:] = [cascades[cas][2][i] for i in order]
        #cascades[cas][3][:] = [cascades[cas][3][i] for i in order]



    ucascades = {}
    '''load user-centered retweet data'''
    for line in open(options.uretweet):
        newslist = []
        userinf = []

        chunks = line.strip().split(',')

        ucascades[chunks[0]] = []

        for chunk in chunks[1:]:
            news, timestamp, inf= chunk.split()
            newslist.append(news)
            userinf.append(inf)

        ucascades[chunks[0]] = np.array([newslist, userinf])

    '''ordered by timestamps'''
    for cas in list(ucascades.keys()):
        order = [i[0] for i in sorted(enumerate(ucascades[cas][1]), key=lambda x: float(x[1]))]
        #ucascades[cas] = cascades[cas][:, order]
        ucascades[cas] = [[x[i] for i in order] for x in ucascades[cas]]
        #ucascades[cas][1][:] = [ucascades[cas][1][i] for i in order]
        #ucascades[cas][0][:] = [ucascades[cas][0][i] for i in order]
    user_set = ucascades.keys()


    if os.path.exists(options.user_mapping):
        with open(options.user_mapping, 'rb') as handle:
            u2idx = pickle.load(handle)
            user_size = len(list(user_set))
        with open(options.news_mapping, 'rb') as handle:
            n2idx = pickle.load(handle)
            news_size = len(news_list)
    else:
        user_size, news_size, u2idx, n2idx = buildIndex(user_set, news_list)
        with open(options.user_mapping, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.news_mapping, 'wb') as handle:
            pickle.dump(n2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for cas in cascades:
        cascades[cas][0] = [u2idx[u] for u in cascades[cas][0]]
    t_cascades = dict([(n2idx[key], cascades[key]) for key in cascades])

    for cas in ucascades:
        ucascades[cas][0] = [n2idx[n] for n in ucascades[cas][0]]
    u_cascades = dict([(u2idx[key], ucascades[key]) for key in ucascades])

    '''load labels'''
    labels = np.zeros((news_size + 1, 1))
    for line in open(options.label):
        news, label = line.strip().split(' ')
        if news in n2idx:
            labels[n2idx[news]] = label

    seq = np.zeros((news_size + 1, max_len))
    timestamps = np.zeros((news_size + 1, max_len))
    user_level = np.zeros((news_size + 1, max_len))
    user_inf = np.zeros((news_size + 1, max_len))
    news_list = [0] + news_list
    for n, s in cascades.items():
        news_list[n2idx[n]] = n
        se_data = np.hstack((s[0], np.array([Constants.PAD] * (max_len - len(s[0])))))
        seq[n2idx[n]] = se_data

        t_data = np.hstack((s[1], np.array([Constants.PAD] * (max_len - len(s[1])))))
        timestamps[n2idx[n]] = t_data

        lv_data = np.hstack((s[2], np.array([Constants.PAD] * (max_len - len(s[2])))))
        user_level[n2idx[n]] = lv_data

        inf_data = np.hstack((s[3], np.array([Constants.PAD] * (max_len - len(s[3])))))
        user_inf[n2idx[n]] = inf_data

    useq = np.zeros((user_size + 1, max_len))
    uinfs = np.zeros((user_size + 1, max_len))

    for n, s in ucascades.items():
        if len(s[0])<max_len:
            se_data = np.hstack((s[0], np.array([Constants.PAD] * (max_len - len(s[0])))))
            useq[u2idx[n]] = se_data

            tinf_data = np.hstack((s[1], np.array([Constants.PAD] * (max_len - len(s[1])))))
            uinfs[u2idx[n]] = tinf_data
        else:
            useq[u2idx[n]] = s[0][:max_len]
            #utimestamps[u2idx[n]] = s[1][:max_len]
            uinfs[u2idx[n]] = s[1][:max_len]

    total_len = sum(len(t_cascades[i][0]) for i in t_cascades)
    total_ulen = sum(len(u_cascades[i][0]) for i in u_cascades)
    print("total size:%d " % (len(seq) - 1))
    print('spread size',(total_len))
    print("average news cascades length:%f" % (total_len / (len(seq) - 1)))
    print("average user participant length:%f" % (total_ulen / (len(useq) - 1)))
    print("user size:%d" % (user_size))
    news_cascades = [seq, timestamps, user_level, user_inf]
    user_parti = [useq, uinfs]

    return news_cascades, user_parti, labels, user_size, news_list

if __name__ == "__main__":
    data_name = 'poli'
    options = Options(data_name)
    news_cascades, user_parti, labels, user_size, news_list= Pre_data(data_name, early_type = Constants.early_type, early = None)
    train_news = np.array([i+1 for i in options.train_idx])
    valid_news = np.array([i+1 for i in options.valid_idx])
    test_news = np.array([i+1 for i in options.test_idx])

    train_data = [train_news, labels[train_news]]
    valid_data = [valid_news,labels[valid_news]]
    test_data = [test_news,labels[test_news]]

    with open(options.news_centered, 'wb') as handle:
        pickle.dump(news_cascades, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.user_centered, 'wb') as handle:
        pickle.dump(user_parti, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(options.train, 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.valid, 'wb') as handle:
        pickle.dump(valid_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.test, 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

