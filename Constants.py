import torch
PAD = 0

step_split = 2
n_heads = 14

#cate = ['retweet', 'support', 'deny']
cate = ['retweet']
early_type = 'time' # 'engage' or 'time'

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
