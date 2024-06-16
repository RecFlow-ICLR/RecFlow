import os
import gc
import time

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import load_pkl

class Recall_Train_SASRec_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, neg_num, 
    path_to_seq, 
    video_corpus
  ):
    t1 = time.time()
    
    raw_df = pd.read_feather(path_to_csv)
    
    df = raw_df[raw_df['effective_view']==1][["request_id", "video_id"]]
    
    self.data = df.to_numpy().copy()

    self.seq_len = seq_len
    
    self.neg_num = neg_num
    
    self.today_seq = load_pkl(path_to_seq)
    
    video_corpus_df = pd.read_feather(video_corpus)
    self.video_corpus = video_corpus_df['video_id'].unique().copy() + 1
    del video_corpus_df

    self.n_video_corpus = self.video_corpus.shape[0]
    
    del raw_df
    del df
    
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def negative_sampling(self, tgt_video, neg_num):
    cnt = 0
    negs_index = np.random.randint(self.n_video_corpus, size=neg_num)
    while tgt_video in self.video_corpus[negs_index]:
      negs_index = np.random.randint(self.n_video_corpus, size=neg_num)
      cnt += 1
      if cnt >= 10:
        break
    return self.video_corpus[negs_index]
    
  def __getitem__(self, idx):
    request_id = self.data[idx][0]
    vid = self.data[idx][1] + 1 
    
    seq_full = self.today_seq[request_id][:,[0,7]].copy()
    
    seq_mask = (seq_full[:,1] > 0).astype(np.int8)
    
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,0]
    
    if seq_len > 0:
      seq_arr[-seq_len:] += 1 
    
    neg_vids = self.negative_sampling(vid, self.neg_num)
    
    return seq_arr, seq_mask, vid, neg_vids

#public
class Recall_Train_SASRec_HardNegMining_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, neg_num, 
    path_to_seq, 
    path_to_request_id_pkl,
    video_corpus,
    flow_negs,
    flow_neg_nums
  ):
    t1 = time.time()
    
    self.flow_negs = flow_negs.split(',')
  
    self.flow_neg_nums = list(map(int, flow_neg_nums.split(",")))
    
    raw_df = pd.read_feather(path_to_csv)
    
    df = raw_df[raw_df['effective_view']==1][["request_id", "video_id"]]
    
    self.data = df.to_numpy().copy()

    self.seq_len = seq_len
    
    self.neg_num = neg_num
    
    self.random_neg_nums = self.neg_num - sum(self.flow_neg_nums)
    
    self.today_seq = load_pkl(path_to_seq)
    
    video_corpus_df = pd.read_feather(video_corpus)
    self.video_corpus = video_corpus_df['video_id'].unique().copy() + 1
    del video_corpus_df

    self.n_video_corpus = self.video_corpus.shape[0]

    self.request_dict = load_pkl(path_to_request_id_pkl)
    
    del df
    del raw_df
    
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def random_negative_sampling(self, tgt_video, neg_num):
    cnt = 0
    negs_index = np.random.randint(self.n_video_corpus, size=neg_num)
    while tgt_video in self.video_corpus[negs_index]:
      negs_index = np.random.randint(self.n_video_corpus, size=neg_num)
      cnt += 1
      if cnt >= 10:
        break
    return self.video_corpus[negs_index]

  def flow_negative_sampling(self, tgt_video, request_id):
    
    flow_neg_lst = []
    
    for idx, flow_neg in enumerate(self.flow_negs):
      if flow_neg in self.request_dict[request_id]:
        flow_arr = self.request_dict[request_id][flow_neg][:,0] + 1
        flow_arr_shape = flow_arr.shape[0]
        cnt = 0
        tmp_neg_arr = flow_arr[np.random.randint(flow_arr_shape,size=self.flow_neg_nums[idx])]
        while tgt_video in tmp_neg_arr:
          tmp_neg_arr = flow_arr[np.random.randint(flow_arr_shape,size=self.flow_neg_nums[idx])]
          cnt += 1
          if cnt >= 10:
            break
      else:
        tmp_neg_arr = np.zeros(self.flow_neg_nums[idx], dtype=np.int64)
        
      flow_neg_lst.extend(tmp_neg_arr)
        
    return np.reshape(np.concatenate([np.reshape(x,[-1,1]) for x in flow_neg_lst]), [-1])
  
  def __getitem__(self, idx):
    request_id = self.data[idx][0]
    vid = self.data[idx][1] + 1 
    
    # 0: padding, 1: behavior
    seq_full = self.today_seq[request_id][:,[0,7]].copy()
    
    seq_mask = (seq_full[:,1] > 0).astype(np.int8) #50
    
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,0]
    
    if seq_len > 0:
      seq_arr[-seq_len:] += 1 #50
    
    #negative sampling
    random_neg_vids = self.random_negative_sampling(vid, self.random_neg_nums)
    
    flow_neg_vids = self.flow_negative_sampling(vid, request_id)
    
    neg_vids = np.append(random_neg_vids, flow_neg_vids)
    
    return seq_arr, seq_mask, vid, neg_vids

#public
class Recall_Train_SASRec_FSLTR_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, neg_num, 
    path_to_seq, 
    path_to_request_id_pkl,
    video_corpus,
    flow_negs,
    flow_neg_nums
  ):
    t1 = time.time()
    
    self.priority = {
      "click":6,
      "realshow":5,
      "rerank_pos":4,
      "rank_pos":4,
      "rerank_neg":3,
      "rank_neg":3,
      "coarse_neg":2,
      "prerank_neg":1
    }
    
    self.flow_negs = flow_negs.split(',')
    
    self.flow_neg_nums = list(map(int, flow_neg_nums.split(",")))
    
    raw_df = pd.read_feather(path_to_csv)
    
    df = raw_df[raw_df['effective_view']==1][["request_id", "video_id"]]
    
    self.data = df.to_numpy().copy()

    self.seq_len = seq_len
    
    self.neg_num = neg_num
    
    self.random_neg_nums = self.neg_num - sum(self.flow_neg_nums)
    
    self.today_seq = load_pkl(path_to_seq)

    video_corpus_df = pd.read_feather(video_corpus)
    self.video_corpus = video_corpus_df['video_id'].unique().copy() + 1 
    del video_corpus_df

    self.n_video_corpus = self.video_corpus.shape[0]
  
    self.request_dict = load_pkl(path_to_request_id_pkl)
    
    del df
    del raw_df
    
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def random_negative_sampling(self, tgt_video, neg_num):
    cnt = 0
    negs_index = np.random.randint(self.n_video_corpus, size=neg_num)
    while tgt_video in self.video_corpus[negs_index]:
      negs_index = np.random.randint(self.n_video_corpus, size=neg_num)
      cnt += 1
      if cnt >= 10:
        break
    return self.video_corpus[negs_index]

  def flow_negative_sampling(self, tgt_video, request_id):
    
    flow_neg_lst = []
    flow_neg_priority_lst = []
    
    flow_dict = self.request_dict[request_id]
    for idx, flow_neg in enumerate(self.flow_negs):
      if flow_neg in flow_dict:
        flow_arr = flow_dict[flow_neg][:,0] + 1
        flow_arr_shape = flow_arr.shape[0]
        cnt = 0
        tmp_neg_arr = flow_arr[np.random.randint(flow_arr_shape,size=self.flow_neg_nums[idx])]
        while tgt_video in tmp_neg_arr:
          tmp_neg_arr = flow_arr[np.random.randint(flow_arr_shape,size=self.flow_neg_nums[idx])]
          cnt += 1
          if cnt >= 10:
            break
        tmp_priority_arr = np.ones(self.flow_neg_nums[idx], dtype=np.float32)*self.priority[flow_neg]
      else:
        tmp_neg_arr = np.zeros(self.flow_neg_nums[idx], dtype=np.int64)
        tmp_priority_arr = np.zeros(self.flow_neg_nums[idx], dtype=np.float32)
      
      flow_neg_lst.append(tmp_neg_arr)
      flow_neg_priority_lst.append(tmp_priority_arr)
        
    return np.concatenate(flow_neg_lst), np.concatenate(flow_neg_priority_lst)
  
  def __getitem__(self, idx):
    request_id = self.data[idx][0]
    vid = self.data[idx][1] + 1 
    pos_priority = np.ones(1, dtype=np.float32) * self.priority['click']
    
    # 0: padding, 1: behavior
    seq_full = self.today_seq[request_id][:,[0,7]].copy()
    
    seq_mask = (seq_full[:,1] > 0).astype(np.int8) #50
    
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,0]
    
    if seq_len > 0:
      seq_arr[-seq_len:] += 1 #50
    
    #negative sampling
    random_neg_vids = self.random_negative_sampling(vid, self.random_neg_nums)
    random_neg_priority = np.zeros(self.random_neg_nums, dtype=np.float32)
    
    flow_neg_vids,flow_neg_priority = self.flow_negative_sampling(vid, request_id)
    
    vids = np.concatenate([np.atleast_1d(vid),random_neg_vids, flow_neg_vids])
    prioritys = np.concatenate([pos_priority,random_neg_priority,flow_neg_priority])
    
    return seq_arr, seq_mask, vids, prioritys

#public
class Recall_Test_SASRec_Recall_Dataset(Dataset):
  def __init__(
    self, 
    path_to_test_feather, 
    seq_len, 
    path_to_seq, 
    max_candidate_cnt=30
  ):
    t1 = time.time()
    
    raw_df = pd.read_feather(path_to_test_feather)
    
    data = raw_df[["request_id", "video_id", "effective_view"]]
    
    self.request_ids = data['request_id'].unique()
    
    self.seq_len = seq_len
    
    self.today_seq = load_pkl(path_to_seq)
    
    self.max_candidate_cnt = max_candidate_cnt
    
    self.data_group = data.copy().groupby('request_id')
    
    del data
    del raw_df
    
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')

  def __len__(self):
    return len(self.request_ids)
  
  def __getitem__(self, idx):
    request_id = self.request_ids[idx]
    
    request_id_df = self.data_group.get_group(request_id)[["video_id","effective_view"]]
    
    request_id_arr = request_id_df.to_numpy().copy()
    
    n_video = request_id_arr.shape[0]
    
    n_complent = self.max_candidate_cnt - n_video
    
    complent_arr = np.zeros(shape=(n_complent,2), dtype=np.int64)
    
    request_id_arr = np.concatenate([request_id_arr, complent_arr], axis=0)
    
    vid = request_id_arr[:,0] + 1
    
    seq_full = self.today_seq[request_id][:,[0,7]].copy()
    
    seq_mask = (seq_full[:,1] > 0).astype(np.int8)
    
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,0]
    
    if seq_len > 0:
      seq_arr[-seq_len:] += 1

    effective = request_id_arr[:,1]

    return seq_arr, seq_mask, vid, effective, n_video