import os
import gc
import time

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import load_pkl

class Rank_Train_Dataset(Dataset):
  def __init__(
    self,
    path_to_csv, 
    seq_len, 
    path_to_seq
  ):
    
    t1 = time.time()
    
    raw_df = pd.read_feather(path_to_csv)
    
    df = raw_df[["request_id", "user_id", "request_timestamp", 
                "device_id", "age", "gender",  "province", 
                "video_id", "author_id", "category_level_two", "upload_type", "upload_timestamp", "category_level_one", 
                "effective_view"]]
    
    self.data = df.to_numpy().copy()
    
    self.seq_len = seq_len
    
    self.today_seq = load_pkl(path_to_seq)
    
    del df
    del raw_df
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province", 
    # 7:"video_id", 8:"author_id", 9:"category_level_two", 10:"upload_type", 11:"upload_timestamp", 12:"category_level_one", 
    # 13:"effective_view"
    
    request_id = self.data[idx][0]

    request_ts = self.data[idx][2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = self.data[idx][1] + 1
    did = self.data[idx][3] + 1
    age = self.data[idx][4] + 1
    gender = self.data[idx][5] + 1
    province = self.data[idx][6] + 1
    
    vid = self.data[idx][7] + 1
    aid = self.data[idx][8] + 1 
    cate_two = self.data[idx][9] + 1
    upload_type = self.data[idx][10] + 1
    upload_ts = self.data[idx][11]
    cate_one = self.data[idx][12]
    
    upload_ts_struct = time.localtime(upload_ts)
    upload_ts_wday = upload_ts_struct.tm_wday + 1
    upload_ts_hour = upload_ts_struct.tm_hour + 1
    upload_ts_min = upload_ts_struct.tm_min + 1

    effective = self.data[idx][13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_timestamp", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1

    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len, 1), \
          effective

class Rank_Train_Data_Dist_Shift_Sampling_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, 
    path_to_seq,
    flows, 
    k_flow_negs
  ):
    t1 = time.time()
    
    self.flows = flows.split(',') if len(flows)>0 else []
    self.k_flow_negs = list(map(int, k_flow_negs.split(',')))
    
    self.date = os.path.splitext(os.path.basename(path_to_csv))[0]
    
    raw_df = pd.read_feather(path_to_csv)
    
    final_df_lst = []
    
    realshow_df = raw_df[raw_df['realshow']==1].copy()
    
    final_df_lst.append(realshow_df)
    
    n_request_ids = raw_df['request_id'].nunique()
      
    if len(self.flows) > 0:
      for idx, flow in enumerate(self.flows):
        tmp_flow_neg_df = raw_df[raw_df[flow]==1].sample(n=n_request_ids*self.k_flow_negs[idx]).copy()
        tmp_flow_neg_df.loc[tmp_flow_neg_df['effective_view']==0, 'effective_view'] = 0
        final_df_lst.append(tmp_flow_neg_df)
      
    final_df = pd.concat(final_df_lst, axis=0)[["request_id", "user_id", "request_timestamp",
                                                "device_id", "age", "gender",  "province", 
                                                "video_id", "author_id", "category_level_two", "upload_type", "upload_timestamp", "category_level_one", 
                                                "effective_view"]].sample(frac=1)
    
    self.data = final_df.to_numpy().copy()

    self.seq_len = seq_len

    self.today_seq = load_pkl(path_to_seq)
    
    del final_df
    del raw_df
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province", 
    # 7:"video_id", 8:"author_id", 9:"category_level_two", 10:"upload_type", 11:"upload_timestamp", 12:"category_level_one", 
    # 13:"effective_view"
    
    request_id = self.data[idx][0]

    request_ts = self.data[idx][2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = self.data[idx][1] + 1
    did = self.data[idx][3] + 1
    age = self.data[idx][4] + 1
    gender = self.data[idx][5] + 1
    province = self.data[idx][6] + 1
    
    vid = self.data[idx][7] + 1
    aid = self.data[idx][8] + 1 
    cate_two = self.data[idx][9] + 1
    upload_type = self.data[idx][10] + 1
    upload_ts = self.data[idx][11]
    cate_one = self.data[idx][12]
    
    upload_ts_struct = time.localtime(upload_ts)
    upload_ts_wday = upload_ts_struct.tm_wday + 1
    upload_ts_hour = upload_ts_struct.tm_hour + 1
    upload_ts_min = upload_ts_struct.tm_min + 1

    effective = self.data[idx][13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_ts", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1
    
    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len, 1), \
          effective

class Rank_Train_Data_Dist_Shift_All_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, 
    path_to_seq, 
    flows
  ):
    t1 = time.time()
    
    self.flows = flows.split(',') if len(flows)>0 else []
    
    self.date = os.path.splitext(os.path.basename(path_to_csv))[0]
    
    raw_df = pd.read_feather(path_to_csv)
    
    df_query_condition = ''
    
    for flow in self.flows:
      df_query_condition = df_query_condition + flow+'==1 | '
    
    df_query_condition = df_query_condition + 'realshow==1'
    
    df = raw_df.query(df_query_condition).copy()
    
    df.insert(df.shape[1], 'label', 0)
    
    if len(self.flows) > 0:
      cols = []
      for flow in self.flows:
        cols.append(flow)
      index_mask = df[cols].sum(axis=1)>0
      df.loc[index_mask,'label']=0
    
    df.loc[df['realshow']==1,'label']=df.loc[df['realshow']==1,'effective_view']
    
    self.data = df[["request_id", "user_id", "request_timestamp", 
                  "device_id", "age", "gender",  "province", 
                  "video_id", "author_id", "category_level_two", "upload_type", "upload_timestamp", "category_level_one", 
                  "label"]].to_numpy().copy()

    self.seq_len = seq_len

    self.today_seq = load_pkl(path_to_seq)
    
    del raw_df
    del df
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province", 
    # 7:"video_id", 8:"author_id", 9:"category_level_two", 10:"upload_type", 11:"upload_timestamp", 12:"category_level_one",
    # 13:"label"
    
    request_id = self.data[idx][0]

    request_ts = self.data[idx][2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = self.data[idx][1] + 1
    did = self.data[idx][3] + 1
    age = self.data[idx][4] + 1
    gender = self.data[idx][5] + 1
    province = self.data[idx][6] + 1
    
    vid = self.data[idx][7] + 1
    aid = self.data[idx][8] + 1 
    cate_two = self.data[idx][9] + 1
    upload_type = self.data[idx][10] + 1
    upload_ts = self.data[idx][11]
    cate_one = self.data[idx][12]
    
    upload_ts_struct = time.localtime(upload_ts)
    upload_ts_wday = upload_ts_struct.tm_wday + 1
    upload_ts_hour = upload_ts_struct.tm_hour + 1
    upload_ts_min = upload_ts_struct.tm_min + 1

    label = self.data[idx][13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_ts", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1
    
    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len,1), \
          label

class Rank_Train_Auxiliary_Ranking_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, 
    path_to_seq, 
    path_to_request, 
    flows
  ):
    
    t1 = time.time()
    
    self.flows = flows.split(',')
    
    self.n_flows = len(self.flows)
    
    self.k_per_flow = 10
    
    self.date = os.path.splitext(os.path.basename(path_to_csv))[0]
    
    raw_df = pd.read_feather(path_to_csv)
    
    df = raw_df[["request_id", "user_id", "request_timestamp", 
                  "device_id", "age", "gender",  "province", 
                  "video_id", "author_id", "category_level_two", "upload_type", "upload_timestamp", "category_level_one", 
                  "effective_view"]]
    
    self.data = df.to_numpy().copy()
    
    self.seq_len = seq_len
    
    self.today_seq = load_pkl(path_to_seq)

    self.request_dict = load_pkl(path_to_request)
    
    del df
    del raw_df
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province", 
    # 7:"video_id", 8:"author_id", 9:"category_level_two", 10:"upload_type", 11:"upload_timestamp", 12:"category_level_one", 
    # 13:"effective_view"
    
    request_id = self.data[idx][0]

    request_ts = self.data[idx][2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = self.data[idx][1] + 1
    did = self.data[idx][3] + 1
    age = self.data[idx][4] + 1
    gender = self.data[idx][5] + 1
    province = self.data[idx][6] + 1
    
    vid = self.data[idx][7] + 1
    aid = self.data[idx][8] + 1 
    cate_two = self.data[idx][9] + 1
    upload_type = self.data[idx][10] + 1
    upload_ts = self.data[idx][11]
    cate_one = self.data[idx][12]
    
    upload_ts_struct = time.localtime(upload_ts)
    upload_ts_wday = upload_ts_struct.tm_wday + 1
    upload_ts_hour = upload_ts_struct.tm_hour + 1
    upload_ts_min = upload_ts_struct.tm_min + 1

    effective = self.data[idx][13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_timestamp", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1
    
    flow_arr = np.zeros(shape=[self.k_per_flow * self.n_flows, 8], dtype=np.int64)
    flow_arr[:,3] = 2
    
    flow_mask = np.zeros(shape=(self.k_per_flow * self.n_flows), dtype=np.int8)
    
    #0:'video_id',1:'author_id',2:'duration',
    #3:'cate_two',4:'upload_type',5:'upload_timestamp',
    #6:'cate_one',7:'effective_view'
    
    for idx, flow in enumerate(self.flows):
      flow_photos = self.request_dict[request_id][flow].copy()
      n_flow_photos = min(flow_photos.shape[0], self.k_per_flow)
      
      if n_flow_photos > 0:
        start_index = idx * self.k_per_flow
        end_index = idx * self.k_per_flow + n_flow_photos
        
        flow_arr[start_index:end_index,0] = flow_photos[:n_flow_photos,0] + 1 #video_id
        flow_arr[start_index:end_index,1] = flow_photos[:n_flow_photos,1] + 1 #author_id
        flow_arr[start_index:end_index,2] = flow_photos[:n_flow_photos,3] + 1 #cate_two
        flow_arr[start_index:end_index,3] = flow_photos[:n_flow_photos,6]     #cate_one_level_one
        flow_arr[start_index:end_index,4] = flow_photos[:n_flow_photos,4] + 1 #upload_type
        
        flow_photos_upt_lst = flow_photos[:n_flow_photos,5].tolist()
        
        flow_arr[start_index:end_index,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, flow_photos_upt_lst)), dtype=np.int64) #upload_time_wday
        flow_arr[start_index:end_index,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, flow_photos_upt_lst)), dtype=np.int64) #upload_time_hour
        flow_arr[start_index:end_index,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  flow_photos_upt_lst)), dtype=np.int64) #upload_time_min
        
        flow_mask[start_index:end_index] = flow_photos[:n_flow_photos,7] #effective_view
    
    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len,1), \
          flow_arr, flow_mask, \
          effective

class Rank_Train_FSLTR_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, 
    path_to_seq, 
    path_to_request
  ):
    
    t1 = time.time()
    
    self.date = os.path.splitext(os.path.basename(path_to_csv))[0]
    
    raw_df = pd.read_feather(path_to_csv)
    df = raw_df[["request_id", "user_id", "request_timestamp","device_id", "age", "gender",  "province"]].drop_duplicates()
    
    self.data = df.to_numpy().copy()
    
    self.seq_len = seq_len
    
    self.today_seq = load_pkl(path_to_seq)

    self.request_dict = load_pkl(path_to_request)
    
    del df
    del raw_df
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province" 
    
    request_id = self.data[idx][0]

    request_ts = self.data[idx][2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = self.data[idx][1] + 1
    did = self.data[idx][3] + 1
    age = self.data[idx][4] + 1
    gender = self.data[idx][5] + 1
    province = self.data[idx][6] + 1
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_timestamp", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1
    
    click_photos = np.zeros(shape=[6, 8], dtype=np.int64)
    click_photos[:,3] = 2
    click_mask = np.zeros(shape=(6), dtype=np.int8)
    
    realshow_photos = np.zeros(shape=[6, 8], dtype=np.int64)
    realshow_photos[:,3] = 2
    realshow_mask = np.zeros(shape=(6), dtype=np.int8)
    
    photo_dict = self.request_dict[request_id]
    
    if 'realshow' in photo_dict:
      realshow_flow_photos = photo_dict['realshow'].copy()
      n_realshow_flow_photos = min(realshow_flow_photos.shape[0], 6)
      
      if n_realshow_flow_photos > 0:
        sampling_realshow_flow_photos = realshow_flow_photos[:n_realshow_flow_photos]
        
        realshow_photos[:n_realshow_flow_photos,0] = sampling_realshow_flow_photos[:,0] + 1 #video_id
        realshow_photos[:n_realshow_flow_photos,1] = sampling_realshow_flow_photos[:,1] + 1 #author_id
        realshow_photos[:n_realshow_flow_photos,2] = sampling_realshow_flow_photos[:,3] + 1 #cate_two
        realshow_photos[:n_realshow_flow_photos,3] = sampling_realshow_flow_photos[:,6]     #cate_one
        realshow_photos[:n_realshow_flow_photos,4] = sampling_realshow_flow_photos[:,4] + 1 #upload_type
        
        sampling_realshow_flow_photos_upt_lst = sampling_realshow_flow_photos[:,5].tolist()
        
        realshow_photos[:n_realshow_flow_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, sampling_realshow_flow_photos_upt_lst)), dtype=np.int64) #upload_time_wday
        realshow_photos[:n_realshow_flow_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, sampling_realshow_flow_photos_upt_lst)), dtype=np.int64) #upload_time_hour
        realshow_photos[:n_realshow_flow_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  sampling_realshow_flow_photos_upt_lst)), dtype=np.int64) #upload_time_min
        
        realshow_mask[:n_realshow_flow_photos] = 1

        if np.sum(realshow_flow_photos[:,-1]) > 0:
          click_index = np.nonzero(realshow_flow_photos[:,-1])[0]
          click_flow_photos = realshow_flow_photos[click_index]
          
          n_click_flow_photos = min(click_flow_photos.shape[0], 6)
          
          sampling_click_flow_photos = click_flow_photos[:n_click_flow_photos]
          
          click_photos[:n_click_flow_photos,0] = sampling_click_flow_photos[:,0] + 1 #video_id
          click_photos[:n_click_flow_photos,1] = sampling_click_flow_photos[:,1] + 1 #author_id
          click_photos[:n_click_flow_photos,2] = sampling_click_flow_photos[:,3] + 1 #cate_two
          click_photos[:n_click_flow_photos,3] = sampling_click_flow_photos[:,6]     #cate_one
          click_photos[:n_click_flow_photos,4] = sampling_click_flow_photos[:,4] + 1 #upload_type
          
          sampling_click_flow_photos_upt_lst = sampling_click_flow_photos[:,5].tolist()
          
          click_photos[:n_click_flow_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, sampling_click_flow_photos_upt_lst)), dtype=np.int64) #upload_time_wday
          click_photos[:n_click_flow_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, sampling_click_flow_photos_upt_lst)), dtype=np.int64) #upload_time_hour
          click_photos[:n_click_flow_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  sampling_click_flow_photos_upt_lst)), dtype=np.int64) #upload_time_min
          
          click_mask[:n_click_flow_photos] = 1
    
    rerank_pos_photos = np.zeros(shape=[10, 8], dtype=np.int64)
    rerank_pos_photos[:,3] = 2
    rerank_pos_mask = np.zeros(shape=(10), dtype=np.int8)
    
    if 'rerank_pos' in photo_dict:
      rerank_pos_flow_photos = photo_dict['rerank_pos'].copy()
      n_rerank_pos_flow_photos = min(rerank_pos_flow_photos.shape[0], 10)
      
      if n_rerank_pos_flow_photos > 0:
        sampling_rerank_pos_flow_photos = rerank_pos_flow_photos[:n_rerank_pos_flow_photos]
        
        rerank_pos_photos[:n_rerank_pos_flow_photos,0] = sampling_rerank_pos_flow_photos[:,0] + 1 #video_id
        rerank_pos_photos[:n_rerank_pos_flow_photos,1] = sampling_rerank_pos_flow_photos[:,1] + 1 #author_id
        rerank_pos_photos[:n_rerank_pos_flow_photos,2] = sampling_rerank_pos_flow_photos[:,3] + 1 #cate_two
        rerank_pos_photos[:n_rerank_pos_flow_photos,3] = sampling_rerank_pos_flow_photos[:,6]     #cate_one
        rerank_pos_photos[:n_rerank_pos_flow_photos,4] = sampling_rerank_pos_flow_photos[:,4] + 1 #upload_type
        
        sampling_rerank_pos_flow_photos_upt_lst = sampling_rerank_pos_flow_photos[:,5].tolist()
        
        rerank_pos_photos[:n_rerank_pos_flow_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, sampling_rerank_pos_flow_photos_upt_lst)), dtype=np.int64) #upload_time_wday
        rerank_pos_photos[:n_rerank_pos_flow_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, sampling_rerank_pos_flow_photos_upt_lst)), dtype=np.int64) #upload_time_hour
        rerank_pos_photos[:n_rerank_pos_flow_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  sampling_rerank_pos_flow_photos_upt_lst)), dtype=np.int64) #upload_time_min
        
        rerank_pos_mask[:n_rerank_pos_flow_photos] = 1
    
    rerank_neg_photos = np.zeros(shape=[10, 8], dtype=np.int64)
    rerank_neg_photos[:,3] = 2
    rerank_neg_mask = np.zeros(shape=(10), dtype=np.int8)
    
    if 'rerank_neg' in photo_dict:
      rerank_neg_flow_photos = photo_dict['rerank_neg'].copy()
      n_rerank_neg_flow_photos = min(rerank_neg_flow_photos.shape[0], 10)
      
      if n_rerank_neg_flow_photos > 0:
        sampling_rerank_neg_flow_photos = rerank_neg_flow_photos[:n_rerank_neg_flow_photos]
        
        rerank_neg_photos[:n_rerank_neg_flow_photos,0] = sampling_rerank_neg_flow_photos[:,0] + 1 #video_id
        rerank_neg_photos[:n_rerank_neg_flow_photos,1] = sampling_rerank_neg_flow_photos[:,1] + 1 #author_id
        rerank_neg_photos[:n_rerank_neg_flow_photos,2] = sampling_rerank_neg_flow_photos[:,3] + 1 #cate_two
        rerank_neg_photos[:n_rerank_neg_flow_photos,3] = sampling_rerank_neg_flow_photos[:,6]     #cate_one
        rerank_neg_photos[:n_rerank_neg_flow_photos,4] = sampling_rerank_neg_flow_photos[:,4] + 1 #upload_type
        
        sampling_rerank_neg_flow_photos_upt_lst = sampling_rerank_neg_flow_photos[:,5].tolist()
        
        rerank_neg_photos[:n_rerank_neg_flow_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, sampling_rerank_neg_flow_photos_upt_lst)), dtype=np.int64) #upload_time_wday
        rerank_neg_photos[:n_rerank_neg_flow_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, sampling_rerank_neg_flow_photos_upt_lst)), dtype=np.int64) #upload_time_hour
        rerank_neg_photos[:n_rerank_neg_flow_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  sampling_rerank_neg_flow_photos_upt_lst)), dtype=np.int64) #upload_time_min
        
        rerank_neg_mask[:n_rerank_neg_flow_photos] = 1
    
    rank_pos_photos = np.zeros(shape=[10, 8], dtype=np.int64)
    rank_pos_photos[:,3] = 2
    rank_pos_mask = np.zeros(shape=(10), dtype=np.int8)
    
    if 'rank_pos' in photo_dict:
      rank_pos_flow_photos = photo_dict['rank_pos'].copy()
      n_rank_pos_flow_photos = min(rank_pos_flow_photos.shape[0], 10)
      
      if n_rank_pos_flow_photos > 0:
        sampling_rank_pos_flow_photos = rank_pos_flow_photos[:n_rank_pos_flow_photos]
        
        rank_pos_photos[:n_rank_pos_flow_photos,0] = sampling_rank_pos_flow_photos[:,0] + 1 #video_id
        rank_pos_photos[:n_rank_pos_flow_photos,1] = sampling_rank_pos_flow_photos[:,1] + 1 #author_id
        rank_pos_photos[:n_rank_pos_flow_photos,2] = sampling_rank_pos_flow_photos[:,3] + 1 #cate_two
        rank_pos_photos[:n_rank_pos_flow_photos,3] = sampling_rank_pos_flow_photos[:,6]     #cate_one
        rank_pos_photos[:n_rank_pos_flow_photos,4] = sampling_rank_pos_flow_photos[:,4] + 1 #upload_type
        
        sampling_rank_pos_flow_photos_upt_lst = sampling_rank_pos_flow_photos[:,5].tolist()
        
        rank_pos_photos[:n_rank_pos_flow_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, sampling_rank_pos_flow_photos_upt_lst)), dtype=np.int64) #upload_time_wday
        rank_pos_photos[:n_rank_pos_flow_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, sampling_rank_pos_flow_photos_upt_lst)), dtype=np.int64) #upload_time_hour
        rank_pos_photos[:n_rank_pos_flow_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  sampling_rank_pos_flow_photos_upt_lst)), dtype=np.int64) #upload_time_min
        
        rank_pos_mask[:n_rank_pos_flow_photos] = 1
    
    rank_neg_photos = np.zeros(shape=[10, 8], dtype=np.int64)
    rank_neg_photos[:,3] = 2
    rank_neg_mask = np.zeros(shape=(10), dtype=np.int8)
    
    if 'rank_neg' in photo_dict:
      rank_neg_flow_photos = photo_dict['rank_neg'].copy()
      n_rank_neg_flow_photos = min(rank_neg_flow_photos.shape[0], 10)
      
      if n_rank_neg_flow_photos > 0:
        sampling_rank_neg_flow_photos = rank_neg_flow_photos[:n_rank_neg_flow_photos]
        
        rank_neg_photos[:n_rank_neg_flow_photos,0] = sampling_rank_neg_flow_photos[:,0] + 1 #video_id
        rank_neg_photos[:n_rank_neg_flow_photos,1] = sampling_rank_neg_flow_photos[:,1] + 1 #author_id
        rank_neg_photos[:n_rank_neg_flow_photos,2] = sampling_rank_neg_flow_photos[:,3] + 1 #cate_two
        rank_neg_photos[:n_rank_neg_flow_photos,3] = sampling_rank_neg_flow_photos[:,6]     #cate_one
        rank_neg_photos[:n_rank_neg_flow_photos,4] = sampling_rank_neg_flow_photos[:,4] + 1 #upload_type
        
        sampling_rank_neg_flow_photos_upt_lst = sampling_rank_neg_flow_photos[:,5].tolist()
        
        rank_neg_photos[:n_rank_neg_flow_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, sampling_rank_neg_flow_photos_upt_lst)), dtype=np.int64) #upload_time_wday
        rank_neg_photos[:n_rank_neg_flow_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, sampling_rank_neg_flow_photos_upt_lst)), dtype=np.int64) #upload_time_hour
        rank_neg_photos[:n_rank_neg_flow_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  sampling_rank_neg_flow_photos_upt_lst)), dtype=np.int64) #upload_time_min
        
        rank_neg_mask[:n_rank_neg_flow_photos] = 1
    
    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          seq_arr, seq_mask, max(seq_len,1), \
          click_photos, realshow_photos, rerank_pos_photos, \
          rerank_neg_photos, rank_pos_photos, rank_neg_photos, \
          click_mask, realshow_mask, rerank_pos_mask, \
          rerank_neg_mask, rank_pos_mask, rank_neg_mask

class Rank_Train_UBM_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, 
    path_to_seq, 
    path_to_request,
    flows, 
    per_flow_seq_len
  ):
    t1 = time.time()
    
    self.per_flow_seq_len = per_flow_seq_len
    
    self.flows = flows.split(',')
    
    self.n_flows = len(self.flows)
    
    self.flow_seq_len = self.n_flows * self.per_flow_seq_len
    
    raw_df = pd.read_feather(path_to_csv)
    df = raw_df[["request_id", "user_id", "request_timestamp", 
                  "device_id", "age", "gender",  "province", 
                  "video_id", "author_id", "category_level_two", "upload_type", "upload_timestamp", "category_level_one", 
                  "effective_view"]]
    
    self.data = df.to_numpy().copy()
    
    self.date = os.path.splitext(os.path.basename(path_to_csv))[0]
    
    self.seq_len = seq_len
    
    self.today_seq = load_pkl(path_to_seq)

    self.request_dict = load_pkl(path_to_request)
    
    del df
    del raw_df
    gc.collect()
    
    t2 = time.time()
    print(f'init data time: {t2-t1}')
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province", 
    # 7:"video_id", 8:"author_id", 9:"category_level_two", 10:"upload_type", 11:"upload_timestamp", 12:"category_level_one", 
    # 13:"effective_view"
    
    request_id = self.data[idx][0]

    request_ts = self.data[idx][2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = self.data[idx][1] + 1
    did = self.data[idx][3] + 1
    age = self.data[idx][4] + 1
    gender = self.data[idx][5] + 1
    province = self.data[idx][6] + 1
    
    vid = self.data[idx][7] + 1
    aid = self.data[idx][8] + 1 
    cate_two = self.data[idx][9] + 1
    upload_type = self.data[idx][10] + 1
    upload_ts = self.data[idx][11]
    cate_one = self.data[idx][12]
    
    upload_ts_struct = time.localtime(upload_ts)
    upload_ts_wday = upload_ts_struct.tm_wday + 1
    upload_ts_hour = upload_ts_struct.tm_hour + 1
    upload_ts_min = upload_ts_struct.tm_min + 1

    effective = self.data[idx][13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_timestamp", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4, request_timestamp-5, request_id-6
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7,9]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1

    seq_request_id = seq_full[:,6] #50
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)

    # request context
    flow_seq_arr = np.zeros(shape=[self.seq_len, self.flow_seq_len, 5], dtype=np.int64)
    flow_seq_arr[:,:,3] = 2
    
    # 0: padding, 1: behavior
    flow_seq_mask = np.zeros(shape=[self.seq_len, self.flow_seq_len], dtype=np.int8)
    
    if seq_len > 0:
      for i in range(seq_len):
        real_i = - (i+1)
        
        tmp_request_id = seq_request_id[real_i]
        
        for idx, flow in enumerate(self.flows):
          #0:'video_id',1:'author_id',2:'duration',
          #3:'cate_two',4:'upload_type',5:'upload_timestamp',
          #6:'cate_one',7:'effective_view'
          
          flow_photos = self.request_dict[tmp_request_id][flow][:,[0,1,3,6,4]].copy() #n*5, array 
          
          n_flow_photos = min(flow_photos.shape[0], self.per_flow_seq_len)
          
          if n_flow_photos > 0:
            start_index = idx * self.per_flow_seq_len
            end_index = idx * self.per_flow_seq_len + n_flow_photos
            
            flow_seq_arr[real_i, start_index:end_index, 0] = flow_photos[:n_flow_photos,0]+1
            flow_seq_arr[real_i, start_index:end_index, 1] = flow_photos[:n_flow_photos,1]+1
            flow_seq_arr[real_i, start_index:end_index, 2] = flow_photos[:n_flow_photos,2]+1
            flow_seq_arr[real_i, start_index:end_index, 3] = flow_photos[:n_flow_photos,3]
            flow_seq_arr[real_i, start_index:end_index, 4] = flow_photos[:n_flow_photos,4]+1
            
            flow_seq_mask[real_i, start_index:end_index] = 1

    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len,1), \
          flow_seq_arr, flow_seq_mask, \
          effective

class Rank_Test_Dataset(Dataset):
  def __init__(
    self, 
    path_to_test_feather, 
    seq_len, 
    path_to_seq, 
    max_candidate_cnt=430
  ):
    
    t1 = time.time()
    
    raw_df = pd.read_feather(path_to_test_feather)
    data = raw_df[["request_id", "user_id", "request_timestamp", 
                  "device_id", "age", "gender",  "province", 
                  "video_id", "author_id", "category_level_two", "upload_type", "upload_timestamp", "category_level_one", 
                  "effective_view"]]
    
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
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province", 
    # 7:"video_id", 8:"author_id", 9:"category_level_two", 10:"upload_type", 11:"upload_timestamp", 12:"category_level_one", 
    # 13:"effective_view"
    
    request_id = self.request_ids[idx]
    
    request_id_df = self.data_group.get_group(request_id)
    
    request_id_arr = request_id_df.to_numpy().copy()
    
    n_photo = request_id_arr.shape[0]
    
    n_complent = self.max_candidate_cnt - n_photo
    
    complent_arr = np.zeros(shape=[n_complent,14], dtype=np.int64)
    
    request_id_arr = np.concatenate([request_id_arr, complent_arr], axis=0)
    
    request_ts = request_id_arr[0,2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = request_id_arr[0,1] + 1
    did = request_id_arr[0,3] + 1
    age = request_id_arr[0,4] + 1
    gender = request_id_arr[0,5] + 1
    province = request_id_arr[0,6] + 1
    
    vid = request_id_arr[:,7] + 1
    aid = request_id_arr[:,8] + 1
    cate_two = request_id_arr[:,9] + 1
    upload_type = request_id_arr[:,10] + 1
    upload_ts = request_id_arr[:,11]
    
    upload_ts_struct = [time.localtime(x) for x in upload_ts]
    upload_ts_wday = np.array([x.tm_wday + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_hour = np.array([x.tm_hour + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_min = np.array([x.tm_min + 1 for x in upload_ts_struct], dtype=np.int64)
    
    cate_one = request_id_arr[:,12]
    
    effective = request_id_arr[:,13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_timestamp", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1

    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len,1), \
          effective, n_photo

class Rank_Test_UBM_Dataset(Dataset):
  def __init__(
    self, 
    path_to_test_feather, 
    seq_len, 
    path_to_seq, 
    path_to_request, 
    flows, per_flow_seq_len,
    max_candidate_cnt=430
  ):
    t1 = time.time()
    self.per_flow_seq_len = per_flow_seq_len
    
    self.flows = flows.split(',')
    
    self.n_flows = len(self.flows)
    
    self.flow_seq_len = self.n_flows * self.per_flow_seq_len
    
    raw_df = pd.read_feather(path_to_test_feather)
    data = raw_df[["request_id", "user_id", "request_timestamp", 
                  "device_id", "age", "gender",  "province", 
                  "video_id", "author_id", "category_level_two", "upload_type", "upload_timestamp", "category_level_one", 
                  "effective_view"]]
    
    self.request_ids = data['request_id'].unique()
    
    self.seq_len = seq_len
    
    self.today_seq = load_pkl(path_to_seq)
    
    self.request_dict = load_pkl(path_to_request)
    
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
    # 0:"request_id", 1:"user_id", 2:"request_timestamp", 
    # 3:"device_id", 4:"age", 5:"gender",  6:"province", 
    # 7:"video_id", 8:"author_id", 9:"category_level_two", 10:"upload_type", 11:"upload_timestamp", 12:"category_level_one", 
    # 13:"effective_view"
    
    request_id = self.request_ids[idx]
    
    request_id_df = self.data_group.get_group(request_id)
    
    request_id_arr = request_id_df.to_numpy().copy()
    
    n_photo = request_id_arr.shape[0]
    
    n_complent = self.max_candidate_cnt - n_photo
    
    complent_arr = np.zeros(shape=[n_complent,14], dtype=np.int64)
    
    request_id_arr = np.concatenate([request_id_arr, complent_arr], axis=0)
    
    request_ts = request_id_arr[0,2]
    
    request_ts_struct = time.localtime(request_ts)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = request_id_arr[0,1] + 1
    did = request_id_arr[0,3] + 1
    age = request_id_arr[0,4] + 1
    gender = request_id_arr[0,5] + 1
    province = request_id_arr[0,6] + 1
    
    vid = request_id_arr[:,7] + 1
    aid = request_id_arr[:,8] + 1
    cate_two = request_id_arr[:,9] + 1
    upload_type = request_id_arr[:,10] + 1
    upload_ts = request_id_arr[:,11]
    
    upload_ts_struct = [time.localtime(x) for x in upload_ts]
    upload_ts_wday = np.array([x.tm_wday + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_hour = np.array([x.tm_hour + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_min = np.array([x.tm_min + 1 for x in upload_ts_struct], dtype=np.int64)
    
    cate_one = request_id_arr[:,12]
    
    effective = request_id_arr[:,13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_timestamp", 
    # 6:"duration", 7:"request_timestamp", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4, request_timestamp-5, request_id-6
    seq_full = self.today_seq[request_id][:,[0,1,2,3,4,7,9]].copy() #50*6
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)
    
    seq_arr = seq_full[:,:5] #50*5
    
    if seq_len > 0:
      seq_arr[-seq_len:,:3] += 1
      seq_arr[-seq_len:,4] += 1

    seq_request_id = seq_full[:,6] #50
    
    # 0: padding, 1: behavior
    seq_mask = (seq_full[:,5] > 0).astype(np.int8) #50
    seq_len = np.sum(seq_mask)

    # request context
    flow_seq_arr = np.zeros(shape=[self.seq_len, self.flow_seq_len, 5], dtype=np.int64)
    flow_seq_arr[:,:,3] = 2
    
    # 0: padding, 1: behavior
    flow_seq_mask = np.zeros(shape=[self.seq_len, self.flow_seq_len], dtype=np.int8)
    
    if seq_len > 0:
      for i in range(seq_len):
        real_i = - (i+1)
        
        tmp_request_id = seq_request_id[real_i]
        
        for idx, flow in enumerate(self.flows):
          #0:'video_id',1:'author_id',2:'duration',
          #3:'cate_two',4:'upload_type',5:'upload_timestamp',
          #6:'cate_one',7:'effective_view'
          
          flow_photos = self.request_dict[tmp_request_id][flow][:,[0,1,3,6,4]].copy() #n*5, array 
          
          n_flow_photos = min(flow_photos.shape[0], self.per_flow_seq_len)
          
          if n_flow_photos > 0:
            start_index = idx * self.per_flow_seq_len
            end_index = idx * self.per_flow_seq_len + n_flow_photos
            
            flow_seq_arr[real_i, start_index:end_index, 0] = flow_photos[:n_flow_photos,0]+1
            flow_seq_arr[real_i, start_index:end_index, 1] = flow_photos[:n_flow_photos,1]+1
            flow_seq_arr[real_i, start_index:end_index, 2] = flow_photos[:n_flow_photos,2]+1
            flow_seq_arr[real_i, start_index:end_index, 3] = flow_photos[:n_flow_photos,3]
            flow_seq_arr[real_i, start_index:end_index, 4] = flow_photos[:n_flow_photos,4]+1
            
            flow_seq_mask[real_i, start_index:end_index] = 1

    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len,1), \
          flow_seq_arr, flow_seq_mask, \
          effective, n_photo
