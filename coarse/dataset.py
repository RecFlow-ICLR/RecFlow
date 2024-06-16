import os
import gc
import time

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import load_pkl

#public
class Prerank_Train_Dataset(Dataset):
  def __init__(self, path_to_csv, seq_len, path_to_seq):
    
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
    
    request_ts_struct = time.localtime(request_ts//1000)
    
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
    # 3:"cate_one", 4:"upload_type", 5:"upload_time", 
    # 6:"duration_ms", 7:"time_ms", 8:"playing_time", 
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

#public
class Prerank_Train_Data_Dist_Shift_Sampling_Dataset(Dataset):
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
    
    request_ts_struct = time.localtime(request_ts//1000)
    
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
    # 3:"cate_one", 4:"upload_type", 5:"upload_time", 
    # 6:"duration_ms", 7:"time_ms", 8:"playing_time", 
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
          effective

#public
class Prerank_Train_Data_Dist_Shift_All_Dataset(Dataset):
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
    
    request_ts_struct = time.localtime(request_ts//1000)
    
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
    # 3:"cate_one", 4:"upload_type", 5:"upload_time", 
    # 6:"duration_ms", 7:"time_ms", 8:"playing_time", 
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

#public
class Prerank_Train_Auxiliary_Ranking_Dataset(Dataset):
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
    
    request_ts_struct = time.localtime(request_ts//1000)
    
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
    # 3:"cate_one", 4:"upload_type", 5:"upload_time", 
    # 6:"duration_ms", 7:"time_ms", 8:"playing_time", 
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
    
    for idx, flow  in enumerate(self.flows):
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
        
        flow_arr[start_index:end_index,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, flow_photos_upt_lst)), dtype=np.int64) #request_ts_wday
        flow_arr[start_index:end_index,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, flow_photos_upt_lst)), dtype=np.int64) #request_ts_hour
        flow_arr[start_index:end_index,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  flow_photos_upt_lst)), dtype=np.int64) #request_ts_min
        
        flow_mask[start_index:end_index] = flow_photos[:n_flow_photos,7] #effective_view
    
    return request_ts_wday, request_ts_hour, request_ts_min, \
          uid, did, gender, age, province, \
          vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
          seq_arr, seq_mask, max(seq_len,1), \
          flow_arr, flow_mask, \
          effective

#public
class Prerank_Train_FSLTR_Dataset(Dataset):
  def __init__(
    self, 
    path_to_csv, 
    seq_len, 
    path_to_seq, 
    path_to_request,
    flows,
    flow_nums
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
    
    self.flows = flows.split(',')
    
    self.flow_nums = list(map(int, flow_nums.split(",")))
    
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
    
    request_ts_struct = time.localtime(request_ts//1000)
    
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
    
    flow_photo_lst = []
    flow_priority_lst = []
    
    flow_dict = self.request_dict[request_id]
    
    #0:'video_id',1:'author_id',2:'duration_ms',
    #3:'cate_two',4:'upload_type',5:'upload_time',
    #6:'cate_one_level_one',7:'effective_view'
    
    for idx, flow in enumerate(self.flows):
      if flow in flow_dict:
        flow_arr = flow_dict[flow] # n_flow_photos*8
        n_flow_photos = flow_arr.shape[0]
        n_sampling_photos = self.flow_nums[idx]
        
        if n_sampling_photos==10 or (n_sampling_photos==6 and flow=='realshow'):
          tmp_photo_arr = np.zeros(shape=[n_sampling_photos, 8], dtype=np.int64)
          tmp_photo_arr[:,3] = 2
          
          n_real_sampling_photos = min(n_sampling_photos, n_flow_photos)
          
          tmp_photo_arr[:n_real_sampling_photos,0] = flow_arr[:n_real_sampling_photos,0] + 1 #video_id
          tmp_photo_arr[:n_real_sampling_photos,1] = flow_arr[:n_real_sampling_photos,1] + 1 #author_id
          tmp_photo_arr[:n_real_sampling_photos,2] = flow_arr[:n_real_sampling_photos,3] + 1 #cate_two
          tmp_photo_arr[:n_real_sampling_photos,3] = flow_arr[:n_real_sampling_photos,6]     #cate_one_level_one
          tmp_photo_arr[:n_real_sampling_photos,4] = flow_arr[:n_real_sampling_photos,4] + 1 #upload_type
          
          tmp_photo_upt_lst = flow_arr[:n_real_sampling_photos,5].tolist()
          
          tmp_photo_arr[:n_real_sampling_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, tmp_photo_upt_lst)), dtype=np.int64) #request_ts_wday
          tmp_photo_arr[:n_real_sampling_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, tmp_photo_upt_lst)), dtype=np.int64) #request_ts_hour
          tmp_photo_arr[:n_real_sampling_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  tmp_photo_upt_lst)), dtype=np.int64) #request_ts_min
          
          tmp_priority_arr = np.zeros(n_sampling_photos, dtype=np.float32)
          tmp_priority_arr[:n_real_sampling_photos] += self.priority[flow]
          
        else:
          tmp_flow_arr = flow_arr[np.random.randint(n_flow_photos, size=n_sampling_photos)]
          
          tmp_photo_arr = np.zeros(shape=[n_sampling_photos, 8], dtype=np.int64)
          tmp_photo_arr[:,3] = 2
          
          tmp_photo_arr[:,0] = tmp_flow_arr[:,0] + 1  #video_id
          tmp_photo_arr[:,1] = tmp_flow_arr[:,1] + 1  #author_id
          tmp_photo_arr[:,2] = tmp_flow_arr[:,3] + 1  #cate_two
          tmp_photo_arr[:,3] = tmp_flow_arr[:,6]      #cate_one_level_one
          tmp_photo_arr[:,4] = tmp_flow_arr[:,4] + 1  #upload_type
          
          tmp_photo_upt_lst = tmp_flow_arr[:,5].tolist()
          
          tmp_photo_arr[:,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, tmp_photo_upt_lst)), dtype=np.int64) #request_ts_wday
          tmp_photo_arr[:,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, tmp_photo_upt_lst)), dtype=np.int64) #request_ts_hour
          tmp_photo_arr[:,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  tmp_photo_upt_lst)), dtype=np.int64) #request_ts_min
          
          tmp_priority_arr = np.zeros(n_sampling_photos, dtype=np.float32)+self.priority[flow]
            
      elif flow == "click":
        if 'realshow' in flow_dict and np.sum(flow_dict['realshow'][:,-1])>0:
          realshow_photos = flow_dict['realshow']
          
          click_index = np.nonzero(realshow_photos[:,-1])[0]
          
          click_photos = realshow_photos[click_index]
          
          n_click_photos = click_photos.shape[0]
          n_sampling_photos = self.flow_nums[idx]
          
          tmp_photo_arr = np.zeros(shape=[n_sampling_photos, 8], dtype=np.int64)
          
          n_real_sampling_photos = min(n_sampling_photos, n_click_photos)
          
          tmp_photo_arr[:n_real_sampling_photos,0] = click_photos[:n_real_sampling_photos,0] + 1 #video_id
          tmp_photo_arr[:n_real_sampling_photos,1] = click_photos[:n_real_sampling_photos,1] + 1 #author_id
          tmp_photo_arr[:n_real_sampling_photos,2] = click_photos[:n_real_sampling_photos,3] + 1 #cate_two
          tmp_photo_arr[:n_real_sampling_photos,3] = click_photos[:n_real_sampling_photos,6]     #cate_one_level_one
          tmp_photo_arr[:n_real_sampling_photos,4] = click_photos[:n_real_sampling_photos,4] + 1 #upload_type
          
          tmp_photo_upt_lst = click_photos[:n_real_sampling_photos,5].tolist()
          
          tmp_photo_arr[:n_real_sampling_photos,5] = np.array(list(map(lambda x: time.localtime(x).tm_wday+1, tmp_photo_upt_lst)), dtype=np.int64) #request_ts_wday
          tmp_photo_arr[:n_real_sampling_photos,6] = np.array(list(map(lambda x: time.localtime(x).tm_hour+1, tmp_photo_upt_lst)), dtype=np.int64) #request_ts_hour
          tmp_photo_arr[:n_real_sampling_photos,7] = np.array(list(map(lambda x: time.localtime(x).tm_min+1,  tmp_photo_upt_lst)), dtype=np.int64) #request_ts_min
          
          tmp_priority_arr = np.zeros(n_sampling_photos, dtype=np.float32)
          tmp_priority_arr[:n_real_sampling_photos] += self.priority[flow]
      
        else:
          tmp_photo_arr = np.zeros(shape=[self.flow_nums[idx], 8], dtype=np.int64)
          tmp_priority_arr = np.zeros(self.flow_nums[idx], dtype=np.float32)
      
      else:
        tmp_photo_arr = np.zeros(shape=[self.flow_nums[idx], 8], dtype=np.int64)
        tmp_priority_arr = np.zeros(self.flow_nums[idx], dtype=np.float32)
      
      flow_photo_lst.append(tmp_photo_arr)
      flow_priority_lst.append(tmp_priority_arr)
    
      flow_photo = np.concatenate(flow_photo_lst, axis=0)
      flow_priority = np.concatenate(flow_priority_lst, axis=0)

    return request_ts_wday,request_ts_hour,request_ts_min, \
          uid, did, gender, age, province, \
          seq_arr, seq_mask, max(seq_len,1), \
          flow_photo, flow_priority

#public
class Prerank_Train_UBM_Dataset(Dataset):
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
    
    self.request_dict = load_pkl(path_to_request) #request_id->request_label->array
    
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
    
    request_ts_struct = time.localtime(request_ts//1000)
    
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
    # 3:"cate_one", 4:"upload_type", 5:"upload_time", 
    # 6:"duration_ms", 7:"time_ms", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4, time_ms-5, request_id-6
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
          #0:'video_id',1:'author_id',2:'duration_ms',
          #3:'cate_two',4:'upload_type',5:'upload_time',
          #6:'cate_one_level_one',7:'effective_view'
          
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

#public
class Prerank_Test_Dataset(Dataset):
  def __init__(self, path_to_test_feather, seq_len, path_to_seq, max_candidate_cnt=500):
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
    
    request_ts_df = self.data_group.get_group(request_id)
    
    request_ts_arr = request_ts_df.to_numpy().copy()
    
    n_photo = request_ts_arr.shape[0]
    
    n_complent = self.max_candidate_cnt - n_photo
    
    complent_arr = np.zeros(shape=[n_complent,14], dtype=np.int64)
    
    request_ts_arr = np.concatenate([request_ts_arr, complent_arr], axis=0)
    
    request_ts = request_ts_arr[0,2]
    
    request_ts_struct = time.localtime(request_ts//1000)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = request_ts_arr[0,1] + 1
    did = request_ts_arr[0,3] + 1
    age = request_ts_arr[0,4] + 1
    gender = request_ts_arr[0,5] + 1
    province = request_ts_arr[0,6] + 1
    
    vid = request_ts_arr[:,7] + 1
    aid = request_ts_arr[:,8] + 1
    cate_two = request_ts_arr[:,9] + 1
    upload_type = request_ts_arr[:,10] + 1
    upload_ts = request_ts_arr[:,11]
    
    upload_ts_struct = [time.localtime(x) for x in upload_ts]
    upload_ts_wday = np.array([x.tm_wday + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_hour = np.array([x.tm_hour + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_min = np.array([x.tm_min + 1 for x in upload_ts_struct], dtype=np.int64)
    
    cate_one = request_ts_arr[:,12]
    
    effective = request_ts_arr[:,13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_time", 
    # 6:"duration_ms", 7:"time_ms", 8:"playing_time", 
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

#public
class Prerank_Test_UBM_Dataset(Dataset):
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
    # 0:"request_id", 1:"user_id", 2:"time_ms", 
    # 3:"device_id", 4:"user_age_segment", 5:"user_gender",  6:"user_province", 
    # 7:"video_id", 8:"author_id", 9:"cate_two", 10:"upload_type", 11:"upload_time", 12:"cate_one_level_one", 
    # 13:"effective_view",14:"request_rank_pos",15:"request_final_pos"
    
    request_id = self.request_ids[idx]
    
    request_ts_df = self.data_group.get_group(request_id)
    
    request_ts_arr = request_ts_df.to_numpy().copy()
    
    n_photo = request_ts_arr.shape[0]
    
    n_complent = self.max_candidate_cnt - n_photo
    
    complent_arr = np.zeros(shape=[n_complent,14], dtype=np.int64)
    
    request_ts_arr = np.concatenate([request_ts_arr, complent_arr], axis=0)
    
    request_ts = request_ts_arr[0,2]
    
    request_ts_struct = time.localtime(request_ts//1000)
    
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = request_ts_arr[0,1] + 1
    did = request_ts_arr[0,3] + 1
    age = request_ts_arr[0,4] + 1
    gender = request_ts_arr[0,5] + 1
    province = request_ts_arr[0,6] + 1
    
    vid = request_ts_arr[:,7] + 1
    aid = request_ts_arr[:,8] + 1
    cate_two = request_ts_arr[:,9] + 1
    upload_type = request_ts_arr[:,10] + 1
    upload_ts = request_ts_arr[:,11]
    
    upload_ts_struct = [time.localtime(x) for x in upload_ts]
    upload_ts_wday = np.array([x.tm_wday + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_hour = np.array([x.tm_hour + 1 for x in upload_ts_struct], dtype=np.int64)
    upload_ts_min = np.array([x.tm_min + 1 for x in upload_ts_struct], dtype=np.int64)
    
    cate_one = request_ts_arr[:,12]
    
    effective = request_ts_arr[:,13]
    
    # 0:"video_id", 1:"author_id", 2:"cate_two", 
    # 3:"cate_one", 4:"upload_type", 5:"upload_time", 
    # 6:"duration_ms", 7:"time_ms", 8:"playing_time", 
    # 9:"request_id"
    #Order: video_id-0, author_id-1, cate_two-2, cate_one-3, upload_type-4, time_ms-5, request_id-6
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
          #0:'video_id',1:'author_id',2:'duration_ms',
          #3:'cate_two',4:'upload_type',5:'upload_time',
          #6:'cate_one_level_one',7:'effective_view'
          
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