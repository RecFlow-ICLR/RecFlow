import torch
import torch.nn as nn
import torch.nn.functional as F

class DSSM(nn.Module):
  def __init__(self, emb_dim, seq_len, device, id_cnt_dict):
    super(DSSM, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device

    #user
    self.uid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['user_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.did_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['device_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.gender_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['gender'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.age_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['age'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.province_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['province'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #item
    self.vid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['photo_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.aid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['author_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_two_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_two'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_one_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_one']+2,
      embedding_dim=emb_dim,
      padding_idx=2
    )
    self.up_type_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['upload_type'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #context
    self.wday_emb = nn.Embedding(
      num_embeddings= 7 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    
    self.hour_emb = nn.Embedding(
      num_embeddings= 24 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    
    self.min_emb = nn.Embedding(
      num_embeddings= 60 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #encoder
    self.user_encoder = nn.Sequential(
      nn.Linear(emb_dim*13, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32)
    )

    self.photo_encoder = nn.Sequential(
      nn.Linear(emb_dim*8, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32)
    )

  def forward(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs

    #context emb    
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    #photo emb
    vid_emb = self.vid_emb(vid) #b*d
    aid_emb = self.aid_emb(aid) #b*d
    cate_two_emb = self.cate_two_emb(cate_two)
    cate_one_emb = self.cate_one_emb(cate_one)
    up_emb = self.up_type_emb(upload_type)
    
    up_wda_emb = self.wday_emb(upload_ts_wday)
    up_hou_emb = self.hour_emb(upload_ts_hour)
    up_min_emb = self.min_emb(upload_ts_min)
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_sum / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean
      ], dim=1)
    
    p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    p_out = self.photo_encoder(p_input) #b*32

    logits = torch.sum(u_out*p_out, dim=1) #b

    return logits
      
  def forward_recall(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs
    
    #context emb
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_sum / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean
      ], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    
    #photo emb
    vid_emb = self.vid_emb(vid) #b*n*d
    aid_emb = self.aid_emb(aid) #b*n*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*n*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*n*d
    up_emb = self.up_type_emb(upload_type) #b*n*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*n*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*n*d
    up_min_emb = self.min_emb(upload_ts_min) #b*n*d
    
    p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=2)

    p_out = self.photo_encoder(p_input) #b*32
    
    logits = torch.bmm(u_out.unsqueeze(dim=1), p_out.transpose(2,1)).squeeze() #b*n

    return logits
  
  def forward_fsltr(self, inputs):
    request_wday, request_hour, request_min, \
    uid, did, gender, age, province, \
    seq_arr, seq_mask, seq_len, \
    flow_photo = inputs
    
    #context emb
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_sum / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean
      ], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    
    #flow photo
    flow_vid_emb = self.vid_emb(flow_photo[:,:,0]) #b*p*d
    flow_aid_emb = self.aid_emb(flow_photo[:,:,1]) #b*p*d
    flow_cate_two_emb = self.cate_two_emb(flow_photo[:,:,2]) #b*p*d
    flow_cate_one_emb = self.cate_one_emb(flow_photo[:,:,3]) #b*p*d
    flow_up_emb = self.up_type_emb(flow_photo[:,:,4]) #b*p*d
    
    flow_up_wda_emb = self.wday_emb(flow_photo[:,:,5]) #b*p*d
    flow_up_hou_emb = self.hour_emb(flow_photo[:,:,6]) #b*p*d
    flow_up_min_emb = self.min_emb(flow_photo[:,:,7]) #b*p*d
    
    flow_p_input = torch.cat([flow_vid_emb, flow_aid_emb, flow_cate_two_emb, flow_cate_one_emb, flow_up_emb,
    flow_up_wda_emb, flow_up_hou_emb, flow_up_min_emb], dim=2) #b*p*8d
    
    flow_p_out = self.photo_encoder(flow_p_input) #b*32

    logits = torch.bmm(flow_p_out, u_out.unsqueeze(dim=-1)).squeeze() #b*p
  
    return logits #b*p

class DSSM_AuxRanking(nn.Module):
  def __init__(
    self, 
    emb_dim, 
    seq_len, 
    device, 
    id_cnt_dict
  ):
    super(DSSM_AuxRanking, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device

    #user
    self.uid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['user_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.did_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['device_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.gender_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['gender'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.age_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['age'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.province_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['province'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #item
    self.vid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['photo_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.aid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['author_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_two_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_two'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_one_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_one']+2,
      embedding_dim=emb_dim,
      padding_idx=2
    )
    self.up_type_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['upload_type'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #context
    self.wday_emb = nn.Embedding(
      num_embeddings= 7 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.hour_emb = nn.Embedding(
      num_embeddings= 24 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.min_emb = nn.Embedding(
      num_embeddings= 60 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #encoder
    self.user_encoder = nn.Sequential(
      nn.Linear(emb_dim*13, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32)
    )

    self.photo_encoder = nn.Sequential(
      nn.Linear(emb_dim*8, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32)
    )

  def forward(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs

    #context emb    
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    #photo emb
    vid_emb = self.vid_emb(vid) #b*d
    aid_emb = self.aid_emb(aid) #b*d
    cate_two_emb = self.cate_two_emb(cate_two)
    cate_one_emb = self.cate_one_emb(cate_one)
    up_emb = self.up_type_emb(upload_type)
    
    up_wda_emb = self.wday_emb(upload_ts_wday)
    up_hou_emb = self.hour_emb(upload_ts_hour)
    up_min_emb = self.min_emb(upload_ts_min)
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_sum / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean
      ], dim=1)
    
    p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    p_out = self.photo_encoder(p_input) #b*32

    logits = torch.sum(u_out*p_out, dim=1)

    return logits
  
  def forward_train(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len, \
    flow_arr = inputs

    #context emb    
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    #photo emb
    vid_emb = self.vid_emb(vid) #b*d
    aid_emb = self.aid_emb(aid) #b*d
    cate_two_emb = self.cate_two_emb(cate_two)
    cate_one_emb = self.cate_one_emb(cate_one)
    up_emb = self.up_type_emb(upload_type)
    
    up_wda_emb = self.wday_emb(upload_ts_wday)
    up_hou_emb = self.hour_emb(upload_ts_hour)
    up_min_emb = self.min_emb(upload_ts_min)
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_sum / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean
      ], dim=1)
    
    p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    p_out = self.photo_encoder(p_input) #b*32

    logits = torch.sum(u_out*p_out, dim=1, keepdim=True)
    
    #flow photo emb
    flow_vid_emb = self.vid_emb(flow_arr[:,:,0]) #b*n*d
    flow_aid_emb = self.aid_emb(flow_arr[:,:,1]) #b*n*d
    flow_cate_two_emb = self.cate_two_emb(flow_arr[:,:,2]) #b*n*d
    flow_cate_one_emb = self.cate_one_emb(flow_arr[:,:,3]) #b*n*d
    flow_up_emb = self.up_type_emb(flow_arr[:,:,4]) #b*n*d
    
    flow_up_wda_emb = self.wday_emb(flow_arr[:,:,5]) #b*n*d
    flow_up_hou_emb = self.hour_emb(flow_arr[:,:,6]) #b*n*d
    flow_up_min_emb = self.min_emb(flow_arr[:,:,7]) #b*n*d
    
    flow_photo_side_emb = torch.cat([flow_vid_emb, flow_aid_emb, flow_cate_two_emb, flow_cate_one_emb, flow_up_emb,
    flow_up_wda_emb, flow_up_hou_emb, flow_up_min_emb], dim=2)  #b*n*8d
    
    flow_photo_side_top = self.photo_encoder(flow_photo_side_emb) #b*n*32
    
    flow_logits = torch.bmm(flow_photo_side_top, u_out.unsqueeze(dim=-1)).squeeze() #b*n
    
    return logits, flow_logits
  
  def forward_recall(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs
    
    #context emb
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_sum / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean
      ], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    
    #photo emb
    vid_emb = self.vid_emb(vid) #b*n*d
    aid_emb = self.aid_emb(aid) #b*n*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*n*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*n*d
    up_emb = self.up_type_emb(upload_type) #b*n*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*n*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*n*d
    up_min_emb = self.min_emb(upload_ts_min) #b*n*d
    
    p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=2)

    p_out = self.photo_encoder(p_input) #b*32
    
    logits = torch.bmm(u_out.unsqueeze(dim=1), p_out.transpose(2,1)).squeeze() #b*n

    return logits

class DSSM_UBM(nn.Module):
  def __init__(
    self, 
    emb_dim, 
    seq_len,
    device, 
    per_flow_seq_len, 
    flow_seq_len,
    id_cnt_dict
  ):
    super(DSSM_UBM, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device

    self.per_flow_seq_len = per_flow_seq_len
    self.flow_seq_len = flow_seq_len

    #user
    self.uid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['user_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.did_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['device_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.gender_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['gender'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.age_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['age'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.province_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['province'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #item
    self.vid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['photo_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.aid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['author_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_two_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_two'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_one_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_one']+2,
      embedding_dim=emb_dim,
      padding_idx=2
    )
    self.up_type_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['upload_type'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #context
    self.wday_emb = nn.Embedding(
      num_embeddings= 7 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.hour_emb = nn.Embedding(
      num_embeddings= 24 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.min_emb = nn.Embedding(
      num_embeddings= 60 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.carm_mlp = nn.Sequential(
      nn.Linear(emb_dim*10, 80),
      nn.ReLU(),
      nn.Linear(80, 1),
    )
    
    #encoder
    self.user_encoder = nn.Sequential(
      nn.Linear(emb_dim*(13+5), 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32)
    )

    self.photo_encoder = nn.Sequential(
      nn.Linear(emb_dim*8, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32)
    )

  def forward(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len, \
    flow_seq_arr, flow_seq_mask = inputs

    #context emb    
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    #photo emb
    vid_emb = self.vid_emb(vid) #b*d
    aid_emb = self.aid_emb(aid) #b*d
    cate_two_emb = self.cate_two_emb(cate_two)
    cate_one_emb = self.cate_one_emb(cate_one)
    up_emb = self.up_type_emb(upload_type)
    
    up_wda_emb = self.wday_emb(upload_ts_wday)
    up_hou_emb = self.hour_emb(upload_ts_hour)
    up_min_emb = self.min_emb(upload_ts_min)
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_emb_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_emb_sum / seq_len #b*5d
    
    #flow behaivor emb
    flow_seq_mask_bool = torch.ne(flow_seq_mask, 0) #b*seq_len*flow_seq_len
    
    flow_vid_seq_emb = self.vid_emb(flow_seq_arr[:,:,:,0]) #b*seq_len*flow_seq_len*d
    flow_aid_seq_emb = self.aid_emb(flow_seq_arr[:,:,:,1]) #b*seq_len*flow_seq_len*d
    flow_cate_two_seq_emb = self.cate_two_emb(flow_seq_arr[:,:,:,2]) #b*seq_len*flow_seq_len*d
    flow_cate_one_seq_emb = self.cate_one_emb(flow_seq_arr[:,:,:,3]) #b*seq_len*flow_seq_len*d
    flow_up_seq_emb = self.up_type_emb(flow_seq_arr[:,:,:,4]) #b*seq_len*flow_seq_len*d

    flwo_seq_emb = torch.cat([
      flow_vid_seq_emb,
      flow_aid_seq_emb,flow_cate_two_seq_emb,
      flow_cate_one_seq_emb,flow_up_seq_emb],
      dim=3
    ) #b*seq_len*flow_seq_len*5d
    
    seq_emb_4dim = seq_emb.unsqueeze(dim=2).expand([-1,-1,self.flow_seq_len,-1]) #b*seq_len*flow_seq_len*5d
    
    flow_din_inputs = torch.cat([flwo_seq_emb,seq_emb_4dim], dim=3)  #b*seq_len*flow_seq_len*10d
    
    flow_din_logits = self.carm_mlp(flow_din_inputs) #b*seq_len*flow_seq_len*1
    
    flow_din_logits = torch.transpose(flow_din_logits, 3, 2) #b*seq_len*1*flow_seq_len
    
    padding_num = -2**30 + 1
    
    flow_din_logits = torch.where(
      flow_seq_mask_bool.unsqueeze(dim=2),  #b*seq_len*1*flow_seq_len
      flow_din_logits,  #b*seq_len*1*flow_seq_len
      torch.fulreq_like(flow_din_logits, filreq_value=padding_num) #b*seq_len*1*flow_seq_len
    ) #b*seq_len*1*flow_seq_len
    
    flow_din_scores = F.softmax(flow_din_logits, dim=3) #b*seq_len*1*flow_seq_len
    
    seq_flow_din_representation = torch.matmul(flow_din_scores, flwo_seq_emb).squeeze() #b*seq_len*d
    
    seq_flow_din_representation_mean = torch.sum(seq_flow_din_representation, dim=1) / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean,seq_flow_din_representation_mean
      ], dim=1)
    
    p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    p_out = self.photo_encoder(p_input) #b*32

    logits = torch.sum(u_out*p_out, dim=1)

    return logits
  
  def forward_recall(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len, \
    flow_seq_arr, flow_seq_mask = inputs
    
    #context emb
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d
    
    #behaivor emb
    seq_len = seq_len.float().unsqueeze(-1) #b*1
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    seq_emb_sum = torch.sum(seq_emb, dim=1)
    
    seq_emb_mean = seq_emb_sum / seq_len #b*5d
    
    #flow behaivor emb
    flow_seq_mask_bool = torch.ne(flow_seq_mask, 0) #b*seq_len*flow_seq_len
    
    flow_vid_seq_emb = self.vid_emb(flow_seq_arr[:,:,:,0]) #b*seq_len*flow_seq_len*d
    flow_aid_seq_emb = self.aid_emb(flow_seq_arr[:,:,:,1]) #b*seq_len*flow_seq_len*d
    flow_cate_two_seq_emb = self.cate_two_emb(flow_seq_arr[:,:,:,2]) #b*seq_len*flow_seq_len*d
    flow_cate_one_seq_emb = self.cate_one_emb(flow_seq_arr[:,:,:,3]) #b*seq_len*flow_seq_len*d
    flow_up_seq_emb = self.up_type_emb(flow_seq_arr[:,:,:,4]) #b*seq_len*flow_seq_len*d

    flwo_seq_emb = torch.cat([
      flow_vid_seq_emb,
      flow_aid_seq_emb,flow_cate_two_seq_emb,
      flow_cate_one_seq_emb,flow_up_seq_emb],
      dim=3
    ) #b*seq_len*flow_seq_len*5d
    
    seq_emb_4dim = seq_emb.unsqueeze(dim=2).expand([-1,-1,self.flow_seq_len,-1]) #b*seq_len*flow_seq_len*5d
    
    flow_din_inputs = torch.cat([flwo_seq_emb,seq_emb_4dim], dim=3)  #b*seq_len*flow_seq_len*10d
    
    flow_din_logits = self.carm_mlp(flow_din_inputs) #b*seq_len*flow_seq_len*1
    
    flow_din_logits = torch.transpose(flow_din_logits, 3, 2) #b*seq_len*1*flow_seq_len
    
    padding_num = -2**30 + 1
    
    flow_din_logits = torch.where(
      flow_seq_mask_bool.unsqueeze(dim=2),  #b*seq_len*1*flow_seq_len
      flow_din_logits,  #b*seq_len*1*flow_seq_len
      torch.fulreq_like(flow_din_logits, filreq_value=padding_num) #b*seq_len*1*flow_seq_len
    ) #b*seq_len*1*flow_seq_len
    
    flow_din_scores = F.softmax(flow_din_logits, dim=3) #b*seq_len*1*flow_seq_len
    
    seq_flow_din_representation = torch.matmul(flow_din_scores, flwo_seq_emb).squeeze() #b*seq_len*d
    
    seq_flow_din_representation_mean = torch.sum(seq_flow_din_representation, dim=1) / seq_len #b*5d
    
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb
      ,seq_emb_mean,seq_flow_din_representation_mean
      ], dim=1)

    u_out = self.user_encoder(u_input) #b*32
    
    #photo emb
    vid_emb = self.vid_emb(vid) #b*n*d
    aid_emb = self.aid_emb(aid) #b*n*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*n*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*n*d
    up_emb = self.up_type_emb(upload_type) #b*n*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*n*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*n*d
    up_min_emb = self.min_emb(upload_ts_min) #b*n*d
    
    p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=2)

    p_out = self.photo_encoder(p_input) #b*32
    
    logits = torch.bmm(u_out.unsqueeze(dim=1), p_out.transpose(2,1)).squeeze() #b*n

    return logits