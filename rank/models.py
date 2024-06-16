import torch
import torch.nn as nn
import torch.nn.functional as F

class DIN(nn.Module):
  def __init__(
    self, 
    emb_dim,
    seq_len, 
    device, 
    max_candidate_cnt, 
    id_cnt_dict
  ):
    super(DIN, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device
    self.max_candidate_cnt = max_candidate_cnt

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
      num_embeddings=id_cnt_dict['video_id'] + 2,
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

    self.din_mlp = nn.Sequential(
      nn.Linear(emb_dim*10, 80),
      nn.ReLU(),
      nn.Linear(80, 1),
    )
    
    #encoder    
    self.mlp = nn.Sequential(
      nn.Linear(emb_dim*(8+13), 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )

  def forward(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs

    #context emb
    req_wda_emb = self.wday_emb(request_wday) #b*d
    req_hou_emb = self.hour_emb(request_hour) #b*d
    req_min_emb = self.min_emb(request_min) #b*d
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    #photo emb
    vid_emb = self.vid_emb(vid) #b*d
    aid_emb = self.aid_emb(aid) #b*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*d
    up_emb = self.up_type_emb(upload_type) #b*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*d
    up_min_emb = self.min_emb(upload_ts_min) #b*d
    
    #behaivor emb
    seq_mask_bool = torch.ne(seq_mask, 0) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
    target = torch.cat([vid_emb,aid_emb,cate_two_emb,cate_one_emb,up_emb], dim=1) #b*5d
    target_3dim_repeat = target.unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #b*seq_len*5d
    
    din_inputs = torch.cat([seq_emb,target_3dim_repeat], dim=2) 
    din_logits = self.din_mlp(din_inputs) ##b*seq_len*1
    din_logits = torch.transpose(din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    din_logits = torch.where(
      seq_mask_bool.unsqueeze(dim=1),  #b*1*seq_len
      din_logits,  #b*1*seq_len
      torch.full_like(din_logits, fill_value=padding_num)
    )
    
    din_scores = F.softmax(din_logits, dim=2) #b*1*seq_len
    din_interest = torch.bmm(din_scores, seq_emb).squeeze() #b*5d
    
    mlp_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb,
      din_interest,
      vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
      up_wda_emb, up_hou_emb, up_min_emb
      ],dim=1) #b*21d

    logits = self.mlp(mlp_input) #b*1

    return logits.squeeze() #b

  def forward_recall(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs
    
    #context emb
    req_wda_emb = self.wday_emb(request_wday) #b*d
    req_hou_emb = self.hour_emb(request_hour) #b*d
    req_min_emb = self.min_emb(request_min) #b*d
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d
    
    #photo emb
    vid_emb = self.vid_emb(vid) #b*n*d
    aid_emb = self.aid_emb(aid) #b*n*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*n*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*n*d
    up_emb = self.up_type_emb(upload_type) #b*n*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*n*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*n*d
    up_min_emb = self.min_emb(upload_ts_min) #b*n*d
    
    #behaivor emb
    seq_mask_bool = torch.ne(seq_mask, 0) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
    seq_emb_repeat = torch.repeat_interleave(seq_emb, self.max_candidate_cnt, dim=0) #bn*seq_len*5d
    
    target = torch.cat([vid_emb,aid_emb,cate_two_emb,cate_one_emb,up_emb], dim=1) #b*n*5d
    
    target_2dim = torch.reshape(target, [-1, self.emb_dim*5]) #bn*5d
    
    target_3dim_repeat = target_2dim.unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bn*seq_len*5d
    
    din_inputs = torch.cat([seq_emb_repeat, target_3dim_repeat], dim=2)  #bn*seq_len*10d
    
    din_logits = self.din_mlp(din_inputs) #bn*seq_len*1
    
    din_logits = torch.transpose(din_logits, 2,1) #bn*1*seq_len
    
    padding_num = -2**30 + 1
    
    din_logits = torch.where(
      torch.repeat_interleave(seq_mask_bool, self.max_candidate_cnt, dim=0).unsqueeze(1),  #bn*1*seq_len
      din_logits,  #bn*1*seq_len
      torch.full_like(din_logits, fill_value=padding_num)
    )
    
    din_scores = F.softmax(din_logits, dim=2) #bn*1*seq_len
    din_interest = torch.bmm(din_scores, seq_emb_repeat).squeeze() #bn*1*seq_len @ #bn*seq_len*5d -> bn*5d
    
    u_inputs = torch.cat([req_wda_emb, req_hou_emb, req_min_emb, uid_emb, did_emb, gen_emb, age_emb, pro_emb],dim=1) #b*8d
    
    u_inputs_3dim = u_inputs.unsqueeze(1).expand(-1,self.max_candidate_cnt,-1) #b*n*8d
    
    mlp_input = torch.cat([
      u_inputs_3dim,
      torch.reshape(din_interest, [-1, self.max_candidate_cnt, self.emb_dim*5]), #b*n*5d
      vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
      up_wda_emb, up_hou_emb, up_min_emb
      ],dim=2) #b*n*5d

    logits = self.mlp(mlp_input) #b*n*1
    
    logits = logits.squeeze() #b*n

    return logits

  def forward_fsltr(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    seq_arr, seq_mask, seq_len, \
    click_photos, realshow_photos, rerank_pos_photos, \
    rerank_neg_photos, rank_pos_photos, rank_neg_photos = inputs

    #context emb
    req_wda_emb = self.wday_emb(request_wday) #b*d
    req_hou_emb = self.hour_emb(request_hour) #b*d
    req_min_emb = self.min_emb(request_min) #b*d
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    user_side_emb = torch.cat([req_wda_emb, req_hou_emb, req_min_emb, uid_emb, did_emb, gen_emb, age_emb, pro_emb],dim=1) #b*8d
    
    user_side_emb_b6 = user_side_emb.unsqueeze(1).repeat([1,6,1]) #b*6*5d
    user_side_emb_b10 = user_side_emb.unsqueeze(1).repeat([1,10,1]) #b*10*5d
    
    #behaivor emb
    seq_mask_bool = torch.ne(seq_mask, 0) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
    seq_emb_b6 = seq_emb.repeat_interleave(6,dim=0) # b6*t*5d
    
    seq_emb_b10 = seq_emb.repeat_interleave(10,dim=0)
    
    seq_mask_bool_b6 = seq_mask_bool.repeat_interleave(6,dim=0).unsqueeze(dim=1) #b6*1*seq_len
    seq_mask_bool_b10 = seq_mask_bool.repeat_interleave(10,dim=0).unsqueeze(dim=1) #b10*1*seq_len
    
    #photo emb
    #click
    click_vid_emb = self.vid_emb(click_photos[:,:,0]) #b*6*d
    click_aid_emb = self.aid_emb(click_photos[:,:,1]) #b*6*d
    click_cate_two_emb = self.cate_two_emb(click_photos[:,:,2]) #b*6*d
    click_cate_one_emb = self.cate_one_emb(click_photos[:,:,3]) #b*6*d
    click_up_emb = self.up_type_emb(click_photos[:,:,4]) #b*6*d
    
    click_up_wda_emb = self.wday_emb(click_photos[:,:,5]) #b*6*d
    click_up_hou_emb = self.hour_emb(click_photos[:,:,6]) #b*6*d
    click_up_min_emb = self.min_emb(click_photos[:,:,7]) #b*6*d
    
    click_p_input = torch.cat([click_vid_emb, click_aid_emb, click_cate_two_emb, click_cate_one_emb, click_up_emb], dim=2) #b*6*5d
    
    click_p_3dim_repeat = click_p_input.reshape(-1,5*self.emb_dim).unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #b6*seq_len*5d
    
    click_p_din_inputs = torch.cat([seq_emb_b6,click_p_3dim_repeat], dim=2) # b6*seq_len*10d
    
    click_p_din_logits = self.din_mlp(click_p_din_inputs) # b6*seq_len*1
    
    click_p_din_logits = torch.transpose(click_p_din_logits, 2,1) # b6*1*seq_len
    
    padding_num = -2**30 + 1
    
    click_p_din_logits = torch.where(
      seq_mask_bool_b6,  #b6*1*seq_len
      click_p_din_logits,  # b6*1*seq_len
      torch.full_like(click_p_din_logits, fill_value=padding_num)
    )
    
    click_p_din_scores = F.softmax(click_p_din_logits, dim=2) # b6*1*seq_len
    click_p_din_interest = torch.bmm(click_p_din_scores, seq_emb_b6).squeeze() #  # b6*1*seq_len @ b6*seq_len*5d -> b6*5d
      
    click_p_mlp_input = torch.cat([
      user_side_emb_b6,
      click_p_din_interest.reshape(-1,6,5*self.emb_dim),
      click_p_input,
      click_up_wda_emb, click_up_hou_emb, click_up_min_emb
      ],dim=2) #b*21d

    click_logits = self.mlp(click_p_mlp_input).reshape(-1,6) #b*p
    
    
    #realshow
    realshow_vid_emb = self.vid_emb(realshow_photos[:,:,0]) #b*p*d
    realshow_aid_emb = self.aid_emb(realshow_photos[:,:,1]) #b*p*d
    realshow_cate_two_emb = self.cate_two_emb(realshow_photos[:,:,2]) #b*p*d
    realshow_cate_one_emb = self.cate_one_emb(realshow_photos[:,:,3]) #b*p*d
    realshow_up_emb = self.up_type_emb(realshow_photos[:,:,4]) #b*p*d
    
    realshow_up_wda_emb = self.wday_emb(realshow_photos[:,:,5]) #b*p*d
    realshow_up_hou_emb = self.hour_emb(realshow_photos[:,:,6]) #b*p*d
    realshow_up_min_emb = self.min_emb(realshow_photos[:,:,7]) #b*p*d
    
    realshow_p_input = torch.cat([realshow_vid_emb, realshow_aid_emb, realshow_cate_two_emb, realshow_cate_one_emb, realshow_up_emb], dim=2) #b*p*5d
    
    realshow_p_3dim_repeat = realshow_p_input.reshape(-1,5*self.emb_dim).unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bp*seq_len*5d
    
    realshow_p_din_inputs = torch.cat([seq_emb_b6,realshow_p_3dim_repeat], dim=2) 
    realshow_p_din_logits = self.din_mlp(realshow_p_din_inputs) ##b*seq_len*1
    realshow_p_din_logits = torch.transpose(realshow_p_din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    realshow_p_din_logits = torch.where(
      seq_mask_bool_b6,  #b*1*seq_len
      realshow_p_din_logits,  #b*1*seq_len
      torch.full_like(realshow_p_din_logits, fill_value=padding_num)
    )
    
    realshow_p_din_scores = F.softmax(realshow_p_din_logits, dim=2) #b*1*seq_len
    realshow_p_din_interest = torch.bmm(realshow_p_din_scores, seq_emb_b6).squeeze() #b*5d
    
    realshow_p_mlp_input = torch.cat([
      user_side_emb_b6,
      realshow_p_din_interest.reshape(-1,6,5*self.emb_dim),
      realshow_p_input,
      realshow_up_wda_emb, realshow_up_hou_emb, realshow_up_min_emb
      ],dim=2) #b*21d

    realshow_logits = self.mlp(realshow_p_mlp_input).reshape(-1,6) #b*p
    
    
    #rerank_pos
    rerank_pos_vid_emb = self.vid_emb(rerank_pos_photos[:,:,0]) #b*p*d
    rerank_pos_aid_emb = self.aid_emb(rerank_pos_photos[:,:,1]) #b*p*d
    rerank_pos_cate_two_emb = self.cate_two_emb(rerank_pos_photos[:,:,2]) #b*p*d
    rerank_pos_cate_one_emb = self.cate_one_emb(rerank_pos_photos[:,:,3]) #b*p*d
    rerank_pos_up_emb = self.up_type_emb(rerank_pos_photos[:,:,4]) #b*p*d
    
    rerank_pos_up_wda_emb = self.wday_emb(rerank_pos_photos[:,:,5]) #b*p*d
    rerank_pos_up_hou_emb = self.hour_emb(rerank_pos_photos[:,:,6]) #b*p*d
    rerank_pos_up_min_emb = self.min_emb(rerank_pos_photos[:,:,7]) #b*p*d
    
    rerank_pos_p_input = torch.cat([rerank_pos_vid_emb, rerank_pos_aid_emb, rerank_pos_cate_two_emb, rerank_pos_cate_one_emb, rerank_pos_up_emb], dim=2) #b*p*5d
    
    rerank_pos_p_3dim_repeat = rerank_pos_p_input.reshape(-1,5*self.emb_dim).unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bp*seq_len*5d
    
    rerank_pos_p_din_inputs = torch.cat([seq_emb_b10,rerank_pos_p_3dim_repeat], dim=2) 
    rerank_pos_p_din_logits = self.din_mlp(rerank_pos_p_din_inputs) ##b*seq_len*1
    rerank_pos_p_din_logits = torch.transpose(rerank_pos_p_din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    rerank_pos_p_din_logits = torch.where(
      seq_mask_bool_b10,  #b*1*seq_len
      rerank_pos_p_din_logits,  #b*1*seq_len
      torch.full_like(rerank_pos_p_din_logits, fill_value=padding_num)
    )
    
    rerank_pos_p_din_scores = F.softmax(rerank_pos_p_din_logits, dim=2) #b*1*seq_len
    rerank_pos_p_din_interest = torch.bmm(rerank_pos_p_din_scores, seq_emb_b10).squeeze() #b*5d
    
    rerank_pos_p_mlp_input = torch.cat([
      user_side_emb_b10,
      rerank_pos_p_din_interest.reshape(-1,10,5*self.emb_dim),
      rerank_pos_p_input,
      rerank_pos_up_wda_emb, rerank_pos_up_hou_emb, rerank_pos_up_min_emb
      ],dim=2) #b*21d

    rerank_pos_logits = self.mlp(rerank_pos_p_mlp_input).reshape(-1,10) #b*p
    
    
    #rerank_neg
    rerank_neg_vid_emb = self.vid_emb(rerank_neg_photos[:,:,0]) #b*p*d
    rerank_neg_aid_emb = self.aid_emb(rerank_neg_photos[:,:,1]) #b*p*d
    rerank_neg_cate_two_emb = self.cate_two_emb(rerank_neg_photos[:,:,2]) #b*p*d
    rerank_neg_cate_one_emb = self.cate_one_emb(rerank_neg_photos[:,:,3]) #b*p*d
    rerank_neg_up_emb = self.up_type_emb(rerank_neg_photos[:,:,4]) #b*p*d
    
    rerank_neg_up_wda_emb = self.wday_emb(rerank_neg_photos[:,:,5]) #b*p*d
    rerank_neg_up_hou_emb = self.hour_emb(rerank_neg_photos[:,:,6]) #b*p*d
    rerank_neg_up_min_emb = self.min_emb(rerank_neg_photos[:,:,7]) #b*p*d
    
    rerank_neg_p_input = torch.cat([rerank_neg_vid_emb, rerank_neg_aid_emb, rerank_neg_cate_two_emb, rerank_neg_cate_one_emb, rerank_neg_up_emb], dim=2) #b*p*5d
    
    rerank_neg_p_3dim_repeat = rerank_neg_p_input.reshape(-1,5*self.emb_dim).unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bp*seq_len*5d
    
    rerank_neg_p_din_inputs = torch.cat([seq_emb_b10,rerank_neg_p_3dim_repeat], dim=2) 
    rerank_neg_p_din_logits = self.din_mlp(rerank_neg_p_din_inputs) ##b*seq_len*1
    rerank_neg_p_din_logits = torch.transpose(rerank_neg_p_din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    rerank_neg_p_din_logits = torch.where(
      seq_mask_bool_b10,  #b*1*seq_len
      rerank_neg_p_din_logits,  #b*1*seq_len
      torch.full_like(rerank_neg_p_din_logits, fill_value=padding_num)
    )
    
    rerank_neg_p_din_scores = F.softmax(rerank_neg_p_din_logits, dim=2) #b*1*seq_len
    rerank_neg_p_din_interest = torch.bmm(rerank_neg_p_din_scores, seq_emb_b10).squeeze() #b*5d
    
    rerank_neg_p_mlp_input = torch.cat([
      user_side_emb_b10,
      rerank_neg_p_din_interest.reshape(-1,10,5*self.emb_dim),
      rerank_neg_p_input,
      rerank_neg_up_wda_emb, rerank_neg_up_hou_emb, rerank_neg_up_min_emb
      ],dim=2) #b*21d

    rerank_neg_logits = self.mlp(rerank_neg_p_mlp_input).reshape(-1,10) #b*p
    
    
    #rank_pos
    rank_pos_vid_emb = self.vid_emb(rank_pos_photos[:,:,0]) #b*p*d
    rank_pos_aid_emb = self.aid_emb(rank_pos_photos[:,:,1]) #b*p*d
    rank_pos_cate_two_emb = self.cate_two_emb(rank_pos_photos[:,:,2]) #b*p*d
    rank_pos_cate_one_emb = self.cate_one_emb(rank_pos_photos[:,:,3]) #b*p*d
    rank_pos_up_emb = self.up_type_emb(rank_pos_photos[:,:,4]) #b*p*d
    
    rank_pos_up_wda_emb = self.wday_emb(rank_pos_photos[:,:,5]) #b*p*d
    rank_pos_up_hou_emb = self.hour_emb(rank_pos_photos[:,:,6]) #b*p*d
    rank_pos_up_min_emb = self.min_emb(rank_pos_photos[:,:,7]) #b*p*d
    
    rank_pos_p_input = torch.cat([rank_pos_vid_emb, rank_pos_aid_emb, rank_pos_cate_two_emb, rank_pos_cate_one_emb, rank_pos_up_emb], dim=2) #b*p*5d
    
    rank_pos_p_3dim_repeat = rank_pos_p_input.reshape(-1,5*self.emb_dim).unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bp*seq_len*5d
    
    rank_pos_p_din_inputs = torch.cat([seq_emb_b10,rank_pos_p_3dim_repeat], dim=2) 
    rank_pos_p_din_logits = self.din_mlp(rank_pos_p_din_inputs) ##b*seq_len*1
    rank_pos_p_din_logits = torch.transpose(rank_pos_p_din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    rank_pos_p_din_logits = torch.where(
      seq_mask_bool_b10,  #b*1*seq_len
      rank_pos_p_din_logits,  #b*1*seq_len
      torch.full_like(rank_pos_p_din_logits, fill_value=padding_num)
    )
    
    rank_pos_p_din_scores = F.softmax(rank_pos_p_din_logits, dim=2) #b*1*seq_len
    rank_pos_p_din_interest = torch.bmm(rank_pos_p_din_scores, seq_emb_b10).squeeze() #b*5d
    
    rank_pos_p_mlp_input = torch.cat([
      user_side_emb_b10,
      rank_pos_p_din_interest.reshape(-1,10,5*self.emb_dim),
      rank_pos_p_input,
      rank_pos_up_wda_emb, rank_pos_up_hou_emb, rank_pos_up_min_emb
      ],dim=2) #b*21d

    rank_pos_logits = self.mlp(rank_pos_p_mlp_input).reshape(-1,10) #b*p
    
    
    #rank_neg
    rank_neg_vid_emb = self.vid_emb(rank_neg_photos[:,:,0]) #b*p*d
    rank_neg_aid_emb = self.aid_emb(rank_neg_photos[:,:,1]) #b*p*d
    rank_neg_cate_two_emb = self.cate_two_emb(rank_neg_photos[:,:,2]) #b*p*d
    rank_neg_cate_one_emb = self.cate_one_emb(rank_neg_photos[:,:,3]) #b*p*d
    rank_neg_up_emb = self.up_type_emb(rank_neg_photos[:,:,4]) #b*p*d
    
    rank_neg_up_wda_emb = self.wday_emb(rank_neg_photos[:,:,5]) #b*p*d
    rank_neg_up_hou_emb = self.hour_emb(rank_neg_photos[:,:,6]) #b*p*d
    rank_neg_up_min_emb = self.min_emb(rank_neg_photos[:,:,7]) #b*p*d
    
    rank_neg_p_input = torch.cat([rank_neg_vid_emb, rank_neg_aid_emb, rank_neg_cate_two_emb, rank_neg_cate_one_emb, rank_neg_up_emb], dim=2) #b*p*5d
    
    rank_neg_p_3dim_repeat = rank_neg_p_input.reshape(-1,5*self.emb_dim).unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bp*seq_len*5d
    
    rank_neg_p_din_inputs = torch.cat([seq_emb_b10,rank_neg_p_3dim_repeat], dim=2) 
    rank_neg_p_din_logits = self.din_mlp(rank_neg_p_din_inputs) ##b*seq_len*1
    rank_neg_p_din_logits = torch.transpose(rank_neg_p_din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    rank_neg_p_din_logits = torch.where(
      seq_mask_bool_b10,  #b*1*seq_len
      rank_neg_p_din_logits,  #b*1*seq_len
      torch.full_like(rank_neg_p_din_logits, fill_value=padding_num)
    )
    
    rank_neg_p_din_scores = F.softmax(rank_neg_p_din_logits, dim=2) #b*1*seq_len
    rank_neg_p_din_interest = torch.bmm(rank_neg_p_din_scores, seq_emb_b10).squeeze() #b*5d
    
    rank_neg_p_mlp_input = torch.cat([
      user_side_emb_b10,
      rank_neg_p_din_interest.reshape(-1,10,5*self.emb_dim),
      rank_neg_p_input,
      rank_neg_up_wda_emb, rank_neg_up_hou_emb, rank_neg_up_min_emb
      ],dim=2) #b*21d

    rank_neg_logits = self.mlp(rank_neg_p_mlp_input).reshape(-1,10) #b*p

    return click_logits, realshow_logits, rerank_pos_logits, rerank_neg_logits, rank_pos_logits, rank_neg_logits
  
class DIN_AuxRanking(nn.Module):
  def __init__(
    self, 
    emb_dim, seq_len, 
    device, max_candidate_cnt, id_cnt_dict
  ):
    super(DIN_AuxRanking, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device
    self.max_candidate_cnt = max_candidate_cnt
        
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
      num_embeddings=id_cnt_dict['video_id'] + 2,
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

    self.din_mlp = nn.Sequential(
      nn.Linear(emb_dim*10, 80),
      nn.ReLU(),
      nn.Linear(80, 1),
    )
    
    self.self_agg_mlp = nn.Sequential(
      nn.Linear(emb_dim*5, 80),
      nn.ReLU(),
      nn.Linear(80, 1),
    )
    
    #encoder    
    self.mlp = nn.Sequential(
      nn.Linear(emb_dim*(8+13), 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )
    
    self.aux_user_mlp = nn.Sequential(
      nn.Linear(emb_dim*18, 80),
      nn.ReLU(),
      nn.Linear(80, 32),
    )
    
    self.aux_photo_mlp = nn.Sequential(
      nn.Linear(emb_dim*8, 80),
      nn.ReLU(),
      nn.Linear(80, 32),
    )

  def forward_train(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len, \
    flow_arr = inputs

    #context emb
    req_wda_emb = self.wday_emb(request_wday) #b*d
    req_hou_emb = self.hour_emb(request_hour) #b*d
    req_min_emb = self.min_emb(request_min) #b*d
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    #photo emb
    vid_emb = self.vid_emb(vid) #b*d
    aid_emb = self.aid_emb(aid) #b*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*d
    up_emb = self.up_type_emb(upload_type) #b*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*d
    up_min_emb = self.min_emb(upload_ts_min) #b*d
    
    #behaivor emb
    seq_mask_bool = torch.ne(seq_mask, 0) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
    target = torch.cat([vid_emb,aid_emb,cate_two_emb,cate_one_emb,up_emb], dim=1) #b*5d
    target_3dim_repeat = target.unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #b*seq_len*5d
    
    din_inputs = torch.cat([seq_emb,target_3dim_repeat], dim=2) 
    din_logits = self.din_mlp(din_inputs) ##b*seq_len*1
    din_logits = torch.transpose(din_logits,2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    din_logits = torch.where(
      seq_mask_bool.unsqueeze(dim=1),  #b*1*seq_len
      din_logits,  #b*1*seq_len
      torch.full_like(din_logits, fill_value=padding_num)
    )
    
    din_scores = F.softmax(din_logits, dim=2) #b*1*seq_len
    din_interest = torch.bmm(din_scores, seq_emb).squeeze() #b*5d
    
    #self-agg behavior embedding
    self_agg_logits = self.self_agg_mlp(seq_emb) #b*seq_len*1
    self_agg_logits = torch.transpose(self_agg_logits, 2, 1) #b*seq_len*1
    
    self_agg_logits = torch.where(
      seq_mask_bool.unsqueeze(dim=1),  #b*1*seq_len
      self_agg_logits,  #b*1*seq_len
      torch.full_like(self_agg_logits, fill_value=padding_num)
    )
    
    self_agg_score = F.softmax(self_agg_logits, dim=2) #b*1*seq_len
    self_agg_interest = torch.bmm(self_agg_score, seq_emb).squeeze() #b*5d
    
    mean_interest = torch.sum(seq_emb, dim=1) / seq_len.float().unsqueeze(-1)
    
    #mlp
    user_side_emb = torch.cat([req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb], dim=1)
    
    photo_side_emb = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
      up_wda_emb, up_hou_emb, up_min_emb], dim=1)
    
    mlp_input = torch.cat([user_side_emb,din_interest,photo_side_emb],dim=1) #b*21d

    logits = self.mlp(mlp_input) #b*1
    
    #aux task
    
    user_side_input = torch.cat([user_side_emb,self_agg_interest,mean_interest], dim=1)
    
    user_side_top = self.aux_user_mlp(user_side_input) #b*32
    
    user_side_top = F.normalize(user_side_top, p=2, dim=1)
    
    photo_side_top = self.aux_photo_mlp(photo_side_emb)
    
    photo_side_top = F.normalize(photo_side_top, p=2, dim=1)
    
    aux_logits = torch.sum(user_side_top*photo_side_top, dim=1, keepdim=True) # b*1
    
    #flow photo emb
    vid_emb = self.vid_emb(flow_arr[:,:,0]) #b*n*d
    aid_emb = self.aid_emb(flow_arr[:,:,1]) #b*n*d
    cate_two_emb = self.cate_two_emb(flow_arr[:,:,2]) #b*n*d
    cate_one_emb = self.cate_one_emb(flow_arr[:,:,3]) #b*n*d
    up_emb = self.up_type_emb(flow_arr[:,:,4]) #b*n*d
    
    up_wda_emb = self.wday_emb(flow_arr[:,:,5]) #b*n*d
    up_hou_emb = self.hour_emb(flow_arr[:,:,6]) #b*n*d
    up_min_emb = self.min_emb(flow_arr[:,:,7]) #b*n*d
    
    photo_side_emb = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb, up_wda_emb, up_hou_emb, up_min_emb], dim=2)  #b*n*8d
    
    photo_side_top = self.aux_photo_mlp(photo_side_emb) #b*n*32
    
    photo_side_top = F.normalize(photo_side_top, p=2, dim=2)
    
    flow_logits = torch.bmm(photo_side_top, user_side_top.unsqueeze(dim=-1)).squeeze() #b*n

    return logits, aux_logits, flow_logits

  def forward(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs

    #context emb
    req_wda_emb = self.wday_emb(request_wday) #b*d
    req_hou_emb = self.hour_emb(request_hour) #b*d
    req_min_emb = self.min_emb(request_min) #b*d
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d

    #photo emb
    vid_emb = self.vid_emb(vid) #b*d
    aid_emb = self.aid_emb(aid) #b*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*d
    up_emb = self.up_type_emb(upload_type) #b*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*d
    up_min_emb = self.min_emb(upload_ts_min) #b*d
    
    #behaivor emb
    seq_mask_bool = torch.ne(seq_mask, 0) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
    target = torch.cat([vid_emb,aid_emb,cate_two_emb,cate_one_emb,up_emb], dim=1) #b*5d
    target_3dim_repeat = target.unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #b*seq_len*5d
    
    din_inputs = torch.cat([seq_emb,target_3dim_repeat], dim=2) 
    din_logits = self.din_mlp(din_inputs) ##b*seq_len*1
    din_logits = torch.transpose(din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    din_logits = torch.where(
      seq_mask_bool.unsqueeze(dim=1),  #b*1*seq_len
      din_logits,  #b*1*seq_len
      torch.full_like(din_logits, fill_value=padding_num)
    )
    
    din_scores = F.softmax(din_logits, dim=2) #b*1*seq_len
    din_interest = torch.bmm(din_scores, seq_emb).squeeze() #b*5d
    
    mlp_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb,
      din_interest,
      vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
      up_wda_emb, up_hou_emb, up_min_emb
      ],dim=1) #b*21d

    logits = self.mlp(mlp_input) #b*1

    return logits.squeeze() #b
  
  def forward_recall(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len = inputs
    
    #context emb
    req_wda_emb = self.wday_emb(request_wday) #b*d
    req_hou_emb = self.hour_emb(request_hour) #b*d
    req_min_emb = self.min_emb(request_min) #b*d
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d
    
    #photo emb
    vid_emb = self.vid_emb(vid) #b*n*d
    aid_emb = self.aid_emb(aid) #b*n*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*n*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*n*d
    up_emb = self.up_type_emb(upload_type) #b*n*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*n*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*n*d
    up_min_emb = self.min_emb(upload_ts_min) #b*n*d
    
    #behaivor emb
    seq_mask_bool = torch.ne(seq_mask, 0) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d

    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
    seq_emb_repeat = torch.repeat_interleave(seq_emb, self.max_candidate_cnt, dim=0) #bn*seq_len*5d
    
    target = torch.cat([vid_emb,aid_emb,cate_two_emb,cate_one_emb,up_emb], dim=1) #b*n*5d
    
    target_2dim = torch.reshape(target, [-1, self.emb_dim*5]) #bn*5d
    
    target_3dim_repeat = target_2dim.unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bn*seq_len*5d
    
    din_inputs = torch.cat([seq_emb_repeat, target_3dim_repeat], dim=2)  #bn*seq_len*10d
    
    din_logits = self.din_mlp(din_inputs) #bn*seq_len*1
    
    din_logits = torch.transpose(din_logits, 2,1) #bn*1*seq_len
    
    padding_num = -2**30 + 1
    
    din_logits = torch.where(
      torch.repeat_interleave(seq_mask_bool, self.max_candidate_cnt, dim=0).unsqueeze(1),  #bn*1*seq_len
      din_logits,  #bn*1*seq_len
      torch.full_like(din_logits, fill_value=padding_num)
    )
    
    din_scores = F.softmax(din_logits, dim=2) #bn*1*seq_len
    din_interest = torch.bmm(din_scores, seq_emb_repeat).squeeze() #bn*1*seq_len @ #bn*seq_len*5d -> bn*5d
    
    u_inputs = torch.cat([req_wda_emb, req_hou_emb, req_min_emb, uid_emb, did_emb, gen_emb, age_emb, pro_emb],dim=1) #b*8d
    
    u_inputs_3dim = u_inputs.unsqueeze(1).expand(-1,self.max_candidate_cnt,-1) #b*n*8d
    
    mlp_input = torch.cat([
      u_inputs_3dim,
      torch.reshape(din_interest, [-1, self.max_candidate_cnt, self.emb_dim*5]), #b*n*5d
      vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
      up_wda_emb, up_hou_emb, up_min_emb
      ],dim=2) #b*n*5d

    logits = self.mlp(mlp_input) #b*n*1
    
    logits = logits.squeeze() #b*n

    return logits

class DIN_UBM(nn.Module):
  def __init__(
    self, 
    emb_dim, 
    seq_len, 
    device, 
    max_candidate_cnt, 
    per_flow_seq_len, 
    flow_seq_len,
    id_cnt_dict
  ):
    super(DIN_UBM, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device
    self.max_candidate_cnt = max_candidate_cnt
    
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
      num_embeddings=id_cnt_dict['video_id'] + 2,
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

    self.din_mlp = nn.Sequential(
      nn.Linear(emb_dim*10, 80),
      nn.ReLU(),
      nn.Linear(80, 1),
    )
    
    self.carm_mlp = nn.Sequential(
      nn.Linear(emb_dim*10, 80),
      nn.ReLU(),
      nn.Linear(80, 1),
    )
    
    #encoder    
    self.mlp = nn.Sequential(
      nn.Linear(emb_dim*(8+13), 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
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
    seq_mask_bool = torch.ne(seq_mask, 0).to(self.device) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d
    
    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
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
      torch.full_like(flow_din_logits, fill_value=padding_num) #b*seq_len*1*flow_seq_len
    ) #b*seq_len*1*flow_seq_len
    
    flow_din_scores = F.softmax(flow_din_logits, dim=3) #b*seq_len*1*flow_seq_len
    
    seq_flow_din_representation = torch.matmul(flow_din_scores, flwo_seq_emb).squeeze() #b*seq_len*d
    
    #scate_twoe two
    target = torch.cat([vid_emb,aid_emb,cate_two_emb,cate_one_emb,up_emb], dim=1) #b*5d
    target_3dim_repeat = target.unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #b*seq_len*5d
    
    din_inputs = torch.cat([seq_flow_din_representation,target_3dim_repeat], dim=2) 
    din_logits = self.din_mlp(din_inputs) ##b*seq_len*1
    din_logits = torch.transpose(din_logits, 2,1) #b*1*seq_len
    
    padding_num = -2**30 + 1
    
    din_logits = torch.where(
      seq_mask_bool.unsqueeze(dim=1),  #b*1*seq_len
      din_logits,  #b*1*seq_len
      torch.full_like(din_logits, fill_value=padding_num)
    )
    
    din_scores = F.softmax(din_logits, dim=2)
    din_interest = torch.bmm(din_scores, seq_flow_din_representation).squeeze()
    
    mlp_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb,
      din_interest,
      vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
      up_wda_emb, up_hou_emb, up_min_emb
      ],dim=1)

    logits = self.mlp(mlp_input)

    return logits.squeeze() #b

  def forward_recall(self, inputs):
    request_wday,request_hour,request_min, \
    uid, did, gender, age, province, \
    vid, aid, cate_two, cate_one, upload_type, upload_ts_wday, upload_ts_hour, upload_ts_min, \
    seq_arr, seq_mask, seq_len, \
    flow_seq_arr, flow_seq_mask = inputs
    
    #context emb
    req_wda_emb = self.wday_emb(request_wday) #b*d
    req_hou_emb = self.hour_emb(request_hour) #b*d
    req_min_emb = self.min_emb(request_min) #b*d
    
    #user emb
    uid_emb = self.uid_emb(uid) #b*d
    did_emb = self.did_emb(did) #b*d
    gen_emb = self.gender_emb(gender) #b*d
    age_emb = self.age_emb(age) #b*d
    pro_emb = self.province_emb(province) #b*d
    
    #photo emb
    vid_emb = self.vid_emb(vid) #b*n*d
    aid_emb = self.aid_emb(aid) #b*n*d
    cate_two_emb = self.cate_two_emb(cate_two) #b*n*d
    cate_one_emb = self.cate_one_emb(cate_one) #b*n*d
    up_emb = self.up_type_emb(upload_type) #b*n*d
    
    up_wda_emb = self.wday_emb(upload_ts_wday) #b*n*d
    up_hou_emb = self.hour_emb(upload_ts_hour) #b*n*d
    up_min_emb = self.min_emb(upload_ts_min) #b*n*d
    
    #behaivor emb
    seq_mask_bool = torch.ne(seq_mask, 0) #b*seq_len
    
    vid_seq_emb = self.vid_emb(seq_arr[:,:,0]) #b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:,:,1]) #b*seq _len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:,:,2]) #b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:,:,3]) #b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:,:,4]) #b*seq_len*d
    
    seq_emb = torch.cat([vid_seq_emb,aid_seq_emb,cate_two_seq_emb,cate_one_seq_emb,up_seq_emb], dim=2) #b*seq_len*5d
    
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
      torch.full_like(flow_din_logits, fill_value=padding_num) #b*seq_len*1*flow_seq_len
    ) #b*seq_len*1*flow_seq_len
    
    flow_din_scores = F.softmax(flow_din_logits, dim=3) #b*seq_len*1*flow_seq_len
    
    seq_flow_din_representation = torch.matmul(flow_din_scores, flwo_seq_emb).squeeze() #b*seq_len*d
    
    #scate_twoe 2
    seq_flow_din_representation_repeat = torch.repeat_interleave(seq_flow_din_representation, self.max_candidate_cnt, dim=0) #bn*seq_len*5d
    
    target = torch.cat([vid_emb,aid_emb,cate_two_emb,cate_one_emb,up_emb], dim=1) #b*n*5d
    
    target_2dim = torch.reshape(target, [-1, self.emb_dim*5]) #bn*5d
    
    target_3dim_repeat = target_2dim.unsqueeze(dim=1).expand([-1,self.seq_len,-1]) #bn*seq_len*5d
    
    din_inputs = torch.cat([seq_flow_din_representation_repeat, target_3dim_repeat], dim=2)  #bn*seq_len*10d
    
    din_logits = self.din_mlp(din_inputs) #bn*seq_len*1
    
    din_logits = torch.transpose(din_logits, 2,1) #bn*1*seq_len
    
    padding_num = -2**30 + 1
    
    din_logits = torch.where(
      torch.repeat_interleave(seq_mask_bool, self.max_candidate_cnt, dim=0).unsqueeze(1),  #bn*1*seq_len
      din_logits,  #bn*1*seq_len
      torch.full_like(din_logits, fill_value=padding_num)
    )
    
    din_scores = F.softmax(din_logits, dim=2) #bn*1*seq_len
    din_interest = torch.bmm(din_scores, seq_flow_din_representation_repeat).squeeze() #bn*1*seq_len @ #bn*seq_len*5d -> bn*5d
    
    u_inputs = torch.cat([req_wda_emb, req_hou_emb, req_min_emb, uid_emb, did_emb, gen_emb, age_emb, pro_emb],dim=1) #b*8d
    
    u_inputs_3dim = u_inputs.unsqueeze(1).expand(-1,self.max_candidate_cnt,-1) #b*n*8d
    
    mlp_input = torch.cat([
      u_inputs_3dim,
      torch.reshape(din_interest, [-1, self.max_candidate_cnt, self.emb_dim*5]), #b*n*5d
      vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
      up_wda_emb, up_hou_emb, up_min_emb
      ],dim=2) #b*n*5d

    logits = self.mlp(mlp_input) #b*n*1
    
    logits = logits.squeeze() #b*n

    return logits