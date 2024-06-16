import torch
import torch.nn as nn
from modules import MultiHeadAttention,PositionwiseFeedForward

class SASRec(nn.Module):
  def __init__(self, emb_dim, seq_len, neg_num, device, id_cnt_dict, num_heads=1):
    super(SASRec, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    
    self.neg_num = neg_num
    
    self.device = device
    
    #item
    self.vid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['video_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.position = nn.Embedding(
      num_embeddings=seq_len,
      embedding_dim=emb_dim,
    )

    self.ln_1 = nn.LayerNorm(emb_dim)
    self.ln_2 = nn.LayerNorm(emb_dim)
    self.ln_3 = nn.LayerNorm(emb_dim)

    self.mh_attn = MultiHeadAttention(emb_dim, emb_dim, num_heads, 0.0)
    self.feed_forward = PositionwiseFeedForward(emb_dim, emb_dim, 0.0)

  
  def forward(self, inputs):
    seq, seq_mask, tgt_vid, neg_vids = inputs
    
    seq_emb = self.vid_emb(seq) #b*t*d
    tgt_emb = self.vid_emb(tgt_vid) #b*d
    neg_emb = self.vid_emb(neg_vids) #b*n*d

    position_emb = self.position(torch.arange(self.seq_len, dtype=torch.int64, device=self.device)) #t*d

    seq_emb = seq_emb + position_emb #b*t*d

    mask = torch.ne(seq_mask, 0).float().unsqueeze(-1) #b*t*1
    
    seq_emb *= mask

    seq_emb_ln = self.ln_1(seq_emb)

    mh_attn_out = self.mh_attn(seq_emb_ln, seq_emb)

    ff_out = self.feed_forward(self.ln_2(mh_attn_out))

    ff_out *= mask

    ff_out = self.ln_3(ff_out) #b*t*d

    final_state = ff_out[:,-1,:] #b*d

    tgt_logits = torch.sum(final_state*tgt_emb, dim=1) #b
    
    neg_logits = torch.bmm(final_state.unsqueeze(1), neg_emb.transpose(2,1)).squeeze() # b*1*d @ b*d*n -> b*1*n -> b*n
    
    neg_logits = neg_logits.view(-1) #bn
    
    return tgt_logits, neg_logits
  
  
  def forward_fsltr(self, inputs):
    seq, seq_mask, vids = inputs
    
    seq_emb = self.vid_emb(seq) #b*t*d
    vids_emb = self.vid_emb(vids) #b*d

    position_emb = self.position(torch.arange(self.seq_len, dtype=torch.int64, device=self.device)) #t*d

    seq_emb = seq_emb + position_emb #b*t*d

    mask = torch.ne(seq_mask, 0).float().unsqueeze(-1) #b*t*1
    
    seq_emb *= mask

    seq_emb_ln = self.ln_1(seq_emb)

    mh_attn_out = self.mh_attn(seq_emb_ln, seq_emb)

    ff_out = self.feed_forward(self.ln_2(mh_attn_out))

    ff_out *= mask

    ff_out = self.ln_3(ff_out) #b*t*d

    final_state = ff_out[:,-1,:] #b*d
    
    logits = torch.bmm(vids_emb, final_state.unsqueeze(-1)).squeeze()
    
    return logits


  def forward_recall(self, inputs):
    
    seq, seq_mask = inputs
    
    seq_emb = self.vid_emb(seq) #b*t*d

    position_emb = self.position(torch.arange(self.seq_len, dtype=torch.int64, device=self.device)) #t*d

    seq_emb += position_emb

    mask = torch.ne(seq_mask, 0).float().unsqueeze(-1) #b*seq_len*1

    seq_emb *= mask
    
    seq_emb_ln = self.ln_1(seq_emb)
    
    mh_attn_out = self.mh_attn(seq_emb_ln, seq_emb)

    ff_out = self.feed_forward(self.ln_2(mh_attn_out))
    
    ff_out *= mask
    
    ff_out = self.ln_3(ff_out)
    
    final_state = ff_out[:,-1,:] #b*d

    return final_state #b*d