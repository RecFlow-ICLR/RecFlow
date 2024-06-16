import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import DIN
from dataset import Rank_Train_FSLTR_Dataset

from utils import load_pkl

def parse_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument('--epochs', type=int, default=1, help='epochs.')
  parser.add_argument('--batch_size', type=int, default=1024, help='train batch size.')
  parser.add_argument('--infer_realshow_batch_size', type=int, default=1024, help='inference batch size.')
  parser.add_argument('--infer_recall_batch_size', type=int, default=1024, help='inference batch size.')
  parser.add_argument('--emb_dim', type=int, default=8, help='embedding dimension.')
  parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
  parser.add_argument('--seq_len', type=int, default=3, help='length of behaivor sequence')
  
  parser.add_argument('--cuda', type=int, default=0, help='cuda device.')

  parser.add_argument('--print_freq', type=int, default=200, help='frequency of print.')
  
  parser.add_argument('--tag', type=str, default="1st", help='exp tag.')
  
  parser.add_argument('--click_rank_loss_w', type=float, default=1e-2, help='learning rate.')
  parser.add_argument('--realshow_rank_loss_w', type=float, default=1e-2, help='learning rate.')
  parser.add_argument('--rerank_pos_rank_loss_w', type=float, default=1e-2, help='learning rate.')
  parser.add_argument('--rank_pos_rank_loss_w', type=float, default=1e-2, help='learning rate.')
  
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  for k,v in vars(args).items():
    print(f"{k}:{v}")
    
  #prepare data
  prefix = "../data"
  
  realshow_prefix = os.path.join(prefix, "all_stage")
  path_to_train_csv_lst = []
  with open("./file.txt", mode='r') as f:
    lines = f.readlines()
    for line in lines:
      tmp_csv_path = os.path.join(realshow_prefix, line.strip()+'.feather')
      path_to_train_csv_lst.append(tmp_csv_path)
      
  num_of_train_csv = len(path_to_train_csv_lst)
  print("training files:")
  print(f"number of train_csv: {num_of_train_csv}")
  for idx, filepath in enumerate(path_to_train_csv_lst):
    print(f"{idx}: {filepath}")

  seq_prefix = os.path.join(prefix, "seq_effective_50_dict")
  path_to_train_seq_pkl_lst = []
  with open("./file.txt", mode='r') as f:
    lines = f.readlines()
    for line in lines:
      tmp_seq_pkl_path = os.path.join(seq_prefix, line.strip()+'.pkl')
      path_to_train_seq_pkl_lst.append(tmp_seq_pkl_path)
  
  print("training seq files:")
  for idx, filepath in enumerate(path_to_train_seq_pkl_lst):
    print(f"{idx}: {filepath}")
  
  request_id_prefix = os.path.join(prefix, "request_id_dict")
  path_to_train_request_pkl_lst = []
  with open("./file.txt", mode='r') as f:
    lines = f.readlines()
    for line in lines:
      tmp_request_pkl_path = os.path.join(request_id_prefix, line.strip()+".pkl")
      path_to_train_request_pkl_lst.append(tmp_request_pkl_path)
      
  print("training request files")
  for idx, filepath in enumerate(path_to_train_request_pkl_lst):
    print(f"{idx}: {filepath}")
  
  others_prefix = os.path.join(prefix, "others")
  path_to_id_cnt_pkl = os.path.join(others_prefix, "id_cnt.pkl")
  print(f"path_to_id_cnt_pkl: {path_to_id_cnt_pkl}")
  
  id_cnt_dict = load_pkl(path_to_id_cnt_pkl)
  for k,v in id_cnt_dict.items():
    print(f"{k}:{v}")
    
  #prepare model
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"device: {device}")
  
  max_candidate_cnt = 430
  
  model = DIN(
    args.emb_dim, args.seq_len, 
    device, max_candidate_cnt, id_cnt_dict
  ).to(device)
  
  loss_fn = nn.CrossEntropyLoss(ignore_index=1)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
  #training
  for epoch in range(args.epochs):
    for n_day in range(num_of_train_csv):
      
      train_dataset = Rank_Train_FSLTR_Dataset(
        path_to_train_csv_lst[n_day],
        args.seq_len,
        path_to_train_seq_pkl_lst[n_day],
        path_to_train_request_pkl_lst[n_day]
      )
    
      train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1, 
        drop_last=True
      )
      
      for iter_step, inputs in enumerate(train_loader):
        
        inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:-6]]
        
        click_logits, realshow_logits, \
        rerank_pos_logits, rerank_neg_logits, \
        rank_pos_logits, rank_neg_logits = model.forward_fsltr(inputs_LongTensor) #b
        
        tmp_logits = torch.cat([realshow_logits,rerank_pos_logits,rerank_neg_logits, rank_pos_logits, rank_neg_logits], dim=1)
        click_bpr_logits = click_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1) #b*6*46
        click_label = torch.FloatTensor(inputs[-6].numpy()).to(device).unsqueeze(-1) #b*6*1
        click_rank_loss = F.binary_cross_entropy_with_logits(
          click_bpr_logits, 
          torch.ones(click_bpr_logits.size(), dtype=torch.float, device=device), 
          weight=click_label, 
          reduction='sum') / (46*torch.sum(click_label))
        
        tmp_logits = torch.cat([rerank_pos_logits,rerank_neg_logits, rank_pos_logits, rank_neg_logits], dim=1)
        realshow_bpr_logits = realshow_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1) #b*6*40
        realshow_label = torch.FloatTensor(inputs[-5].numpy()).to(device).unsqueeze(-1) #b*6*1
        realshow_rank_loss = F.binary_cross_entropy_with_logits(
          realshow_bpr_logits, 
          torch.ones(realshow_bpr_logits.size(), dtype=torch.float, device=device), 
          weight=realshow_label, 
          reduction='sum') / (40*torch.sum(realshow_label))
        
        tmp_logits = torch.cat([rerank_neg_logits, rank_neg_logits], dim=1)
        rerank_pos_bpr_logits = rerank_pos_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1) #b*6*20
        rerank_pos_label = torch.FloatTensor(inputs[-4].numpy()).to(device).unsqueeze(-1) #b*10*1
        rerank_pos_rank_loss = F.binary_cross_entropy_with_logits(
          rerank_pos_bpr_logits, 
          torch.ones(rerank_pos_bpr_logits.size(), dtype=torch.float, device=device), 
          weight=rerank_pos_label, 
          reduction='sum') / (20*torch.sum(rerank_pos_label))
        
        tmp_logits = torch.cat([rerank_neg_logits, rank_neg_logits], dim=1)
        rank_pos_bpr_logits = rank_pos_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1) #b*6*20
        rank_pos_label = torch.FloatTensor(inputs[-2].numpy()).to(device).unsqueeze(-1) #b*10*1
        rank_pos_rank_loss = F.binary_cross_entropy_with_logits(
          rank_pos_bpr_logits, 
          torch.ones(rank_pos_bpr_logits.size(), dtype=torch.float, device=device), 
          weight=rank_pos_label, 
          reduction='sum') / (20*torch.sum(rank_pos_label))
          
        loss = click_rank_loss * args.click_rank_loss_w + \
          realshow_rank_loss * args.realshow_rank_loss_w + \
          rerank_pos_rank_loss * args.rerank_pos_rank_loss_w + \
          rank_pos_rank_loss * args.rank_pos_rank_loss_w
        
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        if iter_step % args.print_freq == 0:
          print(f"Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.6f} \tclick_rank_loss:{click_rank_loss.detach().cpu().item():.6f} \trealshow_rank_loss:{realshow_rank_loss.detach().cpu().item():.6f}\trerank_pos_rank_loss:{rerank_pos_rank_loss.detach().cpu().item():.6f}\trank_pos_rank_loss:{rank_pos_rank_loss.detach().cpu().item():.6f}")
      
  path_to_save_model=f"./checkpoints/bs-{args.batch_size}_lr-{args.lr}_{args.click_rank_loss_w}-{args.realshow_rank_loss_w}-{args.rerank_pos_rank_loss_w}-{args.rank_pos_rank_loss_w}_{args.tag}.pkl"
  
  torch.save(model.state_dict(), path_to_save_model)
  
  print(f"save model to {path_to_save_model} DONE.")