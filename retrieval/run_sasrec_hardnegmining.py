import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import SASRec
from dataset import Recall_Train_SASRec_HardNegMining_Dataset

from utils import load_pkl

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--epochs', type=int, default=1, help='epochs.')
  parser.add_argument('--batch_size', type=int, default=1024, help='train batch size.')
  parser.add_argument('--infer_batch_size', type=int, default=1024, help='inference batch size.')
  parser.add_argument('--emb_dim', type=int, default=8, help='embedding dimension.')
  parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
  parser.add_argument('--seq_len', type=int, default=3, help='length of behaivor sequence')
  
  parser.add_argument('--cuda', type=int, default=0, help='cuda device.')

  parser.add_argument('--print_freq', type=int, default=200, help='frequency of print.')
  
  parser.add_argument('--tag', type=str, default="1st", help='exp tag.')
  
  parser.add_argument('--neg_num', type=int, default=3, help='number of negative samples')
  
  parser.add_argument('--flow_negs', type=str, default='mcd_prerank_neg', help='model name.')
  
  parser.add_argument('--flow_neg_nums', type=str, default=3, help='number of negative samples')
  
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  for k,v in vars(args).items():
    print(f"{k}:{v}")
  
  #prepare data
  prefix = "../data"
  
  realshow_prefix = os.path.join(prefix, "realshow")
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
    
  #prepare seq
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
  
  #prepare request_id
  request_id_prefix = os.path.join(prefix, "request_id_dict")
  path_to_request_id_pkl_lst = []
  with open("./file.txt", mode='r') as f:
    lines = f.readlines()
    for line in lines:
      tmp_request_id_pkl_path = os.path.join(request_id_prefix, line.strip()+'.pkl')
      path_to_request_id_pkl_lst.append(tmp_request_id_pkl_path)
  
  print("training request_id files:")
  for idx, filepath in enumerate(path_to_request_id_pkl_lst):
    print(f"{idx}: {filepath}")
  
  #prepare id_cnt
  others_prefix = os.path.join(prefix, "others")
  path_to_id_cnt_pkl = os.path.join(others_prefix, "id_cnt.pkl")
  print(f"path_to_id_cnt_pkl: {path_to_id_cnt_pkl}")
  
  id_cnt_dict = load_pkl(path_to_id_cnt_pkl)
  for k,v in id_cnt_dict.items():
    print(f"{k}:{v}")
  
  #prepare negatives  
  video_prefix = os.path.join(others_prefix, "realshow_video_info_daily")
  path_to_video_info_feather_lst = []
  with open("./file.txt", mode='r') as f:
    lines = f.readlines()
    for line in lines:
      tmp_video_feather_path = os.path.join(video_prefix, line.strip()+'.feather')
      path_to_video_info_feather_lst.append(tmp_video_feather_path)
  
  print("realshow daily negative")
  for idx, filepath in enumerate(path_to_video_info_feather_lst):
    print(f"{idx}: {filepath}")
  
  #prepare model
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"device: {device}")
  
  model = SASRec(args.emb_dim, args.seq_len, args.neg_num, device, id_cnt_dict).to(device)
  
  loss_fn = nn.LogSigmoid().to(device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  
  #training
  for epoch in range(args.epochs):
    for n_day in range(num_of_train_csv):
      
      train_dataset = Recall_Train_SASRec_HardNegMining_Dataset(
        path_to_train_csv_lst[n_day],
        args.seq_len, args.neg_num,
        path_to_train_seq_pkl_lst[n_day],
        path_to_request_id_pkl_lst[n_day],
        path_to_video_info_feather_lst[n_day],
        args.flow_negs,
        args.flow_neg_nums
      )
    
      train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1, 
        drop_last=False
      )
      
      for iter_step, inputs in enumerate(train_loader):
        
        inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs]
        
        tgt_logits, neg_logits = model(inputs_LongTensor) #b
      
        pos_logits_expand = tgt_logits.repeat_interleave(args.neg_num)
        
        loss = -loss_fn(pos_logits_expand-neg_logits).mean()

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        if iter_step % args.print_freq == 0:
          print(f"Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.6f}")
      
  path_to_save_model=f"./checkpoints/bs-{args.batch_size}_lr-{args.lr}_neg_num-{args.neg_num}_flow_negs-{args.flow_negs}_flow_neg_nums-{args.flow_neg_nums}_{args.tag}.pkl"
  
  torch.save(model.state_dict(), path_to_save_model)
  
  print(f"save model to {path_to_save_model} DONE.")