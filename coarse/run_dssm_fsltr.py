import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import DSSM
from dataset import Prerank_Train_FSLTR_Dataset

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
  
  parser.add_argument('--flows', type=str, default='request_prerank_neg', help='model name.')
  
  parser.add_argument('--flow_nums', type=str, default="1", help='number of negative samples')
  
  parser.add_argument('--flow_weights', type=str, default="1.0", help='learning rate.')
  
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
  
  model = DSSM(args.emb_dim, args.seq_len, device, id_cnt_dict).to(device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  
  sum_num = 0
  weight_lst = []
  
  nums = args.flow_nums.split(',')
  weights = args.flow_weights.split(',')

  for idx,num in enumerate(nums):
    sum_num += int(num)
    weight_lst.extend([float(weights[idx])]*int(num))
  
  loss_weight_gpu = torch.tensor(weight_lst, dtype=torch.float32, device=device).reshape([1,-1,1]) #1*p*1
  
  padding_num = -2**30 + 1
  
  #training
  for epoch in range(args.epochs):
    for n_day in range(num_of_train_csv):
      train_dataset = Prerank_Train_FSLTR_Dataset(
        path_to_train_csv_lst[n_day],
        args.seq_len,
        path_to_train_seq_pkl_lst[n_day],
        path_to_train_request_pkl_lst[n_day],
        args.flows,
        args.flow_nums
      )
    
      train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1, 
        drop_last=True
      )

      for iter_step, inputs in enumerate(train_loader):
        
        inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:-1]]
        
        logits = model.forward_fsltr(inputs_LongTensor) #b
        
        priority = torch.FloatTensor(inputs[-1].numpy()).to(device) #b*p
        
        weight = torch.gt(
          priority.unsqueeze(-1), priority.unsqueeze(1)
        ) #b*p*p
        
        logits_diff = logits.unsqueeze(-1) - logits.unsqueeze(1)
        
        loss = F.binary_cross_entropy_with_logits(
          logits_diff, 
          torch.ones_like(logits_diff), 
          weight=weight*loss_weight_gpu, 
          reduction='sum') / weight.sum()
        
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        if iter_step % args.print_freq == 0:
          print(f"Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.6f}")
    
  path_to_save_model=f"./checkpoints/bs-{args.batch_size}_lr-{args.lr}_{args.loss}_{args.flows}_{args.flow_nums}_{args.flow_weights}_{args.tag}.pkl"
  
  torch.save(model.state_dict(), path_to_save_model)
  
  print(f"save model to {path_to_save_model} DONE.")