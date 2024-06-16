import os
import argparse

import torch
from torch.utils.data import DataLoader

from models import DIN_UBM
from dataset import Rank_Train_UBM_Dataset,Rank_Test_UBM_Dataset

from utils import load_pkl
from metrics import evaluate,evaluate_recall

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
  
  parser.add_argument('--flows', type=str, default="", help='exp tag.')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  #prepare data
  prefix = "../data"
  
  realshow_prefix = os.path.join(prefix, "realshow")
  path_to_test_csv = os.path.join(realshow_prefix, "2024-02-18.feather")
  print(f"testing file: {path_to_test_csv}")
  
  seq_prefix = os.path.join(prefix, "seq_effective_50_dict")
  path_to_test_seq_pkl = os.path.join(seq_prefix, "2024-02-18.pkl")
  print(f"testing seq file: {path_to_test_seq_pkl}")
  
  request_id_prefix = os.path.join(prefix, "ubm_seq_request_id_dict")
  path_to_request_id_pkl = os.path.join(request_id_prefix, "2024-02-18.pkl")
  print(f"testing request_id file: {path_to_test_seq_pkl}")
  
  others_prefix = os.path.join(prefix, "others")
  path_to_id_cnt_pkl = os.path.join(others_prefix, "id_cnt.pkl")
  print(f"path_to_id_cnt_pkl: {path_to_id_cnt_pkl}")
  
  id_cnt_dict = load_pkl(path_to_id_cnt_pkl)
  for k,v in id_cnt_dict.items():
    print(f"{k}:{v}")
    
  path_to_test_pkl = os.path.join(others_prefix, "rank_test.feather")
  print(f"path_to_test_pkl: {path_to_test_pkl}")

  #prepare model
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"device: {device}")
  
  per_flow_seq_len = 10
  n_flows = len(args.flows.split(','))
  flow_seq_len = per_flow_seq_len * n_flows
  
  max_candidate_cnt = 430
  
  model = DIN_UBM(
    args.emb_dim, 
    args.seq_len, 
    device, 
    max_candidate_cnt, 
    per_flow_seq_len,flow_seq_len,
    id_cnt_dict
  ).to(device)
  
  path_to_save_model=f"./checkpoints/bs-{args.batch_size}_lr-{args.lr}_{args.flows}_{args.tag}.pkl"
  
  state_dict = torch.load(path_to_save_model)
  
  model.load_state_dict(state_dict)
  
  print("testing: realshow")

  test_realshow_dataset = Rank_Train_UBM_Dataset(
    path_to_test_csv,
    args.seq_len,
    path_to_test_seq_pkl,
    path_to_request_id_pkl,
    args.flows,
    per_flow_seq_len
  )
  
  test_realshow_loader = DataLoader(
    dataset=test_realshow_dataset, 
    batch_size=args.infer_realshow_batch_size, 
    shuffle=False, 
    num_workers=0, 
    drop_last=True
  )
  
  print_str = evaluate(model, test_realshow_loader, device)
  
  print("testing: recall")
  
  test_recall_dataset = Rank_Test_UBM_Dataset(
    path_to_test_pkl,
    args.seq_len,
    path_to_test_seq_pkl,
    path_to_request_id_pkl,
    args.flows, per_flow_seq_len,
    max_candidate_cnt
  )
  
  test_recall_loader = DataLoader(
    dataset=test_recall_dataset, 
    batch_size=args.infer_recall_batch_size, 
    shuffle=False, 
    num_workers=0, 
    drop_last=True
  )
  
  target_print = evaluate_recall(model, test_recall_loader, device)
  
  print("realshow")
  print(print_str)
  
  print("recall")
  print(target_print[0])
  print(target_print[1])