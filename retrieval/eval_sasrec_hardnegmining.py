import os
import argparse

import torch
from torch.utils.data import DataLoader

from models import SASRec
from dataset import Recall_Test_SASRec_Recall_Dataset

from utils import load_pkl
from metrics import evaluate_recall

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

  #prepare data
  prefix = "../data"
  
  seq_prefix = os.path.join(prefix, "seq_effective_50_dict")
  path_to_test_seq_pkl = os.path.join(seq_prefix, "2024-02-18.pkl")
  print(f"testing seq file: {path_to_test_seq_pkl}")
  
  others_prefix = os.path.join(prefix, "others")
  path_to_id_cnt_pkl = os.path.join(others_prefix, "id_cnt.pkl")
  print(f"path_to_id_cnt_pkl: {path_to_id_cnt_pkl}")
  
  id_cnt_dict = load_pkl(path_to_id_cnt_pkl)
  for k,v in id_cnt_dict.items():
    print(f"{k}:{v}")
    
  #prepare negatives
  path_to_realshow_video_corpus_feather = os.path.join(others_prefix, "realshow_video_info.feather")
  print(f"path_to_video_corpus_pkl: {path_to_realshow_video_corpus_feather}")
  
  #prepare recal_test
  path_to_recall_test_feather = os.path.join(others_prefix, "recall_test.feather")
  print(f"path_to_recall_test_pkl: {path_to_recall_test_feather}")
  
  #prepare model
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"device: {device}")
  
  model = SASRec(
    args.emb_dim, args.seq_len, 
    args.neg_num,
    device, id_cnt_dict
  ).to(device)
  
  path_to_save_model=f"./checkpoints/bs-{args.batch_size}_lr-{args.lr}_neg_num-{args.neg_num}_flow_negs-{args.flow_negs}_flow_neg_nums-{args.flow_neg_nums}_{args.tag}.pkl"
  
  state_dict = torch.load(path_to_save_model)
  
  model.load_state_dict(state_dict)
  
  print("testing: recall")
  
  test_recall_dataset = Recall_Test_SASRec_Recall_Dataset(
    path_to_recall_test_feather,
    args.seq_len,
    path_to_test_seq_pkl,
    max_candidate_cnt=30
  )
  
  test_recall_loader = DataLoader(
    dataset=test_recall_dataset, 
    batch_size=args.infer_batch_size, 
    shuffle=False, 
    num_workers=0, 
    drop_last=True
  )
  
  target_print = evaluate_recall(
    model, 
    test_recall_loader, 
    device, 
    path_to_realshow_video_corpus_feather
  )
  
  print(target_print[0])
  print(target_print[1])