import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

def evaluate(model, data_loader, device):
  model.eval()
  
  logits_lst = np.zeros(shape=(962560,), dtype=np.float32)
  label_lst = np.zeros(shape=(962560,), dtype=np.float32)
  
  with torch.no_grad():
    start_index = 0
    end_index = 0
    
    for inputs in data_loader:
          
      inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:-1]]
      
      logits = model(inputs_LongTensor) #b
      
      logits = torch.sigmoid(logits)
      
      end_index += inputs[-1].size(0)
      
      label_lst[start_index:end_index] = inputs[-1].numpy().astype(np.float32)
      
      logits_lst[start_index:end_index] = logits.cpu().numpy().astype(np.float32)
      
      start_index = end_index
      
  test_auc = roc_auc_score(label_lst, logits_lst)
  test_logloss = log_loss(label_lst, logits_lst)

  print_str = f"Target: auc \t logloss: {test_auc:.6f} \t {test_logloss:.6f}"
  
  return print_str
  

def evaluate_recall(model, data_loader, device):
  model.eval()
  
  target_top_k = [50,100,200]
  
  total_target_cnt = 0.0
  
  target_recall_lst = [0.0 for _ in range(len(target_top_k))]
  target_ndcg_lst = [0.0 for _ in range(len(target_top_k))]

  with torch.no_grad():
    
    for inputs in data_loader:
        
      inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:-2]]
      
      logits = model.forward_recall(inputs_LongTensor) #b*430
      logits = logits.cpu().numpy()
      
      labels = inputs[-2].numpy().astype(np.float) #b*430
      n_photos = inputs[-1].numpy() #b
      
      for i in range(n_photos.shape[0]):
        
        n_photo = n_photos[i]
        
        logit = logits[i,:n_photo]
        label = labels[i,:n_photo]
        
        logit_descending_index = np.argsort(logit*-1.0) #descending order
        logit_descending_rank = np.argsort(logit_descending_index) #descending order
        
        #target metric
        if np.sum(label) > 0 and np.sum(label)!=n_photo:
          target_pos_index = np.nonzero(label)[0]
          target_pos_rank = logit_descending_rank[target_pos_index]
          
          for i in range(len(target_top_k)):
            target_recall_lst[i] += np.sum(target_pos_rank<target_top_k[i])
            target_ndcg_lst[i] += np.sum((1.0/np.log2(target_pos_rank+2))*(target_pos_rank<target_top_k[i]))
            
          total_target_cnt += np.sum(label)

  target_recall = []
  target_ndcg = []
  
  for i in range(len(target_top_k)):
    target_recall.append(target_recall_lst[i]/total_target_cnt)
    target_ndcg.append(target_ndcg_lst[i]/total_target_cnt)
  
  target_print_str = f"Target: "
  for i in range(len(target_top_k)):
    target_print_str += f"recall@{target_top_k[i]},"
    target_print_str += f"ndcg@{target_top_k[i]},"
  
  target_print_value_str = f""
  for i in range(len(target_top_k)):
    target_print_value_str += f"{target_recall[i]:.6f},"
    target_print_value_str += f"{target_ndcg[i]:.6f},"

  return target_print_str[:-1], target_print_value_str[:-1]