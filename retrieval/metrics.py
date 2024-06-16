import gc
import faiss

import torch
import numpy as np
import pandas as pd
  
def evaluate_recall(
  model, data_loader, device,
  path_to_realshow_video_corpus_feather
):
  
  model.eval()
  
  realshow_video_corpus_df = pd.read_feather(path_to_realshow_video_corpus_feather)
  
  realshow_video_corpus = realshow_video_corpus_df['video_id'].unique().copy() + 1
  
  del realshow_video_corpus_df
  
  gc.collect()

  target_top_k = [50, 100, 500, 1000]
  
  total_target_cnt = 0.0
  
  target_realshow_recall_lst = [0.0 for _ in range(len(target_top_k))]
  target_realshow_ndcg_lst = [0.0 for _ in range(len(target_top_k))]
  
  with torch.no_grad():
    
    #construct realshow embedding
    realshow_faiss_obj = faiss.StandardGpuResources()
    realshow_flat_config = faiss.GpuIndexFlatConfig()
    realshow_flat_config.device = 0
    realshow_index_flat = faiss.GpuIndexFlatIP(realshow_faiss_obj, 8, realshow_flat_config)
    realshow_index_flat.add(model.vid_emb.weight.cpu().numpy()[realshow_video_corpus])

    for idx,inputs in enumerate(data_loader):
      
      inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:-3]]
    
      user_emb = model.forward_recall(inputs_LongTensor) #b*d
      
      _, topk_realshow_logits_index = realshow_index_flat.search(user_emb.cpu().numpy(), k=1000)

      topk_realshow_videos = realshow_video_corpus[topk_realshow_logits_index] #k*b
      
      vids = inputs[-3].numpy().astype(np.int64) #b*30
      labels = inputs[-2].numpy().astype(np.float) #b*30
      
      n_videos = inputs[-1].numpy() #b
      
      for i in range(n_videos.shape[0]):
        
        n_video = n_videos[i]
        
        topk_realshow_video = topk_realshow_videos[i] 
        
        vid = vids[i,:n_video]
        label = labels[i,:n_video]
        
        #target metric
        if np.sum(label) > 0:
          target_pos_index = np.nonzero(label)[0]
          target_pos_vid = vid[target_pos_index]

          target_pos_realshow_rank = np.where(topk_realshow_video == target_pos_vid[:,None])[1]
          if target_pos_realshow_rank.shape[0] > 0:
            for i in range(len(target_top_k)):
              target_realshow_recall_lst[i] += np.sum(target_pos_realshow_rank<target_top_k[i])
              target_realshow_ndcg_lst[i] += np.sum((1.0/np.log2(target_pos_realshow_rank+2))*(target_pos_realshow_rank<target_top_k[i]))
          
          total_target_cnt += np.sum(label)

  target_realshow_recall = []
  target_realshow_ndcg = []
  
  for i in range(len(target_top_k)):
    target_realshow_recall.append(target_realshow_recall_lst[i]/total_target_cnt)
    target_realshow_ndcg.append(target_realshow_ndcg_lst[i]/total_target_cnt)
  
  target_print_str = f"Target: "
  for i in range(len(target_top_k)):
    target_print_str += f"realshow_recall@{target_top_k[i]},"
    target_print_str += f"realshow_ndcg@{target_top_k[i]},"
  
  target_print_value_str = f""
  for i in range(len(target_top_k)):
    target_print_value_str += f"{target_realshow_recall[i]:.6f},"
    target_print_value_str += f"{target_realshow_ndcg[i]:.6f},"

  return target_print_str[:-1],target_print_value_str[:-1]