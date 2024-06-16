set -x
set -e
set -o pipefail

tag=dssm_fsltr-1st

flows=click,realshow,rerank_pos,rerank_neg,rank_pos,rank_neg,coarse_neg
flow_nums=6,6,10,10,10,10,10
flow_weights=1.0,1.0,1.0,1.0,1.0,1.0,0.0

python -B -u run_dssm_fsltr.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=900 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--flows=${flows} \
--flow_nums=${flow_nums} \
--flow_weights=${flow_weights} \
--tag=${tag} > "./logs/bs-1024_lr-1e-2_${flows}_${flow_nums}_${flow_weights}_${tag}.log" 2>&1

python -B -u eval_dssm_fsltr.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=900 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--flows=${flows} \
--flow_nums=${flow_nums} \
--flow_weights=${flow_weights} \
--tag=${tag} >> "./logs/bs-1024_lr-1e-2_${flows}_${flow_nums}_${flow_weights}_${tag}.log" 2>&1