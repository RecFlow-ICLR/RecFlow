set -x
set -e
set -o pipefail

tag=din_data_dist_shift_sampling-1st

flows=rank_neg
k_flow_negs=1

python -B -u run_din_data_dist_shift_sampling.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=512 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--flows=${flows} \
--k_flow_negs=${k_flow_negs} \
--tag=${tag} > "./logs/bs-1024_lr-1e-2_${flows}_${k_flow_negs}_${tag}.log" 2>&1

python -B -u eval_din_data_dist_shift_sampling.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=512 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--flows=${flows} \
--k_flow_negs=${k_flow_negs} \
--tag=${tag} >> "./logs/bs-1024_lr-1e-2_${flows}_${k_flow_negs}_${tag}.log" 2>&1