set -x
set -e
set -o pipefail

tag=dssm_auxiliary_ranking-1st

flows=rerank_pos,rerank_neg,rank_pos,rank_neg

rank_loss_weight=0.1

python -B -u run_dssm_auxiliary_ranking.py \
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
--rank_loss_weight=${rank_loss_weight} \
--tag=${tag} > "./logs/bs-1024_lr-1e-2_${flows}_${rank_loss_weight}_${tag}.log" 2>&1

python -B -u eval_dssm_auxiliary_ranking.py \
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
--rank_loss_weight=${rank_loss_weight} \
--tag=${tag} >> "./logs/bs-1024_lr-1e-2_${flows}_${rank_loss_weight}_${tag}.log" 2>&1