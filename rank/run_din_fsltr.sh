set -x
set -e
set -o pipefail

tag=din_fsltr-1st

click_rank_loss_w=1.0
realshow_rank_loss_w=0.5
rerank_pos_rank_loss_w=0.05
rank_pos_rank_loss_w=0.05

python -B -u run_din_fsltr.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=512 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--click_rank_loss_w=${click_rank_loss_w} \
--realshow_rank_loss_w=${realshow_rank_loss_w} \
--rerank_pos_rank_loss_w=${rerank_pos_rank_loss_w} \
--rank_pos_rank_loss_w=${rank_pos_rank_loss_w} \
--tag=${tag} > "./logs/bs-1024_lr-1e-2_${click_rank_loss_w}_${realshow_rank_loss_w}_${rerank_pos_rank_loss_w}_${rank_pos_rank_loss_w}_${tag}.log" 2>&1

python -B -u eval_din_fsltr.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=512 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--click_rank_loss_w=${click_rank_loss_w} \
--realshow_rank_loss_w=${realshow_rank_loss_w} \
--rerank_pos_rank_loss_w=${rerank_pos_rank_loss_w} \
--rank_pos_rank_loss_w=${rank_pos_rank_loss_w} \
--tag=${tag} >> "./logs/bs-1024_lr-1e-2_${click_rank_loss_w}_${realshow_rank_loss_w}_${rerank_pos_rank_loss_w}_${rank_pos_rank_loss_w}_${tag}.log" 2>&1