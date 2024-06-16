set -x
set -e
set -o pipefail

tag=din-1st

python -B -u run_din.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=512 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--tag=${tag} > "./logs/bs-1024_lr-1e-2_${tag}.log" 2>&1

python -B -u eval_dssm.py \
--epochs=1 \
--batch_size=1024 \
--infer_realshow_batch_size=1024 \
--infer_recall_batch_size=512 \
--emb_dim=8 \
--lr=1e-2 \
--seq_len=50 \
--cuda='0' \
--print_freq=100 \
--tag=${tag} >> "./logs/bs-1024_lr-1e-2_${tag}.log" 2>&1