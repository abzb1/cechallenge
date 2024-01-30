torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:6788 \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
inference.py