# Sourced by the training scripts. Fills TORCHRUN_LAUNCH with the torchrun rendezvous
# arguments: single node by default, multi-node when NNODES > 1.
#
#   NNODES       number of nodes                   (default 1)
#   NODE_RANK    rank of this node, 0..NNODES-1    (required when NNODES > 1)
#   MASTER_ADDR  address of node 0                 (required when NNODES > 1)
#   MASTER_PORT  port on node 0                    (default 29500)
#
# Run the same command on every node. --batch-size stays the GLOBAL batch; each
# rank takes batch_size / (NNODES * NPROC).

NNODES="${NNODES:-1}"
if [ "${NNODES}" -gt 1 ]; then
  : "${NODE_RANK:?set NODE_RANK (0..NNODES-1) for a multi-node launch}"
  : "${MASTER_ADDR:?set MASTER_ADDR (address of node 0) for a multi-node launch}"
  TORCHRUN_LAUNCH=(--nnodes="${NNODES}" --node_rank="${NODE_RANK}"
                   --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT:-29500}")
else
  TORCHRUN_LAUNCH=(--standalone)
fi
