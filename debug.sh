# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
CUDA_VISIBLE_DEVICES=7 python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 60400 )) \
	--num-gpus 1 \
	--config configs/pascal-person-part/parsing/maskformer2_R50_bs16_13k.yaml \
	OUTPUT_DIR training_dir/debug \
	SOLVER.IMS_PER_BATCH 1 \
	DATALOADER.NUM_WORKERS 1 \

