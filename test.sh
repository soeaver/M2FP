# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.

#export GLOO_SOCKET_IFNAME=eth0
# CUDA_VISIBLE_DEVICES=7
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 100 + RANDOM % 10 + 40000 )) \
	--num-gpus 8 \
	--config configs/pascal-person-part/m2fp_R101_bs16_16k.yaml \
	--eval-only \
	MODEL.WEIGHTS training_dir/pascal-person-part/model_final.pth

