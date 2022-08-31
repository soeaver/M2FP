# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 20100 )) \
	--num-gpus 8 \
	--config configs/pascal-person-part/parsing/maskformer2_R50_bs16_13k.yaml \
	OUTPUT_DIR training_dir/pascal-person-part \
	WANDB.ENABLED True WANDB.ENTITY soeaver WANDB.NAME m2fp-ppp-parsing-r50_test1 \
	# WANDB.ENABLED True WANDB.ENTITY soeaver WANDB.NAME m2fp-ppp-seg-r101-lr2e-4_test1 \