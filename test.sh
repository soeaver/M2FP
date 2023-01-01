# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.

#export GLOO_SOCKET_IFNAME=eth0
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 100 + RANDOM % 10 + 40000 )) \
	--num-gpus 8 \
	--config configs/cihp/m2fp_R101_bs16_265k.yaml \
	--eval-only \
	MODEL.WEIGHTS weights/cihp/model_final.pth

