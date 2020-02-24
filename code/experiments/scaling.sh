#!/bin/bash

dataset=adult
dl2_weight=10.0
encoder_layers="104 200 200"
decoder_layers="104 200 200"
layers=104_200_200_200_104

# LCIFR data producer
python train_encoder.py --dataset ${dataset} --load \
  --encoder-layers "${encoder_layers}" \
  --decoder-layers "${decoder_layers}" \
  --constraint "GeneralCategorical(0.01, 0.3, [])" \
  --dl2-weight ${dl2_weight}

# LCIFR data consumer
python train_classifier.py --dataset ${dataset} --load \
  --encoder-layers "${encoder_layers}" \
  --decoder-layers "${decoder_layers}" \
  --constraint "GeneralCategorical(0.01, 0.3, [])" \
  --dl2-weight ${dl2_weight} --load-epoch 99 --adversarial --delta 0.01

# complete certification
python certify.py \
  --models-dir "../../models/${dataset}/GeneralCategorical(0.01, 0.3, [])/${layers}/dl2_weight_${dl2_weight}_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_0.0" \
  --num-certify 1000 --complete --epoch 99 --load --adversarial

# incomplete certification
python certify.py \
  --models-dir "../../models/${dataset}/GeneralCategorical(0.01, 0.3, [])/${layers}/dl2_weight_${dl2_weight}_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_0.0" \
  --num-certify 1000 --epoch 99 --load --adversarial
