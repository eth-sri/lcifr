#!/bin/bash

dataset=health
dl2_weight=10.0
dec_weight=10.0
encoder_layers="64 20 20"
decoder_layers="20 20 64"
layers=64_20_20_20_64

# LCIFR data producer with decoder
python train_encoder.py --dataset health --load \
  --encoder-layers "${encoder_layers}" \
  --decoder-layers "${decoder_layers}" \
  --constraint "GeneralCategorical(0.01, 0.3, ['AgeAtFirstClaim', 'Sex'])" \
  --dl2-weight ${dl2_weight} --dec-weight ${dec_weight} --transfer

# LCIFR data consumer original task
python train_classifier.py --dataset health --load \
  --encoder-layers "${encoder_layers}" \
  --decoder-layers "${decoder_layers}" \
  --constraint "GeneralCategorical(0.01, 0.3, ['AgeAtFirstClaim', 'Sex'])" \
  --dl2-weight ${dl2_weight} --dec-weight ${dec_weight} --transfer \
  --load-epoch 99 --adversarial --delta 0.01

# certify original task
python certify.py --transfer \
  --models-dir "../../models/health/GeneralCategorical(0.01, 0.3, ['AgeAtFirstClaim', 'Sex'])/${layers}/dl2_weight_${dl2_weight}_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_${dec_weight}" \
  --num-certify 1000 --complete --epoch 99 --load --adversarial

for label in MSC2a3 METAB3 ARTHSPIN NEUMENT RESPR4; do
  # LCIFR data consumer transfer task
  python train_classifier.py --dataset health --load \
    --encoder-layers "${encoder_layers}" \
    --decoder-layers "${decoder_layers}" \
    --constraint "GeneralCategorical(0.01, 0.3, ['AgeAtFirstClaim', 'Sex'])" \
    --dl2-weight ${dl2_weight} --dec-weight ${dec_weight} --transfer \
    --load-epoch 99 --adversarial --delta 0.01 \
    --label "PrimaryConditionGroup=${label}"

  # certify transfer task
  python certify.py --transfer \
    --models-dir "../../models/health/GeneralCategorical(0.01, 0.3, ['AgeAtFirstClaim', 'Sex'])/${layers}/dl2_weight_${dl2_weight}_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_${dec_weight}" \
    --num-certify 1000 --complete --epoch 99 --load --adversarial \
    --label "PrimaryConditionGroup=${label}"
done
