#!/bin/bash

lawschool=(lawschool 0.1 "37 20" "20 37" 37_20_37)

for parameters in lawschool; do
  declare -n params="$parameters"

  dataset=${params[0]}
  dl2_weight=${params[1]}
  encoder_layers=${params[2]}
  decoder_layers=${params[3]}
  layers=${params[4]}

  # baseline
  python train_encoder.py --dataset ${dataset} --load --quantiles \
    --encoder-layers ${encoder_layers} \
    --decoder-layers ${decoder_layers} \
    --constraint "GeneralCategorical(0.01, 0.3, [])" \
    --dl2-weight 0

  # LCIFR data producer
  python train_encoder.py --dataset ${dataset} --load --quantiles \
    --encoder-layers ${encoder_layers} \
    --decoder-layers ${decoder_layers} \
    --constraint "GeneralCategorical(0.01, 0.3, [])" \
    --dl2-weight ${dl2_weight}

  # LCIFR data consumer
  python train_classifier.py --dataset ${dataset} --load --quantiles \
    --encoder-layers ${encoder_layers} \
    --decoder-layers ${decoder_layers} \
    --constraint "GeneralCategorical(0.01, 0.3, [])" \
    --dl2-weight ${dl2_weight} --load-epoch 99 --adversarial --delta 0.01

  # certify baseline
  python certify.py --quantiles \
    --models-dir "../../models/${dataset}/GeneralCategorical(0.01, 0.3, [])/${layers}/dl2_weight_0.0_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_True_dec_weight_0.0" \
    --num-certify 1000 --complete --epoch 99 --load

  # certify LCIFR
  python certify.py --quantiles \
    --models-dir "../../models/${dataset}/GeneralCategorical(0.01, 0.3, [])/${layers}/dl2_weight_${dl2_weight}_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_True_dec_weight_0.0" \
    --num-certify 1000 --complete --epoch 99 --load --adversarial
done
