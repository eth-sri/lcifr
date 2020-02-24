#!/bin/bash

adult=(adult 10.0 "104 20 20" "20 20 104" 0.2 0.4 "'age'" 37 104_20_20_20_104)
german=(german 10.0 "59 20 20" "20 20 59" 0.2 0.4 "'age'" 33 59_20_20_20_59)
lawschool=(lawschool 0.1 "37 20" "20 37" 0.4 0.2 "'gpa'" 3.4 37_20_37)

for parameters in adult german lawschool; do
  declare -n params="$parameters"

  dataset=${params[0]}
  dl2_weight=${params[1]}
  encoder_layers=${params[2]}
  decoder_layers=${params[3]}
  below_threshold=${params[4]}
  above_threshold=${params[5]}
  attribute=${params[6]}
  attribute_value=${params[7]}
  layers=${params[8]}

  # baseline
  python train_encoder.py --dataset ${dataset} --load \
    --encoder-layers ${encoder_layers} \
    --decoder-layers ${decoder_layers} \
    --constraint "AttributeConditional(0.01, ${below_threshold}, ${above_threshold}, ${attribute}, ${attribute_value})" \
    --dl2-weight 0

  # LCIFR data producer
  python train_encoder.py --dataset ${dataset} --load \
    --encoder-layers ${encoder_layers} \
    --decoder-layers ${decoder_layers} \
    --constraint "AttributeConditional(0.01, ${below_threshold}, ${above_threshold}, ${attribute}, ${attribute_value})" \
    --dl2-weight ${dl2_weight}

  # LCIFR data consumer
  python train_classifier.py --dataset ${dataset} --load \
    --encoder-layers ${encoder_layers} \
    --decoder-layers ${decoder_layers} \
    --constraint "AttributeConditional(0.01, ${below_threshold}, ${above_threshold}, ${attribute}, ${attribute_value})" \
    --dl2-weight ${dl2_weight} --load-epoch 99 --adversarial --delta 0.01

  # certify baseline
  python certify.py \
    --models-dir "../../models/${dataset}/AttributeConditional(0.01, ${below_threshold}, ${above_threshold}, ${attribute}, ${attribute_value})/${layers}/dl2_weight_0.0_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_0.0" \
    --num-certify 1000 --complete --epoch 99 --load

  # certify LCIFR
  python certify.py \
    --models-dir "../../models/${dataset}/AttributeConditional(0.01, ${below_threshold}, ${above_threshold}, ${attribute}, ${attribute_value})/${layers}/dl2_weight_${dl2_weight}_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_0.0" \
    --num-certify 1000 --complete --epoch 99 --load --adversarial
done
