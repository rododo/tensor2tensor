#!/usr/bin/env bash

dir_cur="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
dir_bin=${dir_cur}/../bin
# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
PROBLEM=translate_zwzh_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu


DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

${dir_bin}/t2t-trainer --registry_help

if [ $1 == "train" ]; then
    # Generate data
    ${dir_bin}/t2t-datagen \
      --data_dir=$DATA_DIR \
      --tmp_dir=$TMP_DIR \
      --problem=$PROBLEM

    # Train
    # *  If you run out of memory, add --hparams='batch_size=1024'.
    ${dir_bin}/t2t-trainer \
      --data_dir=$DATA_DIR \
      --problem=$PROBLEM \
      --model=$MODEL \
      --hparams_set=$HPARAMS \
      --output_dir=$TRAIN_DIR
fi



if [ $1 == "predict" ]; then
    # Decode

    ts=$(date +%s)
    decode_file=$DATA_DIR/to_decode_${ts}.txt
    translated_file=$DATA_DIR/translated_${ts}.txt
    ref_file=$DATA_DIR/ref_${ts}.txt

    echo -e '你好世界\n再见世界' > $decode_file
    echo -e '你好世界\n再见世界' > $ref_file

    BEAM_SIZE=4
    ALPHA=0.6

    ${dir_bin}/t2t-decoder \
      --data_dir=$DATA_DIR \
      --problem=$PROBLEM \
      --model=$MODEL \
      --hparams_set=$HPARAMS \
      --output_dir=$TRAIN_DIR \
      --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
      --decode_from_file=$decode_file \
      --decode_to_file=${translated_file}

    # See the translations
    cat ${translated_file}

    # Evaluate the BLEU score
    # Note: Report this BLEU score in papers, not the internal approx_bleu metric.
    ${dir_bin}/t2t-bleu --translation=${translated_file} --reference=${ref_file}
fi



