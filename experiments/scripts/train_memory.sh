#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
NET_ARCH=$4

OIFS=$IFS
IFS='a'
STEP="$5"
STEPSIZE="["
for i in $STEP; do
  STEPSIZE=${STEPSIZE}"${i}0000,"
done
STEPSIZE=${STEPSIZE}"]"
IFS=$OIFS

ITERS=${6}0000
EXTRA_TAG=$7

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:7:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    VAL_IMDB="coco_2014_minival"
    ;;
  vg)
    TRAIN_IMDB="visual_genome_train_rel"
    VAL_IMDB="visual_genome_val_rel"
    ;;
  ade)
    TRAIN_IMDB="ade_train_5"
    VAL_IMDB="ade_mval_5"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  EXTRA_ARGS_SLUG=${EXTRA_ARGS_SLUG}_${5}_${6}_${7}
else
  EXTRA_ARGS_SLUG=${5}_${6}_${7}
fi

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_iter_${ITERS}.ckpt
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_memory.py \
    --weight data/frcnn_weights/vgg16_vg_35-49k/vgg16_iter_900000.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${VAL_IMDB} \
    --iters ${ITERS} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net_base ${NET} \
    --net_arch ${NET_ARCH} \
    --extra_tag ${EXTRA_TAG} \
    --set TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
fi

#./experiments/scripts/test_memory.sh $@

