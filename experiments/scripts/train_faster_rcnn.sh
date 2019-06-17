#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

OIFS=$IFS
IFS='a'
STEP="$4"
STEPSIZE="["
for i in $STEP; do
  STEPSIZE=${STEPSIZE}"${i}0000,"
done
STEPSIZE=${STEPSIZE}"]"
IFS=$OIFS

ITERS=${5}0000
EXTRA_TAG=$6

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  vg)
    TRAIN_IMDB="visual_genome_train_det"
    VAL_IMDB="visual_genome_val_det"
    ANCHORS="[2.22152954,4.12315647,7.21692515,12.60263013,22.7102731]"
    RATIOS="[0.23232838,0.63365731,1.28478321,3.15089189]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  EXTRA_ARGS_SLUG=${EXTRA_ARGS_SLUG}_${4}_${5}_${6}
else
  EXTRA_ARGS_SLUG=${4}_${5}_${6}
fi

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_iter_${ITERS}.ckpt
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_net.py \
    --weight data/imagenet_weights/${NET}.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${VAL_IMDB} \
    --iters ${ITERS} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
    TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
fi

#./experiments/scripts/test_faster_rcnn.sh $@
