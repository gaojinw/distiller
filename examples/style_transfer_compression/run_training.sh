#!/bin/sh
#MODEL=${MODEL_FULLNAME%.*}

[ ! $# -ge 4 ] && { echo "Usage: $0 STYLE config_yaml CUDA style_weights optional: epochs resume lr"; exit 1; }

STYLENAME=$1
CONFIG=$2
GPUID=$3
STYLEWEIGHT=$4
if [ ! -z $5 ]; then
    EPOCHS=$5
else
    EPOCHS="4"
fi
if [ ! -z $6 ]; then
    RESUMEMODEL=$6
else
    RESUMEMODEL="None"
fi
if [ ! -z $7 ]; then
    LR=$7
else
    LR="1e-4"
fi

# CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
# MODEL=${MODEL_FULLNAME%.*}

# CODE_SPACE="/host/frameworks/examples/fast_neural_style"
# RUN_SCRIPT="$CODE_SPACE/neural_style/neural_style.py"
RUN_SCRIPT="./compress_style_transfer.py"
STYLE_PICTURE="./pretrained/images/style_images/$STYLENAME.jpg"

# INTERMEDIATE_STYLE_FOLDER="$CURRENT_DIR/training/$STYLENAME"
#if [ ! -f "$RESUMEMODEL" ]; then
#    WORKSPACE="$INTERMEDIATE_STYLE_FOLDER/${CONFIG}_${STYLEWEIGHT}"
#else
#    WORKSPACE="$INTERMEDIATE_STYLE_FOLDER/${CONFIG}_${STYLEWEIGHT}_RESUME_${LR}"
#fi
#echo "Arguments: $CODE_SPACE $RESUMEMODEL $WORKSPACE\n"

########## configure the correct code base ###########
#cd $CODE_SPACE && git checkout $CONFIG

########## set up work space ###########
#if [ ! -d "$INTERMEDIATE_STYLE_FOLDER" ]; then
#    echo "New Style. Making directory $INTERMEDIATE_STYLE_FOLDER"
#    mkdir "$INTERMEDIATE_STYLE_FOLDER"
#fi
#if [ ! -d "$WORKSPACE" ]; then
#    echo "New Workspace. Making directory $WORKSPACE"
#    mkdir "$WORKSPACE"
#fi

########## check if style image exists  ###########
if [ ! -f "$STYLE_PICTURE" ]; then
    echo "\nStyle image $STYLE_PICTURE not found!\n"
    exit 1
fi

# log_file="$WORKSPACE/run.log"
# touch $log_file
# starttime=$(date +%Y-%m-%d\ %H:%M:%S)
# echo "========================== $starttime ========================" | tee -a $log_file
# echo "Code Space:  $CODE_SPACE" | tee -a $log_file
# echo "Code Branch:  $CONFIG" | tee -a $log_file
# echo "Style:  $STYLENAME" | tee -a $log_file
# echo "Style Weights:  $STYLEWEIGHT" | tee -a $log_file
# echo "CUDA:  $GPUID" | tee -a $log_file
# echo "RESUME:  $RESUMEMODEL" | tee -a $log_file
# echo "Learning Rate:  $LR" | tee -a $log_file


nohup python3 $RUN_SCRIPT \
--dataset /host/dataset/COCO/ \
--name ${CONFIG%.*} \
--epochs ${EPOCHS} \
--batch-size 8 \
--out-dir ./logs \
--print-freq 500 \
--image-size 256 \
--pretrained ./pretrained/models/style4_transfer.model \
--content-weight 1e5 \
--style-weight $STYLEWEIGHT \
--style-image $STYLE_PICTURE \
--compress $CONFIG \
--num-best-scores 1 \
--lr $LR \
--cuda $GPUID &
