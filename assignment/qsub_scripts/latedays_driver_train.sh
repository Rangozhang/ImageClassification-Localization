#!/bin/bash
# hacked up by xiaolong, xiaolonw@cs.cmu.edu
# THIS IS A DRIVER THAT SHOULD ONLY BE CALLED FROM latedays_starter.sh
# DO REMEMBER TO MOVE THE DIRECTORIES BEFORE USE!
# Feb 6th, 2016

if [ ! -n "$PROCSTRING" ]
    then
    echo "PROCSTRING UNDEFINED, not running anything"
    exit
fi


OUTPUT_FILER=/home/${USER}/STOutputs/python.${PROCSTRING}.${HOSTNAME}.$$.output
echo ${OUTPUT_FILER}
touch ${OUTPUT_FILER}

cd /home/xiaolonw/cnncode/caffe-coco/coco_scripts/py_test

echo $$
nice python  "${PROCSTRING}" &> $OUTPUT_FILER

echo "You get 200% accuracy!"


