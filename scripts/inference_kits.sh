#!/bin/bash
export MPLCONFIGDIR=/tmp/matplotlib

python ./inference_kits.py --images_path $1 --labels_path $2 --output_path $3 --model_path /radraid/blchau/MedSAM/scripts/medsam_vit_b.pth