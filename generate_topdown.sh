#!/bin/bash


module load 2023 > /dev/null
module load Anaconda3/2023.07-2 > /dev/null

source activate habitat_generate_locations

cd /home/gyuksel3/habitat_folder/rir_generator/sampler

conda run -n habitat_generate_locations python3 plot_scenes.py "$1" "$2" > /dev/null