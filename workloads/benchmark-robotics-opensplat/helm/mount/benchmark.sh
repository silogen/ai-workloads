#!/bin/bash

# The source for this dataset is available at:
# https://github.com/pierotofy/OpenSplat?tab=readme-ov-file#run

# time opensplat ./banana -n 2000/5000/10000/30000
# time opensplat ./banana -n 2000/5000/10000/30000
# time opensplat ./banana -n 2000/5000/10000/30000
# time opensplat ./banana -n 2000/5000/10000/30000
# time opensplat ./truck -n 2000/5000/10000/30000
# time simple_trainer --width 640 --height 360 --iters 2000/5000/10000/30000 --points 2000/10000/100000/1000000
# time simple_trainer --width 1280 --height 720 --iters 2000/5000/10000/30000 --points 2000/10000/100000/1000000
# time simple_trainer --width 1920 --height 1080 --iters 2000/5000/10000/30000 --points 2000/10000/100000/1000000
# time simple_trainer --width 3840 --height 2160 --iters 2000/5000/10000/30000 --points 2000/10000/100000/1000000

set -e  # Exit on any error

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <opensplat_path>"
  exit 1
fi
opensplat_path="$1"

echo "Benchmark Banana"
for itern_num in 2000 5000 10000 30000
do
  echo "itern_num: $itern_num"
  time $opensplat_path/opensplat ./banana -n $itern_num
done

echo "Benchmark Truck"
for itern_num in 2000 5000 10000 30000
do
  echo "itern_num: $itern_num"
  time $opensplat_path/opensplat ./truck -n $itern_num
done

echo "Benchmark the rasterization process on a set of random gaussians on a single training image with size (640, 360)"
for itern_num in 2000 5000 10000 30000
do
    echo "itern_num: $itern_num"
    for num_points in 2000 10000 100000 1000000
    do
        echo "num_points: $num_points"
        time $opensplat_path/simple_trainer --width 640 --height 360 --iters $itern_num --points $num_points
    done
done

echo "Benchmark the rasterization process on a set of random gaussians on a single training image with size (1280, 720)"
for itern_num in 2000 5000 10000 30000
do
    echo "itern_num: $itern_num"
    for num_points in 2000 10000 100000 1000000
    do
        echo "num_points: $num_points"
        time $opensplat_path/simple_trainer --width 1280 --height 720 --iters $itern_num --points $num_points
    done
done

echo "Benchmark the rasterization process on a set of random gaussians on a single training image with size (1920, 1080)"
for itern_num in 2000 5000 10000 30000
do
    echo "itern_num: $itern_num"
    for num_points in 2000 10000 100000 1000000
    do
        echo "num_points: $num_points"
        time $opensplat_path/simple_trainer --width 1920 --height 1080 --iters $itern_num --points $num_points
    done
done

echo "Benchmark the rasterization process on a set of random gaussians on a single training image with size (3840, 2160)"
for itern_num in 2000 5000 10000 30000
do
    echo "itern_num: $itern_num"
    for num_points in 2000 10000 100000 1000000
    do
        echo "num_points: $num_points"
        time $opensplat_path/simple_trainer --width 3840 --height 2160 --iters $itern_num --points $num_points
    done
done
