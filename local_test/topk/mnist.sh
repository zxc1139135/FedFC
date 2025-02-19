#!/bin/bash
#SBATCH --job-name dp0_main             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完


# GaussianDP #SBATCH --qos high
# Begin with GaussianDP 0.0001
echo 'mnist topk'

### 0.8
python main_pipeline_topk.py --configs topk/mnist

sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 5/g' ./configs/topk/mnist.json
sed -i 's/"lr": 0.01/"lr": 0.005/g' ./configs/topk/mnist.json

python main_pipeline_topk.py --configs topk/mnist

sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 1/g' ./configs/topk/mnist.json
sed -i 's/"lr": 0.005/"lr": 0.01/g' ./configs/topk/mnist.json


#### 0.5 
sed -i 's/"ratio": 0.8/"ratio": 0.7/g' ./configs/topk/mnist.json

python main_pipeline_topk.py --configs topk/mnist

sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 5/g' ./configs/topk/mnist.json
sed -i 's/"lr": 0.01/"lr": 0.005/g' ./configs/topk/mnist.json

python main_pipeline_topk.py --configs topk/mnist

sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 1/g' ./configs/topk/mnist.json
sed -i 's/"lr": 0.005/"lr": 0.01/g' ./configs/topk/mnist.json


sed -i 's/"ratio": 0.7/"ratio": 0.8/g' ./configs/topk/mnist.json
