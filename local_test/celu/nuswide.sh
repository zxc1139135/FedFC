echo 'nuswide quant'

python main_pipeline_celu.py --configs celu/nuswide

# sed -i 's/"iteration_per_aggregation": 1/"iteration_per_aggregation": 5/g' ./configs/celu/nuswide.json
# sed -i 's/"lr": 0.01/"lr": 0.005/g' ./configs/celu/nuswide.json

# python main_pipeline_celu.py --configs celu/nuswide

# sed -i 's/"iteration_per_aggregation": 5/"iteration_per_aggregation": 1/g' ./configs/celu/nuswide.json
# sed -i 's/"lr": 0.005/"lr": 0.01/g' ./configs/celu/nuswide.json