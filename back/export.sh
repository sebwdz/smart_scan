#!/usr/bin/env bash

python3 ~/tensorflow_models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/model/pipeline.config \
    --trained_checkpoint_prefix models/model/model.ckpt-1450040 \
    --output_directory frozen