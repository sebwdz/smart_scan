#!/usr/bin/env bash

tensorflowjs_converter     --input_format=tf_saved_model     --output_node_names='num_detections,detection_boxes' --saved_model_tags=serve     ./frozen/saved_model/     ./web_model
