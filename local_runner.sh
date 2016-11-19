#!/bin/bash
PYSPARK_PYTHON=python3.5 \
spark-submit \
    --master local[6] \
    --name "tsne-opt" \
    --driver-memory 2g \
    --executor-memory 12g \
    param_opt.py \
    csv-path-here-
    2 \
    pickle-path-here-
