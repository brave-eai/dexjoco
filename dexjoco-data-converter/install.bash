#!/usr/bin/env bash

conda env create -f ./dexjoco-data-converter/environment-dc.yaml
conda run --no-capture-output -n dexjoco-dc git submodule update --init --recursive
conda run --no-capture-output -n dexjoco-dc python -m pip install -e ./dexjoco-data-converter



