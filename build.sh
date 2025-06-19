#!/bin/bash
# build.sh
python -m pip install --upgrade pip
pip install setuptools==65.5.0 wheel==0.38.4
pip install -r requirements.txt