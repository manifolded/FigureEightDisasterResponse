#!/bin/bash -

pushd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
popd

pushd models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
popd

# pushd app
# python run.py
# popd
