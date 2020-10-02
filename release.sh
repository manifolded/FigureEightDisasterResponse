#!/bin/bash -

# python -c "import nltk;nltk.download('snowball_data')"
python -c "import nltk; nltk.download('popular')"

pushd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
popd

pushd models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
