#!/bin/bash -

python -c "import nltk;nltk.download('snowball_data')"

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

