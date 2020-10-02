#!/bin/bash -

# RUN python -c "import nltk;nltk.download('snowball_data')"
RUN python -c "import nltk; nltk.download('popular')"

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

