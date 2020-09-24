# FigureEightDisasterResponse

Develop an ML pipeline to triage the deluge of incoming text messages and social media posts during a typical major disaster.

## Motivation

Figure Eight has provided this data from several natural disasters as a challenge in the hopes that we can build a useful model from it.  The messages have been translated into English, as needed, and meticulously labeled in 36 categories like 'missing_people', 'search_and_rescue', 'medical_products', etc.  The goal is to predict how new messages would be categorized with a machine learning pipeline.

Immediately after a disaster aid agencies are simultaneously deluged with messages exactly when their staff are already overwhelmed.  A few of these messages are critical while most are not so helpful.  If a machine could label these messages automatically at least they could be directed to the right agency if not sorted by their importance.  This is the machine we are tasked with inventing.

The final product is to be presented through a web app we will construct.

## Trying Out The Code

There's nothing to install yet.  However, you are welcome to have a look at my Jupyter dev notebooks and see what is working and what is not.  

The `data` directory contains code related to ingesting the dataset provided by Figure Eight and consolidating it in a SQLlite database.  Have a look at 'ETL Pipeline Preparation.ipynb'.  Current work is focused developing the ML pipeline and you can have a look at my progress by examining the 'ML Pipeline Preparation.ipynb' notebook.

