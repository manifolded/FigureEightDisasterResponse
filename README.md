# FigureEightDisasterResponse

Goal: To develop an ML pipeline to triage the incoming text messages and social media posts that deluge agencies during any major disaster.

![](https://github.com/manifolded/FigureEightDisasterResponse/blob/master/Images/26719.jpg?raw=true)
<a href='https://www.freepik.com/vectors/abstract'>Abstract vector created by macrovector - www.freepik.com</a>

## Motivation

Figure Eight has provided this data, collected at several natural disasters, as a challenge.   They hope that with this data we can build a useful model.  The messages have been translated into English and meticulously labeled in 36 categories like 'missing_people', 'search_and_rescue', 'medical_products', etc.  The ultimate goal is automatic categorization of such messages during future disasters.

Immediately after a disaster aid agencies are simultaneously deluged with messages like these while their staff are naturally deluged with duties.  A few of these messages are critical, but which ones?  If a machine could label these messages automatically at least they could be directed to the right agency if not triaged according to their urgency.  This is the machine we are tasked with inventing.

The final product is presented via a flask app provided by Udacity.

## Trying Out The Code

There are three important directories in this repo, `data`, `models` and `app`.  In each of these directories you will need to run a script.  First, in the `data` directory,  execute

    python process_data.py disaster_messages.csv  disaster_categories.csv \
      DisasterResponse.db

This generates a database file containing the data from the two sources properly merged and cleaned.  Second, in the `models` directory, execute

    python train_classifier.py ../data/DisasterResponse.db classifier.pkl

This one takes several minutes to run.  (Longer than average because I also generate what I call a 'canon table' required by the second of the two plots.)  Next, in the `app` directory execute

    python run.py

This is fairly quick starting up since we have pre-computed everything in `train_classifier.py`.  You will need to leave it running while you examine the flask app as it starts up the server.  You can access the webpage by pointing your browser to http://0.0.0.0:3001

If you want to see how the above code was developed have a look at the series of `.ipynb` files contained in this repo (mostly in `models`.)  

## The Flask App

### Trying Out the Classifier

With the flask app in a browser window you can type in the text field near the top and click the button marked "Classify Message".  The results of your query will appear below.  You may have to scroll down in the window to see all the categories.  You will know when the classifier has selected one or more of these target categories because the category label will be highlighted.  

This model isn't a great success and more often than not only the first category, 'related', is indicated.  Also, longer messages work better than short ones.  

### Analysis of the Model and Dataset

After you run a query the plots will no longer be visible.  To bring them back click on the "Disaster Response Project" link in the upper left corner.

1. The first plot depicts the relationship between how good the model is at predicting against how many rows in the training set were positively labeled for that category.  Each data point is a single category.  The plot is an attempt to diagnose the limitations of the model.  It depicts the rise in score as we move to less imbalanced categories.

2. The second plot is very simple on the face of it, but has some hidden gems.  It depicts, as above, the number of positives in each category.  However, if you mouse over one of the bars a little hover blurb will appear.  These blurbs display the 'canon tokens' for that category.  These are the words (more accurately "stemmed tokens") that map to that category.  It is amusing to try and discern the link between the tokens and the category they correspond with.  Also note that it appears that the number of 'canon tokens' appears to correspond with the number of positives.

## Acknowledgements

Thanks for Figure Eight for providing me the data, and Udacity for introducing me to the techniques.  Cheers.
