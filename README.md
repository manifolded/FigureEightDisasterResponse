# FigureEightDisasterResponse

Develop an ML pipeline to triage the incoming text messages and social media posts that deluge agencies during any major disaster.

![](https://github.com/manifolded/FigureEightDisasterResponse/blob/master/Images/26719.jpg?raw=true)
<a href='https://www.freepik.com/vectors/abstract'>Abstract vector created by macrovector - www.freepik.com</a>

## Motivation

Figure Eight has provided this data, collected at several natural disasters, as a challenge.   They hope that with this data we can build a useful model.  The messages have been translated into English and meticulously labeled in 36 categories like 'missing_people', 'search_and_rescue', 'medical_products', etc.  The ultimate goal is automatic categorization of such messages during future disasters.

Immediately after a disaster aid agencies are simultaneously deluged with messages like these while their staff are naturally deluged with duties.  A few of these messages are critical, but which ones?  If a machine could label these messages automatically at least they could be directed to the right agency if not sorted according to their urgency.  This is the machine we are tasked with inventing.

The final product is to be presented through a web app provided by Udacity.

## Trying Out The Code

There's nothing to install yet.  However, you are welcome to have a look at my Jupyter dev notebooks and see what is working and what is not.  

The `data` directory contains code related to ingesting the dataset provided by Figure Eight and consolidating it in a SQLite database.  Have a look at 'ETL Pipeline Preparation.ipynb'.  Current work is focused developing the ML pipeline and you can have a look at my progress by examining the 'ML Pipeline Preparation.ipynb' notebook.

## Acknowledgements

Thanks for Figure Eight for providing me the data, and Udacity to introducing me to the techniques.  Cheers.
