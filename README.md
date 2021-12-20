
# Classifying NLP Data for the Service Industry: Comparing 2 Subreddits

By: Terri John


This ReadMe contains:

* [Project Contents](#contents)
* [Problem Statement](#problemstatement)
* [Background](#background)
* [A description of the data / Data Dictionary](#data)
* [Modeling and Analysis](#model)
* [Sentiment Analysis](#sentiment)
* [Conclusion](#conclusion)
* [Sources](#sources)



## <a name="contents"></a>Project Contents:


This projects contains the following:

|Name|File Type|Description|
|---|---|---|
|1.intro_and_webscrape|Jupyter Notebook|Provides an introduction to the project, including problem statement and background, and the code used for gathering the data.|
|2.eda|Jupyter Notebook|Displays data cleaning and exploratory data analysis.|
|3.modeling|Jupyter Notebook|Builds 4 classification models.|
|4.sentiment_analysis|Jupyter Notebook|Explores sentiment analysis of the language used in each subreddit|
|README.md|txt|An executive summary of the project.|
|models|folder|contains saved copies of the models produced|
|data|folder|contains csv files of data used|

## <a name="problemstatement"></a>Problem Statement

During the past year, the service industry has seen an unprecedented rate of resignations.  I wanted to develop a classification  model that could predict if written language originates from a service worker in the restaurant industry or in the retail industry.  I wanted to see which language was common to both industries, and which language differentiates the two, as well as the sentiment of that language.  By examining these similarities and differences, businesses can better understand what steps they need to take in order to retain workers, as well as to attract new employees.

## <a name="background"></a>Background

Wages for workers in the service industry have remained stagnant for years.  While retail and serving work schedules have tended to fluctuate in general (as anyone who has worked in retail or serving knows) this is something that was made worse by the COVID-19 pandemic when demand for stores and restaurants shrank and many businesses were temporarily closed.  Many workers lost significant income due to hours being cut, hours that they relied on for supporting themselves and in some cases their families. One reason often mentioned is the expanded jobless benefits that came out of the COVID-19 pandemic, which may have reduced the incentive to accept jobs. But states that canceled those benefits early on saw no increase in employment compared with those that didn’t.

Through this study I hoped to gain a better idea of what is actually motivating workers to resign, and therefore help businesses to better understand what changes they can make to retain their workforce.

Source: https://www.nytimes.com/2021/10/14/opinion/workers-quitting-wages.html?searchResultPosition=3

## <a name="data"></a>Data:

To gather the data for this study, I used Pushshift’s API to collect the most recent 180,000 comments from 2 subreddits: TalesFromRetail and TalesFromYourServer.  That large of a dataset proved impractical for my modeling purposes due to slow fit times, and I shrank each data set to the 40,000 most recent comments for each subreddits. Tales from Retail is a subreddit where retail workers share their experiences, and Tales from your server is the equivalent for servers. Both subreddits have been around for about 10 years.

TalesFromRetail:     
* https://www.reddit.com/r/TalesFromRetail/
* Members 645,000
* Created November 9, 2011
* “A place to exchange stories about your daily experiences in brick & mortar retail.”

TalesFromYourServer:      
* https://www.reddit.com/r/TalesFromYourServer/
* Members: 444,000
* Created September 24, 2012
* A subreddit where servers share stories and advice.


Pushshift API:
* https://github.com/pushshift/api

### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|body|str|stop_words_removed|The text of the comment.|
|subreddit|str|stop_words_removed|The title of the subreddit.
|author|str|stop_words_removed|The username of the comment author.
|created_utc|int|stop_words_removed|The Universal Time Coordinated (UTC) time when the comment was posted.
|comment_length|int|stop_words_removed|The wordcount of the comment.
|comment_tokens|str|stop_words_removed|Cleaned copy of body column following EDA process.

## <a name="model"></a>Modeling and Analysis
I created 4 main models during the course of this project.
* [SELECTED MODEL: LogisticRegression](#logreg_selected)
* [Baseline model](#baseline)
* [Logistic Regression Model #1 with TF IDF Vectorizer](#logreg1)
* [RandomForest](#randomforest)
* [MultinomialNB](#mnb)


### <a name="baseline"></a>  Baseline Model

My baseline model has an accuracy score of 0.53 - it will accurately predict the correct subreddict 53% of the time, because that is the size of the majority class in my dataset.

### <a name="logreg_selected"></a>Selected Model: LogisticRegression with TF IDF Vectorizer

For my fourth and final model I tested a simplied LogisticRegression model. I created a pipeline to for my transformer (TfidfVectorizer) and estimator (LogisticRegression).

Instead of using a gridsearch for this model, I took what I'd learned in earlier model attempts and kept most settings as their defaults, but landed an alpha regularization setting of .15 after some testing.

With this C setting, I was able to get a well fit model with an accuracy score of 0.78.

### <a name="logreg1"></a>Logistic Regression Model #1 with TF IDF Vectorizer

For my first model in this notebook, I began with a Logistic Regression model. I created a pipeline for my transformer (TfidfVectorizer) and estimator (LogisticRegression). I preferred TfidfVectorizer for modeling in this study because I wanted to make sure to focus on words that were strongly related to one subreddit or the other, while putting less emphasis on the words that were common in both subreddits.

I used a gridsearch to test the following features in this model:
* Max Features:  500, 1000, 2000 : to limit the number of features selected by the transformer.
* ngram_range: to limit the transformer to ngrams and bigrams.
* max_df: to ignore words that had a frequency above 90% versus the default of 100%.
* Logistic Regression 'C' hyperparameter to set the alpha for regularization to .0001, .01, and 1.0.

This model was not over or underfit, but the overall accuracy was low at only 60%.  I used the information gained from gridsearching over these hyperparameters to build a stronger Logistic Regression model at the bottom of this notebook, which turned out to be my strongest model overall.

### <a name="randomforest"></a>RandomForest with TF IDF Vectorizer

For my second model in this notebook, I tested a RandomForest model. I created a pipeline for my transformer (TfidfVectorizer) and estimator (RandomForest). I preferred TfidfVectorizer for modeling in this study because I wanted to make sure to focus on words that were strongly related to one subreddit or the other, while putting less emphasis on the words that were common in both subreddits.

I used a gridsearch to test the following features in this model:
* Max Features:  500, 1000, 2000, 3000 : to limit the number of features selected by the transformer.
* ngram_range: to limit the transformer to ngrams and bigrams.
* max_df: to ignore words that had a frequency above 70% or 90% versus the default of 100%.
* RandomForest alpha hyperparameter to set the regularization to 0.0, .001, .01, and 1.0.

This model was drastically overfit, with a training accuracy score of .98 and a testing accuracy score of .74. I would have liked to have performed more hyperparameter testing for this model to correct the bias, but this particular model took a very long time to fit. Given how strong my final LogisticRegression model performed, I decided not to move forward with RandomForest for this classification model.

### <a name="mnb"></a> MultinomialNB with CountVectorizer

For my third model in this notebook, I tested a MultinomialNB model with a Countvectorizor transformer. I created a pipeline to for my transformer (CountVectorizer) and estimator (MultinomialNB).

I used a gridsearch to test out several hyperparameters for CountVectorizer, and to test various regularization settings for MultinomialNB.

Even with testing out several alpha settings, I continued to get an overfit model, with a training accuracy score of .88 and a testing accuracy score of .79.

The best parameters from this gridsearch were:

*  'cvec__max_df': 0.85,
*  'cvec__max_features': None,
*  'cvec__min_df': 2,
*  'cvec__ngram_range': (1, 2),
*  'cvec__stop_words': 'english',
*  'nb__alpha': 1.0

## <a name="sentiment"></a>Sentiment Analysis:

Moving on to sentiment analysis, I was interested to see that the majority of comments were neutral.  Both subreddits had extreme outliers of a perfect positive or negative score, but most comments fell between a -.25 score or a positive .5 score.



## <a name="conclusion"></a>Conclusion:

* The model is able to differentiate between the TalesFromYourServer and the TalesFromRetail subreddits with about 80% accuracy.
* Generally, the sentiment from both groups is fairly neutral.
* Based on top word usage, it would appear that things like tips, food, and tables are important to servers, while things like customers are important to both servers and retail workers.


### Recommendation for next steps:

Since the bulk of the comments were neutral in nature, I didn’t get a good feel for what workers are looking for that their employers may be able to provide.  Given more time, I would focus on breaking down comments to look at the 25% most negative and most positive.  This way we could get a better idea of negative and positive things that workers are discussing, while avoiding the noise of the bulk of the neutral comments

I would also recommend actually conducting a survey of server and retail workers to better gauge what matters most to them in a work environment in order to maximize employee retention and to attract new employees.  


### <a name="sources"></a>Sources

* The New York Times: https://www.nytimes.com/2021/10/14/opinion/workers-quitting-wages.html?searchResultPosition=3

* TalesFromRetail Subreddit: https://www.reddit.com/r/TalesFromRetail/

* TalesFromYourServer Subreddit: https://www.reddit.com/r/TalesFromYourServer/

* Pushshift API: https://github.com/pushshift/api

* source for TOC functionality: https://stackoverflow.com/questions/5319754/cross-reference-named-anchor-in-markdown/7335259#7335259

* source for sentiment analysis histogram code: https://medium.com/@arseniytyurin/how-to-make-your-histogram-shine-69e432be39ca
