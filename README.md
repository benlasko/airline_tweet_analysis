# **Airline Tweet Sentiment Analysis**

![emojis_head](/images/emojishead.png)


<br>

## Technologies Used
* Python 
* NumPy
* Pandas
* Scikit-Learn
* Natural Language Tookit - NLTK
* Matplotlib
* Seaborn
* Jupyter Notebook
* AWS



## Introduction

#### Natural language processing is a profound tool with the power to help us understand our collective history, culture, and state of mind.  NLP can be used to conduct sentiment analysis which can be used for a variety of purposes including assessing public opinions, appraisals, emotions, or attitudes towards a topic, person or entity, predicting social unrest, and preventing violence.  Although the dataset I used for this project was a collection of tweets about airlines my interest in doing this was to learn and practice using NLP for sentiment analysis to be able to apply these techniques in a variety of ways.  To do this I wrote code and built functions to perform exploratory data analysis, text cleaning, text analysis, predictive modeling, cross validation, and supervised/unsupervised learning techniques.


## Summary

####  The dataset I used was from Kaggle and it had a collection of about 14,000 tweets from February of 2015 about six major airlines.  The tweets were prelabeled  as *positive*, *negative*, or *neutral*.  I conducted exploratory data analysis and plotted the distribution of all sentiment counts and the distribution of sentiment counts by airline.  Then I created a word cloud to visualize the most common words in the corpus of tweets.  After that I used natural language processing techniques to analyze the tweets.  The tweets were cleaned using a text cleaning pipeline I built.  I show some examples of the original vs cleaned tweets and some information about the lexical diversity of the corpus.  I created a second word cloud to show the most common words found in the cleaned corpus.  Following that I printed out some examples of the context in which two of the most common words in the corpus were found.
#### After cleaning the text I tested five supervised learning models to classify the tweets by sentiment.  You'll see the scores I was able to get with my models after some tuning.  I wanted to try an unsupervised learning approach to analyze the data so I also used TF-IDF, PCA, and k-means clustering to begin that process.

<br>

## Exploratory Data Analysis


####  The tweets were classified into *positive*, *negative*, and *neutral* sentiments. The distribution was 63% *negative*, 21% *neutral*, and 16% *positive*.

<br>

![Sentiment-Counts](/images/sent-counts.png)

<br>

![Sentiment-Counts-Airline](/images/sent-counts-airline.png)

<br>

## Natural Language Processing
#### This wordcloud shows the most common words from all the tweets before the text was cleaned and analyzed.

![Unclean-Corp-WC](/images/unclean-corp-wc.png)

<br>

#### To be able to analyze and process the tweets I made a text cleaning pipeline that does the following:
* Lowercases
* Removes puncuation
* Removes numbers
* Removes new lines
* Removes URLs
* Removes stopwords
* Lemmatizes 

#### So text like this:
1. @USAirways 2 hours and counting waiting to get into a gate in Philadelphia. Just icing on the cake for a miserable flight experience 
2. @VirginAmerica why are your first fares in May over three times more than other carriers when all seats are available to select???
3. To be or not to be?

#### Becomes:
1. hour counting waiting get gate philadelphia icing cake miserable experience
2. first fare may time carrier seat available select 
3. ''

#### Line 3 is a good example of why natural language processing can be difficult.

<br>

#### The cleaned corpus had about 18,000 unique words and the lexical diversity of the corpus was somewhat limited.
 
 * Total words: 129,374
 * Total unique words: 18,350
 * Proportion of unique words: 0.14

#### The wordcloud and graph below show the most common words from all the tweets after going through the cleaning process.

![Clean-WC](/images/clean-corp-wc.png)

![Most-Common-Words](/images/common-words.png)

<br>

#### Examples of the context in which two of the most common words were found:
##### "thank"

![thank-context](/images/thank-context.png)

##### "service"

![service-context](/images/service-context.png)

<br>

## Supervised Learning Classification Models

#### I used supervised learning classification models to try and build a good predictive model to classify tweets into their *positive*, *negative*, or *neutral* classes.

##### Models tested:
* Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Machine
* Multi Layer Perceptron neural network

#### By tuning the models and making changes to my stopwords list I was able to raise the MLPClassifier accuracy about 7% but other models performed relatively similarly to when they ran using the default parameters.  SVM, Random Forest, and MLP models performed better than the others.

##### Model scores after some tuning:

![Some-Tuning](/images/some-tuning.png)

#### The model with the highest f1 score, 77.46%, was the SVC(C=3) model.

<br>

## Unsupervised Learning Techniques

#### Although the dataset had the tweets prelabeled I wanted to try out using some unsupervised learning techniques to analyze the corpus.  I used a TF-IDF vectorizer to create a term frequency matrix of the entire corpus, used PCA to shrink the matrix from over 13,000 features to 100 features, and used k-means clustering to form three clusters to begin that process.

<br>

## Next Steps
#### Next steps for this project include using transformer models like XLNet, GPT-3, or BERT and continuing to explore the data with unsupervised learning techniques.


<br>
