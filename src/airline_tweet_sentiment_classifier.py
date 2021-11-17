'''
Sentiment analysis on airline tweets.  
Capstone 2 for the Galvanize Data Science Immersive.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.text import Text
import itertools
import re
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


'''
EDA
'''

def data_overview(df):
    '''
    Prints the following to get an overview of the data for starting EDA:
        First five rows (.head())
        Shape (.shape)
        All columns (.columns)
        Readout of how many non-null values and the dtype for each column (.info())
        Numerical column stats (.describe())
        Sum of unique value counts of each column
        Total null values per column
        Total duplicate rows

    Parameter
    ----------
    df:  pd.DataFrame 
        A Pandas DataFrame

    Returns
    ----------
       None
    '''
    print("\u0332".join("HEAD "))
    print(f'{df.head()} \n\n')
    print("\u0332".join("SHAPE "))
    print(f'{df.shape} \n\n')
    print("\u0332".join("COLUMNS "))
    print(f'{df.columns}\n\n')
    print("\u0332".join("INFO "))
    print(f'{df.info()}\n\n')
    print("\u0332".join("UNIQUE VALUES "))
    print(f'{df.nunique()} \n\n')
    print("\u0332".join("NUMERICAL COLUMN STATS "))
    print(f'{df.describe()}\n\n')
    print('\u0332'.join("TOTAL NULL VALUES IN EACH COLUMN "))
    print(f'{df.isnull().sum()} \n\n')
    print('\u0332'.join("TOTAL DUPLICATE ROWS "))
    print(f' {df.duplicated().sum()}')


'''
NLP
'''

def custom_stopwords(stop_words, additional_stopwords):
    '''
    Creates a new stopwords set with additional stopwords added to the original stopwords.

    Parameters
    ----------
    stop_words:  set or list
        Original set of stopwords to add new words to.
    additional_stopwords:  set or list
        New stopwords to add to the original stopwords.
    
    Returns
    ----------
    A new stopwords set with all original and additional stopwords.
    '''
    add_stopwords = set(additional_stopwords)
    StopWords = stop_words.union(add_stopwords)
    return set(StopWords)


def lowercase_text(text):
    '''
    Lowercases text.

    Parameter
    ----------
    text: str
        Text to lowercase.
    
    Returns
    ----------
    Lowercased text.
    '''
    return text.lower()

def remove_nums_and_punctuation(text):
    '''
    Removes numbers and puncuation from text.

    Parameter
    ----------
    text: str
        Text to remove numbers and puncuation from.
    
    Returns
    ----------
    Text with numbers and puncuation removed.
    '''
    punc = '!()-[]{};:\\,<>./?@#$%^&*_~;1234567890'
    for ch in text:
        if ch in punc:
            text = text.replace(ch, '')
    return text

def remove_newlines(text):
    '''
    Removes new lines from text.

    Parameter
    ----------
    text: str
        Text to remove new lines from.
    
    Returns
    ----------
    Text with new lines removed.
    '''
    text.replace('\n', '')
    return text

def remove_urls(text):
    '''
    Removes URLs from text.

    Parameter
    ----------
    text: str
        Text to remove URLs from.
    
    Returns
    ----------
    Text with URLs removed.
    '''
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())

def split_text_into_words(text):
    '''
    Splits text into a list of words.

    Parameter
    ----------
    text: str
        Text to create list of words from.
    
    Returns
    ----------
    List of words from the text.
    '''
    return text.split(' ')

def remove_stopwords(word_lst, stop_words):
    '''
    Removes stopwords from text.

    Parameters
    ----------
    word_lst: list
        List of words from which to remove stopwords.
    stop_words: set or list
        Stopwords to remove from the list of words.
    
    Returns
    ----------
    List of words with stopwords removed.
    '''
    return [word for word in word_lst if word not in stop_words]

def lemmatize_word_list(word_lst):
    '''
    Lemmatizes a list of words.

    Parameter
    ----------
    word_lst: list
        List of words to lemmatize.
    
    Returns
    ----------
    List of words to lemmatize.
    '''
    lemmatizer = WordNetLemmatizer()
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in word_lst])
    return lemmatized

def word_list_to_string(word_lst):
    '''
    Creates a string with all words from a list of words.

    Parameter
    ----------
    word_lst: list
        List of words to join into a string.
    
    Returns
    ----------
    String of the words in the word list passed in.
    '''
    return ''.join(word_lst)

def text_cleaner(text, stop_words):
    '''
    A text cleaning pipeline combining the above functions to clean a string of text by lowercasing, removing numbers/puncuation/urls and new lines, lemmatizing.

    Parameters
    ----------
    text:  str 
        The text to be cleaned.

    stop_words set
        Set of stopwords to remove from text.
    
    Returns
    ----------
    String of the cleaned text.
    '''
    text_lc = lowercase_text(text)
    text_np = remove_nums_and_punctuation(text_lc)
    text_nnls = remove_newlines(text_np)
    text_nurl = remove_urls(text_nnls)
    words = split_text_into_words(text_nurl)
    words_nsw = remove_stopwords(words, stop_words)
    lemmatized = lemmatize_word_list(words_nsw)
    cleaned_text = word_list_to_string(lemmatized)
    return cleaned_text


def create_word_cloud(text, 
                width=700, 
                height=700, 
                background_color='black', 
                min_font_size=12
                ):
    '''
    Generates a wordcloud.

    Parameters
    ----------
    text:  str 
        A string of words to create the wordcloud from.
    width:  int 
        Width in pixels of wordcloud image.
    height:  int 
        Height in pixels of wordcloud image.
    background_color:  str
        Color of background of wordcloud image.
    min_font_size:  int
        Minimum font size of words in wordcloud
    Returns
    ----------
    A wordcloud for the text passed in.
    '''
    return WordCloud(
        width=width, 
        height=height,
        background_color=background_color,
        min_font_size=min_font_size).generate(text)


def common_words_graph(text, num_words=15, title='Most Common Words'):
    '''
    Shows horizontal bar graph of most common words in text with their counts in descending order.  Saves png file of the graph.

    Parameters
    ----------
    text:  str 
        Text to find most common words in.
    num_words:  int
        Number of most common words to graph.
    title:  str
        Title of graph.

    Returns
    ----------
    None.
    '''
    txt = text.split()
    sorted_word_counts = collections.Counter(txt)
    sorted_word_counts.most_common(num_words)
    most_common_words = sorted_word_counts.most_common(num_words)
    word_count_df = pd.DataFrame(most_common_words, columns=['Words', 'Count'])

    fig, ax = plt.subplots(figsize=(8, 8))
    word_count_df.sort_values(by='Count').plot.barh(x='Words',
                      y='Count',
                      ax=ax,
                      color='deepskyblue',
                      edgecolor='k')

    ax.set_title(title)
    plt.legend(edgecolor='inherit')
    plt.show()
    plt.savefig(title)


def lexical_diversity(text):
    '''
    Shows total words, total unique words, average word repetition, and proportion of unique words in text.

    Parameter
    ----------
    text:  str 
        Text to analyze.

    Returns
    ----------
    None.
    '''
    txt = text.split()
    print(f'Total words: {len(txt)}')
    print(f'Total unique words: {len(set(txt))}')
    print(f'Average word repetition: {round(len(text)/len(set(text)), 2)}')
    print(f'Proportion of unique words: {round(len(set(txt)) / len(txt), 2)}')


def get_word_context(text, word, lines=10):
    '''
    See examples of the context in which a word in your corpus appears.

    Parameter
    ----------
    text:  str 
        String of text.  The corpus.
    word:  str
        Word to find see .
    lines:  int
        Number of lines to return.

    Returns
    ----------
    Specified number of lines each showing an example from the text of the context of the word passed in.
    '''
    txt = nltk.Text(text.split())
    return txt.concordance(word, lines=lines)


'''
Supervised Learning Classification
'''

def score_class_models(models, X_train, y_train, X_test, y_test):
    '''
    Fits and scores a list of predictive models and returns accuracy and f1 scores for each model.

    Parameter
    ----------
    models:  list 
        A list of models to test.
    X_train:  arr
        X_train data.
    y_train:  arr
        y_train data.
    X_test:  arr
        X_test data.
    y_test:  arr
        y_test data.
        
    Returns
    ----------
    Accuracy and f1 scores for each model in the list of models passed in.
    '''
    acc_score_list = []
    f1_score_list = []

    for model in models:   
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_score_list.append(model.score(X_test, y_test))
        f1_score_list.append(f1_score(y_test, y_pred, average='weighted'))
        
    for model, score, f1_scored in zip(models, acc_score_list, f1_score_list):
        print(f'{model} accuracy: {round(score * 100, 2)} %')
        print(f'{model} f1: {round(f1_scored * 100, 2)} %')     


def conf_matrix(model, X_train, y_train, X_test, y_test):
    '''
    Prints simple confusion matrix for the passed in model.

    Parameters
    ----------
    model: 
        Model to test.
    X_train:  arr
        X_train data.
    y_train:  arr
        y_train data.
    X_test:  arr
        X_test data.
    y_test:  arr
        y_test data.

    Returns
    ----------
    Confusion matrix.
    '''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


if __name__ == '__main__':

    all_data = pd.read_csv('/Users/bn/Galvanize/Twitter-Sentiment-Analysis/data/Tweets.csv')

    df = all_data.drop(columns=['tweet_id','airline_sentiment_confidence', 'negativereason', 'negativereason_confidence', 'airline_sentiment_gold', 'negativereason_gold'])

    df.drop_duplicates(inplace=True)

    df.columns = ['sent', 'airline', 'name', 'rts', 'tweet', 'coords', 'time', 'location', 'timezone']

    df['sent'] = df['sent'].map({'positive':1, 'negative':-1, 'neutral':0})

    df['time'] = df['time'].str.slice(0,16)
    df['time'] = pd.to_datetime(df['time'])

    df['location'] = df['location'].fillna(df['location'].mode()[0])
    df['coords'] = df['coords'].fillna(df['coords'].mode()[0])
    df['timezone'] = df['timezone'].fillna(df['timezone'].mode()[0])

    corpus = ''
    for text in df.tweet:  
        corpus += ''.join(text) + ' '


    StopWords = set(stopwords.words('english'))

    add_stopwords = {'flight', 'virgin america', 'virginamerica', 'virgin', 'united', 'southwest', 'southwestair', 'delta', 'us airways', 'usairways', 'usair', 'airways', 'american', 'americanair', 'aa', 'jet blue', 'jetblue', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'twenty', 'twenty four', '@virginamerica', '@united', '@southwest', '@delta', '@usairways', '@americanair', '@jetblue', '@delta', 'amp'}

    StopWords = custom_stopwords(StopWords, add_stopwords)


    ax = df.groupby(['airline','sent'])['sent'].count().unstack(0).plot.bar(figsize=(10,10), edgecolor='k')
    ax.set_title('Sentiment Counts for each Airline', size=20)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Counts')
    ticks = [0,1,2]
    labels = ['Positive', 'Neutral', 'Negative']
    plt.legend(edgecolor = 'k')
    plt.xticks(ticks, labels, rotation=0)
    # plt.show()
    # plt.savefig('Sentiment Counts by Airline')


    corpus_wordcloud = create_word_cloud(corpus)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(corpus_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 1)
    # plt.show()
    # plt.savefig('corpus-word-cloud')

    cleaned_corpus = text_cleaner(corpus, StopWords)
    cleaned_corpus_wordcloud = create_word_cloud(cleaned_corpus)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(cleaned_corpus_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 1)
    # plt.show()
    # plt.savefig('cleaned-corpus-word-cloud')

    # print(common_words_graph(cleaned_corpus))

    # print(get_word_context(cleaned_corpus, 'thank', lines=10))

    # print(get_word_context(cleaned_corpus, 'service', lines=10))


    def text_cleaner_custom(text, stop_words=StopWords):
        '''
        A text cleaning pipeline combining the above functions to clean a string of text by lowercasing, removing numbers/puncuation/urls and new lines, lemmatizing.

        Parameters
        ----------
        text:  str 
            The text to be cleaned.

        stop_words set
            Set of stopwords to remove from text.
        
        Returns
        ----------
        String of the cleaned text.
        '''
        text_lc = lowercase_text(text)
        text_np = remove_nums_and_punctuation(text_lc)
        text_nnls = remove_newlines(text_np)
        text_nurl = remove_urls(text_nnls)
        words = split_text_into_words(text_nurl)
        words_nsw = remove_stopwords(words, stop_words)
        lemmatized = lemmatize_word_list(words_nsw)
        cleaned_text = word_list_to_string(lemmatized)
        return cleaned_text

    df['tweet'] = df['tweet'].apply(text_cleaner_custom)

    X = df.tweet
    y = df.sent
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

    cv = CountVectorizer(stop_words=StopWords)
    tv = TfidfVectorizer()

    X_train = cv.fit_transform(X_train).toarray()
    X_test = cv.transform(X_test).toarray()
    # X_train = tv.fit_transform(X_train)
    # y_test = tv.fit_transform(y_test)

    untuned_models = [MultinomialNB(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(), SVC(C=6, class_weight='balanced')]

    tuned_models = [MLPClassifier(hidden_layer_sizes=500, activation='relu', solver='adam', alpha=.05, batch_size=10, learning_rate='adaptive'), RandomForestClassifier(n_estimators=12000, max_features=3), SVC(C=6, class_weight='balanced')]

    # print(score_class_models(untuned_models, X_train, y_train, X_test, y_test))

    # print(score_class_models(tuned_models, X_train, y_train, X_test, y_test))

    # print(conf_matrix(SVC(C=3)))


    # corpus_tfm = tv.fit_transform(df.tweet)

    # pca = TruncatedSVD(100)
    # truncated_corpus_tfm = pca.fit_transform(corpus_tfm)

    # kmeans = KMeans(n_clusters=3, n_jobs=-1)
    # kmeans.fit(truncated_corpus_tfm)

    # label = kmeans.fit_predict(truncated_corpus_tfm)

    # filtered_label1 = truncated_corpus_tfm[label == 1]
    # filtered_label2 = truncated_corpus_tfm[label == 2]
    # filtered_label3 = truncated_corpus_tfm[label == 3]

    # plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'blue')
    # plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
    # plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'black')
    # plt.show()




