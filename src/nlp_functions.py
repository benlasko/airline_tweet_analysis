import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.text import Text
import collections
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

def custom_stopwords(stop_words, additional_stopwords):
    '''
    Creates a new stopwords set with additional stopwords added to the original stopwords.

    Parameters
    ----------
    stop_words:  set 
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

def lemmatize_string(string):
    '''
    Lemmatize all words in a string using the spaCy lemmatizer.

    Parameter
    ----------
    text:  str 
        Text to lemmatize.

    Returns
    ----------
    List with each word replaced by its lemma.
    '''
    lem_string = spacy_lemmatizer(string)
    lemmatized = ' '.join([w.lemma_ for w in lem_string])
    return lemmatized

def string_to_word_list(text):
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
    return ' '.join(word_lst)

def text_cleaner(text, stop_words):
    '''
    A text cleaning pipeline combining the above functions to clean a string of text by lowercasing, removing numbers/puncuation/urls and new lines, lemmatizing.

    Parameters
    ----------
    text:  str 
        The text to be cleaned.
    
    Returns
    ----------
    String of the cleaned text.
    '''
    text_lc = lowercase_text(text)
    text_np = remove_nums_and_punctuation(text_lc)
    text_nnls = remove_newlines(text_np)
    text_nurl = remove_urls(text_nnls)
    text_lemd = lemmatize_string(text_nurl)
    words = string_to_word_list(text_lemd)
    words_nsw = remove_stopwords(words, stop_words)
    cleaned_str = word_list_to_string(words_nsw)
    return cleaned_str


def create_word_cloud(text, 
                width=700, 
                height=700, 
                background_color='black', 
                min_font_size=12
                ):
    '''
    Generates word cloud of text.

    Parameter
    ----------
    text:  str 
        A string of words to create the word cloud from.
    width:  int 
        Width in pixels of word cloud image.  default = 700
    height:  int 
        Height in pixels of word cloud image.  default = 700
    background_color:  str
        Color of background of word cloud image.  default = "black"
    min_font_size:  int
        Minimum font size of words.  default = 12

    Returns
    ----------
    Word cloud object.
    '''
    return WordCloud(width=width, 
                    height=height,
                    background_color=background_color,
                    min_font_size=min_font_size).generate(text)


def common_words_graph(text, num_words=15, title='Most Common Words'):
    '''
    Shows horizontal bar graph of most common words in text with their counts in descending order.  Saves the graph as a png file.

    Parameter
    ----------
    text:  str 
        Text to find most common words in.
    num_words:  int
        Number of most common words to graph.  default = 15
    title:  str
        Title of graph. default = "Most Common Words"

    Returns
    ----------
    None.
    '''
    txt = text.split()
    sorted_word_counts = collections.Counter(txt)
    sorted_word_counts.most_common(num_words)
    most_common_words = sorted_word_counts.most_common(num_words)
    word_count_df = pd.DataFrame(most_common_words,columns=['Words', 'Count'])

    fig, ax = plt.subplots(figsize=(8, 8))
    word_count_df.sort_values(by='Count').plot.barh(x='Words',
                                                    y='Count',
                                                    ax=ax,
                                                    color='deepskyblue',
                                                    edgecolor='k')
    ax.set_title(title)
    plt.legend(edgecolor='inherit')
    plt.show()
    plt.savefig('common words graph')


def lexical_diversity(text):
    '''
    Prints total words, total unique words, average word repetition, and proportion of unique words for the text passed in.

    Parameter
    ----------
    text:  str 
        Text to analayze.

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
    See examples of the context in which a word is found in the text passed in.

    Parameter
    ----------
    text:  str 
        Text to find the context of words in.
    word:  str
        Word to see the context for.
    lines:  int
        Number of lines to return.  default = 10

    Returns
    ----------
    Specified number of lines each showing the context of the specified word.
    '''
    txt = nltk.Text(text.split())
    return txt.concordance(word, lines=lines)


