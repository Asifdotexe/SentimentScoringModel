import pandas as pd
import re  # for regex
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from string import punctuation

def clean_html_column(dataframe, column_name, new_column_name):
    """
    Remove HTML tags from the text in the specified column of the DataFrame
    and create a new column with the cleaned text.

    Parameters:
    - dataframe (DataFrame): The DataFrame containing the text data.
    - column_name (str): The name of the column containing the text to be cleaned.
    - new_column_name (str): The name of the new column to be created with the cleaned text.

    Returns:
    - dataframe (DataFrame): The DataFrame with the new column added, containing the cleaned text.
    """

    def remove_html_tags(text):
        """
        Remove HTML tags from the input text using regex.

        Parameters:
        - text (str): The input text containing HTML tags.

        Returns:
        - clean_text (str): The text with HTML tags removed.
        """
        if isinstance(text, str):
            clean_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags using regex
            return clean_text
        else:
            return text

    # Apply remove_html_tags function to the specified column
    dataframe[new_column_name] = dataframe[column_name].apply(remove_html_tags)
    return dataframe

def penn_to_wn(tag):
    """
    Convert between a Penn Treebank tag to a simplified Wordnet tag.

    Parameters:
    - tag (str): Penn Treebank tag.

    Returns:
    - wn_tag (str): Simplified Wordnet tag.
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_lemmas(tokens):
    """
    Get lemmas for a list of tokens.

    Parameters:
    - tokens (list): List of tokens.

    Returns:
    - lemmas (list): List of lemmatized tokens.
    """
    lemmas = []  # Initialize an empty list to store lemmas

    # Iterate through each token in the input list of tokens
    for token in tokens:
        # Get the part of speech (POS) of the token using penn_to_wn function
        pos = penn_to_wn(pos_tag([token])[0][1])
        if pos:  # Check if the POS is not None
            # Lemmatize the token using WordNetLemmatizer, considering its POS
            lemma = lemmatizer.lemmatize(token, pos)
            # Check if the lemma is not None
            if lemma:
                # Append the lemma to the list of lemmas
                lemmas.append(lemma)
    return lemmas

def get_sentiment_score(text):
    
    """
        This method returns the sentiment score of a given text using SentiWordNet sentiment scores.
        input: text
        output: numeric (double) score, >0 means positive sentiment and <0 means negative sentiment.
    """    
    total_score = 0
    #print(text)
    raw_sentences = sent_tokenize(text)
    #print(raw_sentences)
    
    for sentence in raw_sentences:

        sent_score = 0     
        sentence = str(sentence)
        #print(sentence)
        sentence = sentence.replace("<br />"," ").translate(str.maketrans('','',punctuation)).lower()
        tokens = TreebankWordTokenizer().tokenize(text)
        tags = pos_tag(tokens)
        for word, tag in tags:
            wn_tag = penn_to_wn(tag)
            if not wn_tag:
                continue
            lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sent_score += swn_synset.pos_score() - swn_synset.neg_score()

        total_score = total_score + (sent_score / len(tokens))

    
    return (total_score / len(raw_sentences)) * 100