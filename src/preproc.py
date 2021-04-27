# -*- coding: utf-8 -*-
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from langdetect import DetectorFactory
from langdetect import detect
from tqdm import tqdm
import pandas as pd
import numpy as np
from timer import timer
import logging

import csv
import argparse
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')


def build_stop_words():
    '''
    Using NLTK's list of stopwords

    @return: stopword list
    '''
    stopword_list = stopwords.words('english')

    # custom stopwords
    more_sw = ['doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author',
               'figure', 'table', 'rights', 'reserved', 'permission', 'use', 'used',
               'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.',
               'results', 'study', 'shown', 'based', 'found', 'could', 'observed',
               'cases', 'publisher', 'including', 'against', 'say', 'beyond', 'seem',
               'thereby', 'bottom', 'get', 'peer', 'review', 'permission', 'Elsevier',
               'PMC', 'CZI', 'may', 'also', 'due', 'however', 'within', 'del', 'las',
               'por', 'los', 'para', 'que', 'non', 'none', 'il', 'paper', 'undergo',
               'underwent', 'rt', 'les', 'number', 'numbers', 'propose', 'proposed',
               'one', 'two', 'three', 'assay', 'article', 'group', 'conclusion',
               'conclusions']
    for sw in more_sw:
        stopword_list.append(sw)

    return stopword_list


def clean_str(s, stopword_list):
    '''
    Clean the text features to make sure that the remaining words are meaningful

    @param s: input string
    @param stopword_list: a list of stopwords from nltk and customization
    @return: cleaned string
    '''

    s = s.lower()

    # Remove HTML tags and attributes from the string
    html_tags = re.compile('<.*?>')
    s = re.sub(html_tags, '', s)

    # Replace HTML character codes with ASCII equivalent
    s = unicodedata.normalize('NFKD', s).encode(
        'ascii', 'ignore').decode('utf-8')

    # Remove URLs
    s = re.sub(r'http\S+', '', s)

    # Remove new line and line breaks characters
    s = s.replace('b"', '').replace("b'", '').replace('\n', '').replace(
        '\\n', '').replace('\\n\\n', '').replace('\t', '')

    # Remove punctuations
    s = re.sub('[^a-zA-z0-9\s-]', ' ', s)

    # Replace extra white spaces with one space
    s = re.sub(r'\s+', ' ', s)

    # Lemmatization
    s = lemmatizer.lemmatize(s)

    # Remove stop words
    text_tokens = word_tokenize(s)
    s = [word for word in text_tokens if not word in stopword_list]

    # Remove words with only 2 letters or less
    s = [i for i in s if len(i) > 2]

    return ' '.join(s)


def check_language(df):
    '''
    Check if each research paper is written in English, if not we drop them
    @param df: input df
    @return pandas dataframe
    '''
    # set seed
    DetectorFactory.seed = 0

    # hold label - language
    languages = []

    # go through each text
    for ii in tqdm(range(0, len(df))):
        # split by space into list, take the first x intex, join with space
        text = str(df.iloc[ii]['abstract']).split(" ")

        lang = "en"
        try:
            lang = detect(" ".join(text[:len(text)]))
        except Exception as e:
            lang = "unknown"

        # get the language
        languages.append(lang)

    df['language'] = languages
    df = df[df['language'] == 'en']

    return df.drop(["language"], axis=1)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence)))


def main(args):

    logger = logging.getLogger(__name__)
    logger.info('Preprocessing ' + args.input)

    results = []
    stopword_list = build_stop_words()
    df = pd.DataFrame([])
    docs = np.array([])

    with timer("Load & Clean"):
        with open(f"{args.dir}/{args.input}", 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                abstract_col = clean_str(row[8], stopword_list)
                # title, abstract, publish_time, authors, url
                results.append(
                    [row[3], abstract_col, row[9], row[10], row[17]])

        df = pd.DataFrame(results[1:], columns=results[0])

    with timer("Drop NA & Duplicates"):
        df = df.drop_duplicates(subset='abstract', keep='first')
        df = df.dropna(subset=["abstract"])

    with timer("Drop Non-English Papers"):
        df = check_language(df)

    with timer("Vectorize abstract column"):
        docs = np.array(list(sent_to_words(df.abstract)))

    with timer("Add bigrams to docs"):
        # Add bigrams to docs (only ones that appear 20 times or more).
        bigram = Phrases(docs, min_count=20)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

    with timer("Export results"):
        df.to_csv(f"{args.dir}/df_cleaned.csv", encoding='utf-8', index=False)
        np.save(f"{args.dir}/docs.npy", docs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument(
        "-i", "--input", help="Input csv file name", default='metadata.csv')
    parser.add_argument(
        "-dir", help="Data directory", default='data')

    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args)
