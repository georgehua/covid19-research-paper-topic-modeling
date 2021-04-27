# -*- coding: utf-8 -*-
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


def clean_str(s, stopword_list):
    '''
    Clean the text features to make sure that the remaining words are meaningful

    @param s: input string
    @param stopword_list: a list of stopwords from nltk and customization
    @return: cleaned string
    '''
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


def main(args):

    logger = logging.getLogger(__name__)
    logger.info('Preprocessing ' + args.input)

    results = []
    stopword_list = build_stop_words()

    with timer("Load & Clean"):
        with open(f"{args.dir}/{args.input}", 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                abstract_col = clean_str(row[9], stopword_list)
                # title, abstract, publish_time, authors, url
                results.append(
                    [row[4], abstract_col, row[10], row[11], row[18]])

    with timer("Write to " + args.output):
        with open(f"{args.dir}/{args.output}", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument(
        "-i", "--input", help="Input csv file name", default='small.csv')
    parser.add_argument(
        "-o", "--output", help="Output csv file name", default='small_cleaned.csv')
    parser.add_argument(
        "-dir", help="Data directory", default='../data')

    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args)
