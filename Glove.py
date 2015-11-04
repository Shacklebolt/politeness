import numpy as np
import pickle
import csv
import re


def make_glove_pickle(glove_file, output_file, text_file):
    # Make sure the text_file parameter is a csv with the request in the second place
    all_gloves = {}
    treebanks_gloves = {}

    with open(glove_file, 'rb') as f:
        lines = f.readlines()

    # Load all gloves into a dictionary
    for line in lines:
        parts = line.split(' ', 1)
        vec = np.fromstring(parts[1], dtype='float64', sep=' ')
        all_gloves[parts[0]] = vec[:, np.newaxis]

    # Filter out the gloves in the corpus
    with open(text_file, 'rU') as fh:
        data = list(csv.reader(fh))
        for request in data:
            text = request[1]
            words = text.split()
            for word in words:
                cleaned = re.sub('[^A-Za-z]', '', word)
                try:
                    treebanks_gloves[cleaned] = all_gloves[cleaned]
                except KeyError:
                    pass

    # Also, these are super important
    treebanks_gloves['unknown'] = all_gloves['unknown']
    treebanks_gloves['<person>'] = all_gloves['person']
    treebanks_gloves['<url>'] = all_gloves['url']
    treebanks_gloves['!'] = all_gloves['!']
    treebanks_gloves['.'] = all_gloves['.']
    treebanks_gloves[';'] = all_gloves[';']
    treebanks_gloves['?'] = all_gloves['?']
    treebanks_gloves[','] = all_gloves[',']
    treebanks_gloves['-'] = all_gloves['-']

    with open(output_file, 'wb') as pickle_file:
        pickle.dump(treebanks_gloves, pickle_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print 'Generating new glove pickle...'
    make_glove_pickle('glove-100d.txt', 'treebank_vectors_100d_new.pickle', 'treebanks/wiki_quartiles.csv')