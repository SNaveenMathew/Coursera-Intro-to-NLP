import math
import nltk
import time
import datetime

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    total=0
    uni_counts={}
    bi_counts={}
    tri_counts={}
    unigram_p={}
    bigram_p={}
    trigram_p={}
    for sentence in training_corpus:
        sentence=sentence.strip()
        current_uni=sentence.split()
        current_uni=[START_SYMBOL]+[START_SYMBOL]+current_uni[:-1]+[STOP_SYMBOL]
        for uni in current_uni:
            try:
                uni_counts[uni]+=1
            except KeyError:
                uni_counts[uni]=1

        current_bi=list(nltk.bigrams(current_uni))
        for bi in current_bi:
            try:
                bi_counts[bi]+=current_bi.count(bi)
            except KeyError:
                bi_counts[bi]=current_bi.count(bi)

        current_tri=list(nltk.trigrams(current_uni))
        for tri in current_tri:
            try:
                tri_counts[tri]+=current_tri.count(tri)
            except KeyError:
                tri_counts[tri]=current_tri.count(tri)
        total+=len(current_bi)

    unigram_p = {uni : math.log(float(uni_counts[uni])/696742,2) for uni in uni_counts.keys()}
    for bi in bi_counts.keys():
        bigram_p[bi] = math.log(float(bi_counts[bi])/uni_counts[bi[0]],2)
    for tri in tri_counts.keys():
        trigram_p[tri] = math.log(float(tri_counts[tri])/bi_counts[(tri[0], tri[1])],2)
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        sentence=sentence.strip()
        current_uni=sentence.split()
        current_uni=[START_SYMBOL]+[START_SYMBOL]+current_uni[:-1]+[STOP_SYMBOL]
        if n==1:
            n_grams=current_uni
        elif n==2:
            n_grams=list(nltk.bigrams(current_uni))
        else:
            n_grams=list(nltk.trigrams(current_uni))
        score=0
        for n_gram in n_grams:
            try:
                score=score+ngram_p[n_gram]
            except KeyError:
                continue
                # score+=MINUS_INFINITY_SENTENCE_LOG_PROB
        scores.append(score)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for sentence in corpus:
        sentence=sentence.strip()
        current_uni=sentence.split()
        current_uni=[START_SYMBOL]+[START_SYMBOL]+current_uni[:-1]+[STOP_SYMBOL]
        score=0
        for uni in current_uni:
            try:
                score+=unigrams[uni]
            except KeyError:
                continue
                # score+=MINUS_INFINITY_SENTENCE_LOG_PROB

        current_bi=list(nltk.bigrams(current_uni))
        for bi in current_bi:
            try:
                score+=bigrams[bi]
            except KeyError:
                continue
                # score+=MINUS_INFINITY_SENTENCE_LOG_PROB

        current_tri=list(nltk.trigrams(current_uni))
        for tri in current_tri:
            try:
                score+=trigrams[tri]
            except KeyError:
                continue
                # score+=MINUS_INFINITY_SENTENCE_LOG_PROB
        score=score/3
        scores.append(score)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()
    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()
    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)
    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
