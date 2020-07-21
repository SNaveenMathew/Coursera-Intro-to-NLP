import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        brown_words+=[START_SYMBOL]*2
        brown_tags+=[START_SYMBOL]*2
        words_tags=sentence.strip().split()
        for word_tag in words_tags:
            all=word_tag.split('/')
            tag=all.pop(-1)
            word='/'.join(all)
            brown_words.append(word)
            brown_tags.append(tag)
        brown_words+=[STOP_SYMBOL]
        brown_tags+=[STOP_SYMBOL]
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    tot=0
    uni_count={}
    bi_count={}
    tri_count={}
    trigrams=list(nltk.trigrams(brown_tags))
    bigrams=list(nltk.bigrams(brown_tags))
    for uni in brown_tags:
        try:
            uni_count[uni]+=1
        except:
            uni_count[uni]=1
    for bi in bigrams:
        try:
            bi_count[bi]+=1
        except:
            bi_count[bi]=1
    for tri in trigrams:
        try:
            tri_count[tri]+=1
        except:
            tri_count[tri]=1
    for tri in tri_count.keys():
        q_values[tri]=math.log(float(tri_count[tri])/bi_count[(tri[0], tri[1])], 2)
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    counts={}
    words=[]
    for word in brown_words:
        try:
            counts[word]+=1
            if counts[word]>RARE_WORD_MAX_FREQ:
                words.append(word)
        except:
            counts[word]=1
    known_words = set(words)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    i=0
    while i<len(brown_words):
        sentence=[]
        while brown_words[i]!=STOP_SYMBOL:
            word=brown_words[i]
            if word in known_words:
                sentence.append(word)
            else:
                sentence.append(RARE_SYMBOL)
            i=i+1
        sentence.append(STOP_SYMBOL)
        i=i+1
        brown_words_rare.append(sentence)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    tags=[]
    int_dic={}
    tag_dic={}
    taglist=[]
    j=0
    for i in range(len(brown_words_rare)):
        tokens=brown_words_rare[i]
        for token in tokens:
            try:
                int_dic[(token, brown_tags[j])]+=1
            except:
                int_dic[(token, brown_tags[j])]=1
            try:
                tag_dic[brown_tags[j]]+=1
            except:
                tag_dic[brown_tags[j]]=1
            j=j+1
    j=0
    for i in range(len(brown_words_rare)):
        tokens=brown_words_rare[i]
        for token in tokens:
            tag=brown_tags[j]
            e_values[(token,tag)]=math.log(float(int_dic[token,tag])/tag_dic[tag],2)
            taglist.append(tag)
            j=j+1
    taglist = set(taglist)
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    taglist=list(taglist)
    for i in range(len(brown_dev_words)):
        tagg=[]
        sentence_words=[START_SYMBOL]+[START_SYMBOL]+brown_dev_words[i]+[STOP_SYMBOL]
        current_trigrams=list(nltk.trigrams(sentence_words))
        bigram_tag=['*','*']
        for trigram in current_trigrams:
            if trigram[2]!=STOP_SYMBOL:
                dic={}
                probs=[]
                for tag in taglist:
                    dic[tag]=0
                    try:
                        dic[tag]+=e_values[(trigram[2],tag)]
                    except:
                        dic[tag]+=LOG_PROB_OF_ZERO
                    try:
                        dic[tag]+=q_values[(bigram_tag[0],bigram_tag[1],tag)]
                    except:
                        dic[tag]+=LOG_PROB_OF_ZERO
                    probs.append(dic[tag])
                tagg.append(trigram[2]+"/"+taglist[probs.index(max(probs))])
                bigram_tag[0]=bigram_tag[1]
                bigram_tag[1]=taglist[probs.index(max(probs))]
        tagg=" ".join(tagg)+'\n'
        tagged.append(tagg)
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    tags=list(set(brown_tags))
    tagged = []
    perc=0
    tag_dic={}
    for i in range(len(brown_words)):
        try:
            tag_dic[(brown_words[i], brown_tags[i])]+=1
        except:
            tag_dic[(brown_words[i], brown_tags[i])]=1
    for i in range(len(brown_dev_words)):
        sentence=brown_dev_words[i]
        for word in sentence:
            probs=[]
            for tag in tags:
                try:
                    probs.append(tag_dic[(word, tag)])
                except:
                    probs.append(0)
            sentence[sentence.index(word)]=word+'/'+tags[probs.index(max(probs))]
        sentence=" ".join(sentence)+'\n'
        tagged.append(sentence)
        if float(i)/len(brown_dev_words)>perc:
            print perc
            perc+=10
    # IMPLEMENT THE REST OF THE FUNCTION HERE
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
