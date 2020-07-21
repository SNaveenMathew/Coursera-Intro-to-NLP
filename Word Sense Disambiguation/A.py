import unicodedata, itertools, nltk, datetime, pandas, numpy
from nltk.corpus import stopwords
from sklearn import neighbors, svm

# don't change the window size
window_size = 10

# A.1

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])


def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    #Implement code here

    for key in data.keys():
        lis=data[key]
        for item in lis:
            left=nltk.word_tokenize(item[1])
            right=nltk.word_tokenize(item[3])
            if len(left)<window_size:
                tot=list(left)
            else:
                tot=list(left[-window_size:])
            if len(right)<window_size:
                tot=tot+list(right)
            else:
                tot=tot+right[:window_size]
            try:
                s[key].append(tot)
            except:
                s[key]=[tot]

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}
    all_words=[]

    for words in s:
        all_words=all_words+words
    stop = stopwords.words('english')
    ob=nltk.stem.snowball.EnglishStemmer()
    all_words=[ob.stem(word).lower() for word in all_words if word not in stop and len(word)>2]
    all_words=list(set(all_words))

    for i in range(len(data)):
        inst=data[i][0]
        counts=[]
        left=nltk.word_tokenize(data[i][1])
        right=nltk.word_tokenize(data[i][3])
        if len(left)<window_size:
            words=list(left)
        else:
            words=list(left[-window_size:])
        if len(right)<window_size:
            words=words+list(right)
        else:
            words=words+list(right[-window_size:])
        for j in range(len(words)):
            words[j]=ob.stem(words[j]).lower()
        # pos=nltk.pos_tag(words)
        for word in all_words:
            counts.append(words.count(word))
            # if word in words:
                # counts.append(pos[words.index(word)])
            # else:
                # counts.append('')
        vectors[inst]=list(counts)
        labels[inst]=data[i][4]

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []
    x=[]
    y=[]
    x_test=[]

    for key in X_train.keys():
        X=X_train[key]
        Y=y_train[key]
        x.append(X)
        y.append(y_train[key])

    x=pandas.DataFrame(x)
    y=pandas.DataFrame(y)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    svm_clf = svm.LinearSVC()
    svm_clf.fit(X=x, y=y)
    knn_clf.fit(X=x, y=y)
    keys=X_test.keys()

    for key in keys:
        x_test.append(X_test[key])

    x_test=pandas.DataFrame(x_test)
    knn=knn_clf.predict(x_test)
    svm=svm_clf.predict(x_test)

    for i in range(len(keys)):
        knn_results.append((keys[i], knn_clf.predict(x_test)[i]))
        svm_results.append((keys[i], svm_clf.predict(x_test)[i]))

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results, output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    out=open(output_file, 'w')
    keys=sorted(results.keys())
    for key in keys:
        lis=results[key]
        instances=[]
        for row in lis:
            inst=row[0].split('.')
            inst=inst[-1]
            instances.append(int(inst))
        sorted_instances=sorted(instances)
        for inst in sorted_instances:
            row=lis[instances.index(inst)]
            inst1=row[0]
            lab=unicode(row[1])
            key=unicode(key)
            inst1=replace_accented(inst1)
            try:
                lab=replace_accented(lab)
            except:
                lab=lab
            try:
                key=replace_accented(key)
            except:
                key=key
            out.writelines(key+" "+inst1+" "+lab+"\n")

# run part A
def run(train, test, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s.keys():
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)
