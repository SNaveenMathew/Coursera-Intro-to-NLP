import A, nltk, pandas
from sklearn.feature_extraction import DictVectorizer


# You might change the window size
window_size = 15

# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    # implement your code here

    for tupl in data:
        inst=tupl[0]
        dic={}
        left=nltk.word_tokenize(tupl[1])
        right=nltk.word_tokenize(tupl[3])
        if len(left)<window_size:
            tot=list(left)
        else:
            tot=list(left[-window_size:])
        if len(right)<window_size:
            tot=tot+list(right)
        else:
            tot=tot+right[:window_size]
        uniq=list(set(tot))
        for word in uniq:
            dic[word]=tot.count(word)
        features[inst]=dic
        labels[inst]=tupl[4]

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''
    from sklearn import svm
    results = []


    # implement your code here

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
    x=x.fillna(value=0)

    svm_clf = svm.LinearSVC()
    svm_clf.fit(X=x, y=y)
    keys=X_test.keys()

    for key in keys:
        x_test.append(X_test[key])

    x_test=pandas.DataFrame(x_test)
    x_test=x_test.fillna(value=0)
    svm=svm_clf.predict(x_test)

    for i in range(len(keys)):
        results.append((keys[i], svm_clf.predict(x_test)[i]))

    return results

# run part B
def run(train, test, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)
