import matplotlib.pyplot as plt
import numpy as np
import csv
import html
import re
import random
import json
import nltk
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from string import ascii_lowercase
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

filename = 'data/twitter.csv'
filename_cache = 'data/twitter.json'
nb_samples = 100000
nb_words = 10000
nb_classes = 2
nb_alpha = 1
ds_from_cache = True

class Dataset:
    def __init__(self):
        if ds_from_cache:
            print('Using cached dataset...')
            self.deserialize()
        else:
            print('Using new dataset...')
            self.load(filename)
            self.clean()
            self.calculate_bow_lr()
            self.split()
            self.calculate_occurrences()
            self.calculate_likelihoods()
            self.serialize()

    def compress_np1d(self, arr):
        return {i: str(arr[i]) for i in range(len(arr)) if arr[i] != 0}

    def decompress_np1d(self, map):
        arr = np.zeros(nb_words, dtype=np.float32)
        for (i, x) in map.items():
            arr[int(i)] = float(x)
        return arr

    def serialize(self):
        print('Serializing dataset...')
        with open(filename_cache, 'w') as f:
            compress_train_x = [self.compress_np1d(x) for x in self.train_x]
            compress_test_x = [self.compress_np1d(x) for x in self.test_x]
            ds_json = {
                'train_x': compress_train_x,
                'train_y': self.train_y.tolist(),
                'test_x': compress_test_x,
                'test_y': self.test_y.tolist(),
                'like': self.like.tolist(),
                'top_neg': self.top_neg,
                'top_pos': self.top_pos,
                'lr_min': self.lr_min,
                'lr_max': self.lr_max
            }
            json.dump(ds_json, f)

    def deserialize(self):
        with open(filename_cache, 'r') as f:
            ds_json = json.load(f)
            self.train_x = [self.decompress_np1d(x) for x in ds_json['train_x']]
            self.train_y = ds_json['train_y']
            self.test_x = [self.decompress_np1d(x) for x in ds_json['test_x']]
            self.test_y = ds_json['test_y']
            self.like = ds_json['like']
            self.top_neg = ds_json['top_neg']
            self.top_pos = ds_json['top_pos']
            self.lr_min = ds_json['lr_min']
            self.lr_max = ds_json['lr_max']

    # Ucitavanje podataka
    def load(self, filename):
        print('Loading data...')

        self.data_x = []
        self.data_y = []

        with open(filename, 'r', encoding='latin1') as fin:
            reader = csv.reader(fin, delimiter=',')
            next(reader, None)
            for row in reader:
                self.data_y.append(int(row[1]))
                self.data_x.append(row[2])

    # Ciscenje podataka
    def clean(self):
        print('Cleaning data...')

        self.data_x = [html.unescape(x) for x in self.data_x]

        self.data_x = [re.sub(r'https?://\S+', '', x) for x in self.data_x]
        self.data_x = [re.sub(r'[^\w\s]|\d+', '', x) for x in self.data_x]
        self.data_x = [re.sub(r'\s\s+', ' ', x) for x in self.data_x]
        self.data_x = [x.strip().lower() for x in self.data_x]
        for c in ascii_lowercase:
            self.data_x = [re.sub(c + '{3,}', c+c, x) for x in self.data_x]
        
        self.data_x = [regexp_tokenize(x, '\w+') for x in self.data_x]
        
        stops = set(stopwords.words('english'))
        self.data_x = [[w for w in x if not w in stops] for x in self.data_x]

        self.data_x = self.data_x[:nb_samples]
        self.data_y = self.data_y[:nb_samples]

    # Racunanje BOW reprezentacije i LR metrike
    def calculate_bow_lr(self):
        print('Calculating BOW representation and LR metric...')
        
        freq = FreqDist([w for x in self.data_x for w in x])
        self.vocab, _ = zip(*freq.most_common(nb_words))
        
        self.vec_x = np.zeros((len(self.data_x), nb_words), dtype=np.float32)
        lr = np.zeros(nb_words, dtype=np.float32)

        for j, w in enumerate(self.vocab):
            neg = 0
            pos = 0
            for i, x in enumerate(self.data_x):
                cnt = x.count(w)
                self.vec_x[i][j] = cnt
                if self.data_y[i] == 0:
                    neg += cnt
                else:
                    pos += cnt
            if pos >= 10 and neg >= 10:
                lr[j] = pos / neg
            if j % 100 == 0:
                print('[calculate_bow_lr] Word: {}/{}'.format(j, nb_words))

        # Pronalazenje pet najcesce koriscenih reci u negativnim tvitovima
        freq_neg = FreqDist([w for i, x in enumerate(self.data_x) for w in x if self.data_y[i] == 0])
        self.top_neg, _ = zip(*freq_neg.most_common(5))

        # Pronalazenje pet najcesce koriscenih reci u pozitivnim tvitovima
        freq_pos = FreqDist([w for i, x in enumerate(self.data_x) for w in x if self.data_y[i] == 1])
        self.top_pos, _ = zip(*freq_pos.most_common(5))

        # Pronalazenje pet reci sa najmanjom vrednoscu LR metrike
        self.lr_min = []
        min_cnt = 1
        for i in lr.argsort():
            if min_cnt > 5:
                break
            if lr[i] > 0:
                self.lr_min.append(self.vocab[i])
                min_cnt += 1

        # Pronalazenje pet reci sa najvecom vrednoscu LR metrike
        self.lr_max = []
        max_cnt = 1
        for i in (-lr).argsort():
            if max_cnt > 5:
                break
            if lr[i] > 0:
                self.lr_max.append(self.vocab[i])
                max_cnt += 1

    # Deljenje podataka na skup za treniranje i testiranje
    def split(self):
        print('Splitting data...')

        self.train_x, self.test_x = np.split(self.vec_x, [int(len(self.vec_x)*0.8)])
        self.train_y, self.test_y = np.split(self.data_y, [int(len(self.data_y)*0.8)])
        
        self.nb_train = len(self.train_x)
        self.nb_test = len(self.test_x)

    # Racunanje broja pojavljivanja svake reci u svakoj klasi
    def calculate_occurrences(self):
        print('Calculating every word occurrence for every class...')
        self.occs = np.zeros((nb_classes, nb_words), dtype=np.float32)
        for i, y in enumerate(self.train_y):
            for w in range(nb_words):
                self.occs[y][w] += self.train_x[i][w]
            if i % 1000 == 0:
                print('[calculate_occurrences] Object: {}/{}'.format(i, self.nb_train))

    # Racunanje P(rec|klasa)
    def calculate_likelihoods(self):
        print('Calculating P(word|class)...')
        self.like = np.zeros((nb_classes, nb_words), dtype=np.float32)
        for c in range(nb_classes):
            for w in range(nb_words):
                up = self.occs[c][w] + nb_alpha
                down = np.sum(self.occs[c]) + nb_words * nb_alpha
                self.like[c][w] = up / down
                if w % 1000 == 0:
                    print('[calculate_likelihoods] Word: {}/{}'.format(w, nb_words))

ds = Dataset()    

# Racunanje P(klasa|test)
print('Calculating P(class|test)...')
hyps = []
acts = []
priors = np.bincount(ds.train_y) / nb_samples
class_trans = {'Negative': 0, 'Positive': 1, 0: 'Negative', 1: 'Positive'}
nb_test = len(ds.test_x)
for i, x in enumerate(ds.test_x):
    probs = np.zeros(nb_classes)
    for c in range(nb_classes):
        prob = np.log(priors[c])
        for w in range(nb_words):
            prob += x[w] * np.log(ds.like[c][w])
        probs[c] = prob
    hyp_val = np.argmax(probs)
    act_val = ds.test_y[i]
    match = hyp_val == act_val
    hyps.append(class_trans[hyp_val])
    acts.append(class_trans[act_val])
    if i % 100 == 0:
        print('{}/{} Predicted: {} Actual: {} Match: {}'
        .format(i+1, nb_test, class_trans[hyp_val], class_trans[act_val], match))

# Racunanje tacnosti modela
cm = confusion_matrix(acts, hyps)
tn, _, _, tp = cm.ravel()
acc = (tn + tp) / nb_test
print('Accuracy:', acc)

# Prikazivanje matrice konfuzije
plt.matshow(cm)
plt.colorbar()
plt.xlabel('Hypothesis')
plt.ylabel('Actual')
plt.show()

print('Top negative:', ds.top_neg)
print('Top positive:', ds.top_pos)

print('LR lowest:', ds.lr_min)
print('LR highest:', ds.lr_max)

# ODGOVOR 4C
# Posmatrajući skup reči koje se najčešće pojavljuju u pozitivnim
# i skup reči koje se najčešće pojavljuju u negativnim tvitovima,
# možemo primetiti da presek ta dva skupa nije prazan, tj.
# postoje reči koje se inače često upotrebljavju.
# Kako bi se stekao uvid u realnu situaciju
# (skup reči koje su jako vezane za sentiment tvita),
# može se koristiti LR metrika (likelihood ratio)
# koja relativizuje frekvencije pojavljivanja reči.
