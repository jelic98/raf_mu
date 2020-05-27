#!pip install yfinance

import math
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# Hiperparametri
epoch_max = 10
alpha_max = 0.025
alpha_min = 0.001
batch_size = 32
window_size = 7
test_ratio = 0.1
max_time = 16
lstm_size = 64
embed_size = 2

# Simboli akcija
train_symbols, test_symbols = None, ['AAPL', 'GOOG']

with open('data/symbols.txt') as f:
    train_symbols = list(f)

train_symbols = [sym.replace('\n', '') for sym in train_symbols]
nb_stocks = len(train_symbols)

# Preuzimanje istorijskih podataka svake akcije
for sym in train_symbols:
    print('Downloading {} ...'.format(sym))
    df = yf.download(sym, '2010-05-25', '2020-05-25')
    df.to_csv('data/sp500_{}.csv'.format(sym))

dates = []
nb_train, nb_test, nb_batches = [], [], []
train_x, train_y = [], []
test_x, test_y = [], []
norm_test_y = []

# Ucitavanje podataka svake akcije
for sym_index, sym in enumerate(train_symbols):
    csv = pd.read_csv('data/sp500_{}.csv'.format(sym))
    dts, data = csv['Date'].values, csv['Close'].values

    # Konverzija datuma
    dts = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dts]
    dates.append([dts[i + max_time] for i in range(len(dts) - max_time)])

    # Grupisanje podataka pomocu kliznog prozora
    data = [data[i : i + window_size] for i in range(len(data) - window_size)]

    # Normalizacija podataka
    norm = [data[0][0]] + [data[i-1][-1] for i, _ in enumerate(data[1:])]
    data = [curr / norm[i] - 1.0 for i, curr in enumerate(data)]

    nb_samples = len(data) - max_time
    nb_train.append(int(nb_samples * (1.0 - test_ratio)))
    nb_test.append(nb_samples - nb_train[sym_index])
    nb_batches.append(math.ceil(nb_train[sym_index] / batch_size))

    # Grupisanje podataka za propagaciju greske kroz vreme
    x = [data[i : i + max_time] for i in range(nb_samples)]
    y = [data[i + max_time][-1] for i in range(nb_samples)]

    # Skup podataka za treniranje
    train_x.append([x[i : i + batch_size] for i in range(0, nb_train[sym_index], batch_size)])
    train_y.append([y[i : i + batch_size] for i in range(0, nb_train[sym_index], batch_size)])

    # Skup podataka za testiranje
    test_x.append(x[-nb_test[sym_index]:])
    test_y.append(y[-nb_test[sym_index]:])

    # Skup podataka za denormalizaciju
    norm_y = [norm[i + max_time] for i in range(nb_samples)]
    norm_test_y.append(norm_y[-nb_test[sym_index]:])

tf.reset_default_graph()

# Cene tokom prethodnih dana
X = tf.placeholder(tf.float32, [None, max_time, window_size])

# Cena na trenutni dan
Y = tf.placeholder(tf.float32, [None])

# Stopa ucenja
L = tf.placeholder(tf.float32)

# Celobrojna reprezentacija simbola akcija
S = tf.placeholder(tf.int32, [None, 1])

# Embedovanje simbola akcija
embeds = tf.Variable(tf.random_uniform([nb_stocks, embed_size], maxval=1.0))
sym_tiles = tf.tile(S, [1, max_time])
lookup = tf.nn.embedding_lookup(embeds, sym_tiles)

# Spajanje cena tokom prethodnih dana sa embedingom simbola akcija
x_embed = tf.concat([X, lookup], axis=2)

# LSTM sloj
rnn = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(lstm_size)])

# Izlaz LSTM sloja
val, _ = tf.nn.dynamic_rnn(rnn, x_embed, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])

# Poslednji izlaz LSTM sloja
last = tf.gather(val, val.get_shape()[0] - 1)

# Obucavajuci parametri
weight = tf.Variable(tf.random_normal([lstm_size, 1]))
bias = tf.Variable(tf.constant(0.0, shape=[1]))

# Predvidjena cena
prediction = tf.add(tf.matmul(last, weight), bias)

# MSE za predikciju
loss = tf.reduce_mean(tf.square(tf.subtract(prediction, Y)))

# Gradijentni spust pomocu Adam optimizacije
optimizer = tf.train.AdamOptimizer(L).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Treniranje modela
    for epoch in range(epoch_max):
        # Adaptiranje stope ucenja
        epoch_loss, alpha = 0, max(alpha_min, alpha_max * (1 - epoch / epoch_max))

        # Treniranje modela za odredjene akcije
        for sym_index, sym in enumerate(train_symbols):
            sym_feed = [[sym_index]] * batch_size

            # Mini batch gradijentni spust
            for b in np.random.permutation(nb_batches[sym_index]):
                feed = {X: train_x[sym_index][b], Y: train_y[sym_index][b], L: alpha, S: sym_feed}
                loss_val, _ = sess.run([loss, optimizer], feed)
                epoch_loss += loss_val

        print('Epoch: {}/{}\tLoss: {}'.format(epoch+1, epoch_max, epoch_loss))

    # Testiranje modela za odredjene akcije
    for sym in test_symbols:
        sym_index = train_symbols.index(sym)
        sym_feed = [[sym_index]] * nb_test[sym_index]
        feed = {X: test_x[sym_index], Y: test_y[sym_index], L: alpha, S: sym_feed}
        test_pred = sess.run(prediction, feed)

        # Tacnost modela za predikciju monotonosti fluktuacije cene
        acc = sum(1 for i in range(nb_test[sym_index]) if test_pred[i] * test_y[sym_index][i] > 0) / nb_test[sym_index]
        print('Accuracy ({}): {}'.format(sym, acc))

        # Denormalizacija podataka
        denorm_y = [(curr + 1.0) * norm_test_y[sym_index][i] for i, curr in enumerate(test_y[sym_index])]
        denorm_pred = [(curr + 1.0) * norm_test_y[sym_index][i] for i, curr in enumerate(test_pred)]

        # Prikazivanje predikcija
        plt.figure(figsize=(16,4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.plot(dates[sym_index][-nb_test[sym_index]:], denorm_y, '-b', label='Actual')
        plt.plot(dates[sym_index][-nb_test[sym_index]:], denorm_pred, '--r', label='Predicted')
        plt.gcf().autofmt_xdate()
        plt.title(sym)
        plt.legend()
        plt.show()
