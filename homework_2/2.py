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
alpha_max = 0.05
alpha_min = 0.001

num_steps = 30
window_size = 1
batch_size = 32
lstm_size = 64

embed_size = 2

test_ratio = 0.1

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

    nb_samples = len(data)

    # Konverzija datuma
    dts = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dts]
    dates.append([dts[i + num_steps] for i in range(nb_samples - num_steps)])

    # Grupisanje podataka pomocu kliznog prozora
    data = [data[i * window_size : (i + 1) * window_size] for i in range(nb_samples // window_size)]

    # Normalizacija podataka
    norm = [data[0][0]] + [data[i-1][-1] for i, _ in enumerate(data[1:])]
    data = [curr / norm[i] - 1.0 for i, curr in enumerate(data)]

    # Grupisanje podataka za propagaciju greske kroz vreme
    x = [data[i : i + num_steps] for i in range(nb_samples - num_steps)]
    y = [data[i + num_steps] for i in range(nb_samples - num_steps)]

    nb_train.append(int(len(x) * (1.0 - test_ratio)))
    nb_test.append(nb_samples - nb_train[sym_index] - num_steps)
    nb_batches.append(math.ceil(nb_train[sym_index] / batch_size))

    # Skup podataka za treniranje
    train_x.append([x[b * batch_size : (b + 1) * batch_size] for b in range(nb_batches[sym_index])])
    train_y.append([y[b * batch_size : (b + 1) * batch_size] for b in range(nb_batches[sym_index])])

    # Skup podataka za testiranje
    test_x.append(x[nb_train[sym_index]:])
    test_y.append(y[nb_train[sym_index]:])

    # Skup podataka za denormalizaciju
    norm_y = [norm[i + num_steps] for i in range(nb_samples - num_steps)]
    norm_test_y.append(norm_y[nb_train[sym_index]:])

tf.reset_default_graph()

# Cene tokom prethodnih dana
X = tf.placeholder(tf.float32, [None, num_steps, window_size])

# Cena na trenutni dan
Y = tf.placeholder(tf.float32, [None, window_size])

# Stopa ucenja
L = tf.placeholder(tf.float32, None)

# Celobrojna reprezentacija simbola akcija
S = tf.placeholder(tf.int32, [None, 1])

# Embedovanje simbola akcija
embed_matrix = tf.Variable(tf.random_uniform([nb_stocks, embed_size], -1.0, 1.0))
stack_symbols = tf.tile(S, [1, num_steps])
stack_embeds = tf.nn.embedding_lookup(embed_matrix, stack_symbols)

# Spajanje cena tokom prethodnih dana sa embedingom simbola akcija
inputs = tf.concat([X, stack_embeds], axis=2)

# LSTM sloj
lstm = [tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)]
rnn = tf.contrib.rnn.MultiRNNCell(lstm, state_is_tuple=True)

# Izlaz LSTM sloja
val, _ = tf.nn.dynamic_rnn(rnn, inputs, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])

# Poslednji izlaz LSTM sloja
last = tf.gather(val, val.get_shape()[0] - 1)

# Obucavajuci parametri
weight = tf.Variable(tf.truncated_normal([lstm_size, window_size]))
bias = tf.Variable(tf.constant(0.1, shape=[window_size]))

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
        epoch_loss = 0

        # Adaptiranje stope ucenja
        alpha = max(alpha_min, alpha_max * (1 - epoch / epoch_max))

        # Treniranje modela za odredjene akcije
        for sym_index, sym in enumerate(train_symbols):
            sym_feed = [[sym_index]] * batch_size

            # Mini batch gradijentni spust
            for b in np.random.permutation(nb_batches[sym_index]):
                feed = {X: train_x[sym_index][b], Y: train_y[sym_index][b], L: alpha, S: sym_feed}
                loss_val, _ = sess.run([loss, optimizer], feed)
                epoch_loss += loss_val

            print('Loss ({}): {}'.format(sym, epoch_loss))

        epoch_loss /= epoch_max
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
        test_y[sym_index] = [(curr + 1.0) * norm_test_y[sym_index][i] for i, curr in enumerate(test_y[sym_index])]
        test_pred = [(curr + 1.0) * norm_test_y[sym_index][i] for i, curr in enumerate(test_pred)]

        # Prikazivanje predikcija
        plt.figure(figsize=(16,4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=nb_test[sym_index]//16))
        plt.plot(dates[sym_index][nb_train[sym_index]:], test_y[sym_index], '-b', label='Actual')
        plt.plot(dates[sym_index][nb_train[sym_index]:], test_pred, '--r', label='Predicted')
        plt.gcf().autofmt_xdate()
        plt.title(sym)
        plt.legend()
        plt.show()