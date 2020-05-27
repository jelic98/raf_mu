import math
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

# Hiperparametri
epoch_max = 10
alpha_max = 0.025
alpha_min = 0.001
batch_size = 32
window_size = 14
test_ratio = 0.1
max_time = 16
lstm_size = 64

# Ucitavanje podataka
csv = pd.read_csv('data/sp500.csv')
dates, data = csv['Date'].values, csv['Close'].values

# Konverzija datuma
dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
dates = [dates[i + max_time] for i in range(len(dates) - max_time)]

# Grupisanje podataka pomocu kliznog prozora
data = [data[i : i + window_size] for i in range(len(data) - window_size)]

# Normalizacija podataka
norm = [data[0][0]] + [data[i-1][-1] for i, _ in enumerate(data[1:])]
data = [curr / norm[i] - 1.0 for i, curr in enumerate(data)]

nb_samples = len(data) - max_time
nb_train = int(nb_samples * (1.0 - test_ratio))
nb_test = nb_samples - nb_train
nb_batches = math.ceil(nb_train / batch_size)

# Grupisanje podataka za propagaciju greske kroz vreme
x = [data[i : i + max_time] for i in range(nb_samples)]
y = [data[i + max_time][-1] for i in range(nb_samples)]

# Skup podataka za treniranje
train_x = [x[i : i + batch_size] for i in range(0, nb_train, batch_size)]
train_y = [y[i : i + batch_size] for i in range(0, nb_train, batch_size)]

# Skup podataka za testiranje
test_x, test_y = x[-nb_test:], y[-nb_test:]

# Skup podataka za denormalizaciju
norm_y = [norm[i + max_time] for i in range(nb_samples)]
norm_test_y = norm_y[-nb_test:]

tf.reset_default_graph()

# Cene tokom prethodnih dana
X = tf.placeholder(tf.float32, [None, max_time, window_size])

# Cena na trenutni dan
Y = tf.placeholder(tf.float32, [None])

# Stopa ucenja
L = tf.placeholder(tf.float32)

# LSTM sloj
rnn = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(lstm_size)])

# Izlaz LSTM sloja
val, _ = tf.nn.dynamic_rnn(rnn, X, dtype=tf.float32)
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

        # Mini batch gradijentni spust
        for b in np.random.permutation(nb_batches):
            loss_val, _ = sess.run([loss, optimizer], {X: train_x[b], Y: train_y[b], L: alpha})
            epoch_loss += loss_val

        print('Epoch: {}/{}\tLoss: {}'.format(epoch+1, epoch_max, epoch_loss))

    # Testiranje modela
    test_pred = sess.run(prediction, {X: test_x, Y: test_y, L: alpha})

    # Tacnost modela za predikciju monotonosti fluktuacije cene
    acc = sum(1 for i in range(nb_test) if test_pred[i] * test_y[i] > 0) / nb_test
    print('Accuracy: {}'.format(acc))

    # Denormalizacija podataka
    denorm_y = [(curr + 1.0) * norm_test_y[i] for i, curr in enumerate(test_y)]
    denorm_pred = [(curr + 1.0) * norm_test_y[i] for i, curr in enumerate(test_pred)]

    # Prikazivanje predikcija
    plt.figure(figsize=(16,4))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.plot(dates[-nb_test:], denorm_y, '-b', label='Actual')
    plt.plot(dates[-nb_test:], denorm_pred, '--r', label='Predicted')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
