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
alpha_max = 0.05
alpha_min = 0.001

window_size = 1
num_steps = 30
batch_size = 64

lstm_size = 128
num_layers = 1

test_ratio = 0.1
date_interval = 7

# Ucitavanje podataka
csv = pd.read_csv('data/sp500.csv')
dates, data = csv['Date'].values, csv['Close'].values

# Konverzija datuma
dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
dates = [dates[i + num_steps] for i in range(nb_samples - num_steps)]

nb_samples = len(data)

# Grupisanje podataka (window_size)
data = [data[i * window_size : (i + 1) * window_size] for i in range(nb_samples // window_size)]

# Normalizacija podataka
norm = [data[0][0]] + [data[i-1][-1] for i, _ in enumerate(data[1:])]
data = [curr / norm[i] for i, curr in enumerate(data)]

# Grupisanje podataka (num_steps)
x = [data[i : i + num_steps] for i in range(nb_samples - num_steps)]
y = [data[i + num_steps] for i in range(nb_samples - num_steps)]

nb_train = int(len(x) * (1.0 - test_ratio))
nb_batches = math.ceil(nb_train / batch_size)

# Skup podataka za treniranje
train_x = [x[b * batch_size : (b + 1) * batch_size] for b in range(nb_batches)]
train_y = [y[b * batch_size : (b + 1) * batch_size] for b in range(nb_batches)]

# Skup podataka za testiranje
test_x = x[nb_train:]
test_y = y[nb_train:]

# Skup podataka za denormalizaciju
norm_y = [norm[i + num_steps] for i in range(nb_samples - num_steps)]
norm_test_y = norm_y[nb_train:]

# Konstrukcija grafa
tf.reset_default_graph()

# Cene tokom prethodnih dana
X = tf.placeholder(tf.float32, [None, num_steps, window_size])

# Cena na trenutni dan
Y = tf.placeholder(tf.float32, [None, window_size])

# Stopa ucenja
L = tf.placeholder(tf.float32, None)

lstm = [tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)]
rnn = tf.contrib.rnn.MultiRNNCell(lstm, state_is_tuple=True)
        
val, _ = tf.nn.dynamic_rnn(rnn, X, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])

last = tf.gather(val, val.get_shape()[0] - 1)

weight = tf.Variable(tf.truncated_normal([lstm_size, window_size]))
bias = tf.Variable(tf.constant(0.1, shape=[window_size]))

prediction = tf.add(tf.matmul(last, weight), bias)

loss = tf.reduce_mean(tf.square(tf.subtract(prediction, Y)))
optimizer = tf.train.AdamOptimizer(L).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Treniranje modela
    for epoch in range(epoch_max):
        epoch_loss = 0

        # Adaptiranje stope ucenja
        alpha = max(alpha_min, alpha_max * (1 - epoch / epoch_max))

        # Mini batch gradijentni spust
        for b in np.random.permutation(nb_batches):
            feed = {X: train_x[b], Y: train_y[b], L: alpha}
            loss_val, _ = sess.run([loss, optimizer], feed)
            epoch_loss += loss_val

        epoch_loss /= epoch_max
        print('Epoch: {}/{}\tLoss: {}'.format(epoch+1, epoch_max, epoch_loss))

    # Testiranje modela
    test_pred = sess.run(prediction, {X: test_x, Y: test_y, L: alpha})

    # Denormalizacija podataka
    test_y = [curr * norm_test_y[i] for i, curr in enumerate(test_y)]
    test_pred = [curr * norm_test_y[i] for i, curr in enumerate(test_pred)]

    # Prikazivanje predikcija
    plt.figure(figsize=(16,4))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=date_interval))
    plt.plot(dates[nb_train:], test_y, '-b', label='Actual')
    plt.plot(dates[nb_train:], test_pred, '--r', label='Predicted')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()