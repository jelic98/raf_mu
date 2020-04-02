import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorboardcolab import *

# Pomocna funkcija koja od niza trening primera pravi feature matricu
def create_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features+1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)

# Ucitavanje podataka
filename = 'data/corona.csv'
all_data = np.loadtxt(filename, delimiter=',')
data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]

# Nasumicno mesanje podataka
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

# Cuvanje domena i kodomena ulaznih podataka
min_x = min(data['x'])
max_x = max(data['x'])
min_y = min(data['y'])
max_y = max(data['y'])

data_loss = []

nb_epochs = 100

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)

lmbds = [0, 0.001, 0.01, 0.1, 1, 10, 100]

for i, lmbd in enumerate(lmbds):
    data_x = data['x'][:]
    data_y = data['y'][:]

    data_loss.append([])

    tf.reset_default_graph()

    # Kreiranje feature matrice
    nb_features = 3
    data_x = create_feature_matrix(data_x, nb_features)

    # Ulazni podaci
    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=(None), dtype=tf.float32, name='Y')
    Y_col = tf.reshape(Y, (-1, 1), name='Y_col')

    # Koeficijenti polinoma
    w = tf.Variable(tf.zeros(nb_features), name='w')
    bias = tf.Variable(0.0, name='bias')

    # Matrica parametara
    w_col = tf.reshape(w, (nb_features, -1), name='w_col')

    # Hipoteza
    hyp = tf.add(tf.matmul(X, w_col), bias, name='hyp')

    # Funkcija greske
    mse = tf.reduce_mean(tf.square(hyp - Y_col), name='mse')

    # Regularizacija
    l2_reg = lmbd * tf.reduce_mean(tf.square(w), name='l2_reg')
    loss = tf.add(mse, l2_reg, name='loss')

    # Optimizacija
    opt_op = tf.train.AdamOptimizer().minimize(loss)

    # Ispisivanje log fajla za TensorBoard
    if i == 0:
        tbc = TensorBoardColab()
        writer = tbc.get_writer()
        writer.add_graph(tf.get_default_graph())
        writer.flush()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Trening
        for epoch in range(nb_epochs):
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: data_x[sample].reshape((1, nb_features)), Y: data_y[sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                epoch_loss += curr_loss
            epoch_loss /= nb_samples
            data_loss[i].append(epoch_loss)
            if epoch % 10 == 0:
                print('{}/{} : {:.5f}'.format(epoch+1, nb_epochs, epoch_loss))
 
        # Racunanje finalnih vrednosti parametara i funkcije greske
        w_val = sess.run(w)
        bias_val = sess.run(bias)
        loss_val = sess.run(mse, feed_dict={X: data_x, Y: data_y})
        print('w = {}, bias = {:.5f}, loss {:.5f}'.format(w_val, bias_val, loss_val))

        # Grafik koji prikazuje funkciju koja opisuje ulazne podatke
        xs = create_feature_matrix(np.linspace(min_x, max_x, 100), nb_features)
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        stage = i / len(lmbds)
        plt.plot(xs[:,0].tolist(), hyp_val.tolist(), color=(1-stage, stage, 0))

# Grafik koji prikazuje ulazne podatke
plt.scatter(data['x'], data['y'])
plt.xlim([min_x, max_x])
plt.ylim([min_y, max_y])

# Grafik koji prikazuje zavisnost funkcije troska od stepena polinoma
plt.subplot(1, 2, 2)
for i, lmbd in enumerate(lmbds):
    stage = i / len(lmbds)
    plt.plot(range(1, nb_epochs+1), data_loss[i], color=(1-stage, stage, 0))
plt.show()

# ODGOVOR 2B
# Očigledno je da se sa povećanjem vrednosti lambda parametra,
# povećava i vrednost funkcije greške zato što u tom slučaju
# sabirak za L2 regularizaciju imati značajno veću vrednost, tj.
# kompleksnost modela će više uticati na povećanje vrednosti funkcije troška.
