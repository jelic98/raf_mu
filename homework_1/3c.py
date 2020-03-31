import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

k_min = 1
k_max = 15

# Hiperparametri
nb_classes = 2
nb_features = 8
class_trans = {'B': 0, 'M': 1, 0: 'B', 1: 'M'}

# Ucitavanje podataka
filename = 'data/Prostate_Cancer.csv'
cols = range(1, 10)
all_data = np.genfromtxt(filename, dtype='str', delimiter=',', skip_header=1, usecols=cols)

# Nasumicno mesanje podataka
nb_samples = all_data.shape[0]
indices = np.random.permutation(nb_samples)
all_data = all_data[indices]

# Konvertovanje podataka u brojeve
for row in all_data:
    row[0] = class_trans[row[0]]
all_data = [[x if isinstance(x, float) else 0 for x in row] for row in all_data]
all_data = [[float(x) for x in row] for row in all_data]

# Deljenje podataka na skup za treniranje i testiranje
train, test = np.split(all_data, [int(len(all_data)*0.8)])

train_y, train_x = zip(*[(x[0], x[1:].tolist()) for x in train])
test_y, test_x = zip(*[(x[0], x[1:].tolist()) for x in test])

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

nb_train = len(train)
nb_test = len(test)

ax_k = []
ax_acc = []

for k in range(k_min, k_max+1):
    tf.reset_default_graph()

    # Ulazni podaci
    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.int32)

    # Upiti
    Q = tf.placeholder(shape=(nb_features), dtype=tf.float32)

    # Racunanje K minimalnih udaljenosti
    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(X, Q)), axis=1))
    _, idxs = tf.nn.top_k(-dists, k)
    classes = tf.gather(Y, idxs)
    dists = tf.gather(dists, idxs)
    w = 1 / dists

    # Odabir klase na osnovu frekvencije u skupu najblizih objekata
    w_col = tf.reshape(w, (k, 1))
    one_hot = tf.one_hot(classes, nb_classes)
    scores = tf.reduce_sum(w_col * one_hot, axis=0)
    hyp = tf.argmax(scores)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Trening
        matches = 0
        for i in range(nb_test):
            feed = {X: train_x, Y: train_y, Q: test_x[i]}
            hyp_val = sess.run(hyp, feed_dict = feed)
            act_val = test_y[i]
            match = hyp_val == act_val
            matches += 1 if match else 0
            print('{}/{} Predicted: {} Actual: {} Match: {}'
            .format(i+1, nb_test, class_trans[hyp_val], class_trans[act_val], match))

        print('Examples: {} Matches: {}'.format(nb_test, matches))

        # Racunanje tacnosti modela
        accuracy = matches / nb_test
        print('Accuracy: ', accuracy)

        ax_k.append(k)
        ax_acc.append(accuracy)

# Grafik koji prikazuje zavisnost funkcije troska od stepena polinoma
plt.plot(ax_k, ax_acc)
plt.show()

# ODGOVOR 3C
# Na grafiku se može videti horizontalna linija koja pokazuje
# da je tačnost modela 100%, nezavisno od vrednosti parametra K.
# Ovo pojava se dešava kada su klase dobro grupisane.
# Ipak, tvrđenje iz prethodog dela zadatka ostaje da važi
# zato što bi se povećanjem vrednosti parametra K u jednom trenutku
# primetilo opadanje tačnosti modela zato što bi veličina klase bila manja
# od broja susednih objekata koje uzimamo u obzir pri određivanju
# klase objekta iz skupa podataka za testiranje.
