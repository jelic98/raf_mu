import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Hiperparametri
k = 3
nb_classes = 2
nb_features = 4
class_trans = {'B': 0, 'M': 1, 0: 'B', 1: 'M'}

# Ucitavanje podataka
filename = 'data/Prostate_Cancer.csv'
all_data = np.genfromtxt(filename, dtype='str', delimiter=',', skip_header=1, usecols=range(1, 6))

# Nasumicno mesanje podataka
nb_samples = all_data.shape[0]
indices = np.random.permutation(nb_samples)
all_data = all_data[indices]

# Konvertovanje podataka u brojeve
for row in all_data:
    row[0] = class_trans[row[0]]
all_data = [[0 if x == 'NA' else float(x) for x in row] for row in all_data]

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
w = tf.fill([k], 1/k)

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

    plt.figure(figsize=(6, 6))

    idxs = np.array(train_y) == 0.0

    # Grafik koji prikazuje ulazne podatke
    min_ax1, max_ax1 = min(train_x[:, 1]), max(train_x[:, 1])
    min_ax2, max_ax2 = min(train_x[:, 2]), max(train_x[:, 2])
    plt.scatter(train_x[idxs, 1], train_x[idxs, 2], c='g', label='Benign')
    plt.scatter(train_x[~idxs, 1], train_x[~idxs, 2], c='b', label='Malignant')

    # Konstruisanje mreze koja pokriva ceo grafik
    x1, x2 = np.meshgrid(np.arange(min_ax1 - 1, max_ax1 + 1, 0.15),
                         np.arange(min_ax2 - 1, max_ax2 + 1, 1.0))

    # Adaptacija ulaznih podataka za dvodimenzionalni prikaz
    plot_x = np.array([[0, x[1], x[2], 0] for x in train_x])

    # Grafik koji pokazuje oblasti koje bivaju klasifikovane u svaku od klasa
    hyps = np.zeros((x1.shape[0], x1.shape[1]), dtype=np.int8)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            plot_q = np.array([0, x1[i][j], x2[i][j], 0])
            feed = {X: plot_x, Y: train_y, Q: plot_q}
            hyps[i][j] = sess.run(hyp, feed_dict = feed)
    cmap = LinearSegmentedColormap.from_list('cmap', ['green', 'blue'])
    plt.contourf(x1, x2, hyps, cmap=cmap, alpha=0.5)

plt.xlim([min_ax1, max_ax1])
plt.ylim([min_ax2, max_ax2])
plt.show()
