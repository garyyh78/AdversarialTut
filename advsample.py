import json
import os
import tarfile
import tempfile
from urllib.request import urlretrieve

import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

DIM = 299
PIXEL_DEPTH = 8
PIXEL_RANGE = 2 ** PIXEL_DEPTH - 1
RGB = 3
ORIGINAL_CLASS = 281  # "cat"
TARGET_CLASS = 924  # "guacamole"
MAX_PIXEL_OFF = 3  # 3 pixel off allowed


# TF Inception graph wiring, main purpose is to expose the logits output
# note slim only provides the graph, data file will be downloaded next
def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs


tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()
image = tf.Variable(tf.zeros((DIM, DIM, RGB)))
logits, probs = inception(image, reuse=False)

# inception 3 model download to temp location and restore
data_dir = tempfile.mkdtemp()
print("temp dir = " + data_dir + "\n")
inception_tarball, _headers = urlretrieve(
    'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))

# loading labels
imagenet_json, _headers = urlretrieve(
    'http://www.anishathalye.com/media/2017/07/25/imagenet.json')
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)
print("size of imagenet label = %d" % (len(imagenet_labels)))


# main helper to show graph
def classify(title, img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]  # model output
    ax1.imshow(img)
    fig.sca(ax1)

    top_k = list(p.argsort()[-10:][::-1])
    top_probs = p[top_k]
    barlist = ax2.bar(range(10), top_probs)
    if target_class in top_k:
        barlist[top_k.index(target_class)].set_color('r')
    if correct_class in top_k:
        barlist[top_k.index(correct_class)].set_color('g')

    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in top_k],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.title(title)
    plt.show()


def textClassify(img, k=10):
    p = sess.run(probs, feed_dict={image: img})[0]  # model output
    top_k = list(p.argsort()[-k:][::-1])
    top_probs = p[top_k]

    np.set_printoptions(precision=4)
    print(top_k, "\n")
    print(top_probs, "\n")


# quick test with resizing
img_path, _headers = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
img_class = ORIGINAL_CLASS  # 281 is cat
img = PIL.Image.open(img_path)

print('raw image size = w %d, h %d' % (img.width, img.height))
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = DIM if not wide else int(img.width * DIM / img.height)
new_h = DIM if wide else int(img.height * DIM / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, DIM, DIM))
img = (np.asarray(img) / PIXEL_RANGE).astype(np.float32)
# classify( "original", img, correct_class = img_class )
textClassify(img)

# ----------- adversarial part
x = tf.placeholder(tf.float32, (DIM, DIM, RGB))

x_hat = image  # our trainable adversarial input
assign_op = tf.assign(x_hat, x)
learning_rate = tf.placeholder(tf.float32, ())

# objective is to train to y_hat
y_hat = tf.placeholder(tf.int32, ())  # tgt class name
labels = tf.one_hot(y_hat, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])

# maybe a better tgt, train to a point that it is completely un-classifible ???
# u_labels = tf.ones([1000], tf.float32) * 1e-3
# loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[u_labels])

optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

# noise bound
epsilon = tf.placeholder(tf.float32, ())
below = x - epsilon
above = x + epsilon

# clipped perturbation
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)

demo_epsilon = MAX_PIXEL_OFF / PIXEL_RANGE
demo_lr = 1e-1
demo_steps = 300
demo_stepsize = 10
demo_target = TARGET_CLASS

# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i + 1) % demo_stepsize == 0:
        print('step %02d: loss = %.6f' % (i + 1, loss_value))

adv = x_hat.eval()  # retrieve the adversarial example
# classify("grad attack", adv, correct_class=img_class, target_class=demo_target)
textClassify(adv)
