# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    print("x:",np.shape(x))
    print("w1:",np.shape(network['W1']))
    print("b1:",np.shape(network['b1']))

    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    print("z1:",np.shape(z1))
    print("w2:",np.shape(network['W2']))
    print("b2:",np.shape(network['b2']))

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    print("z2:",np.shape(z2))
    print("w3:",np.shape(network['W3']))
    print("b3:",np.shape(network['b3']))
    a3 = np.dot(z2, w3) + b3

    y = softmax(a3)
    print("y:",np.shape(y))
    return y


x, t = get_data()
network = init_network()

batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    # print (i,len(x))
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
