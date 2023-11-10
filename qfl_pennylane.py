import numpy as np
import tensorflow as tf

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28*28) / 255.0
test_images = test_images.reshape(test_images.shape[0], 28*28) / 255.0

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import time

# Federated learning setup
num_devices = 10
device_list = []
dev = qml.device("default.qubit", wires=2)

class Device:
  def __init__(self, feats_train, y_train, feats_val, y_val,opt,  weights, bias, num_train, features, Y):
    self.dev = qml.device("default.qubit", wires=2)
    self.feats_train = feats_train
    self.Y_train = y_train
    self.feats_val = feats_val
    self.y_val = y_val
    self.opt = opt
    self.weights = weights
    self.bias = bias
    self.num_train = num_train
    self.features = features
    self.Y = Y

def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.CNOT(wires=[0, 1])

def statepreparation(a):
    qml.RY(a[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)
    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)

@qml.qnode(device=dev, interface="autograd")
def circuit(weights, angles):
  statepreparation(angles)
  for W in weights:
      layer(W)
  circ = qml.expval(qml.PauliZ(0))
  return circ

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss

def variational_classifier(weights, bias, angles):
  return circuit(weights, angles) + bias

def cost(weights, bias, X, Y):
  predictions = [variational_classifier(weights, bias, x) for x in X]
  return square_loss(Y, predictions)

def get_angles(x):
  beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
  beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
  beta2 = 2 * np.arcsin(
      np.sqrt(x[2] ** 2 + x[3] ** 2)
      / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
  )
  return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

data1 = np.loadtxt("iris_classes1and2_scaled.txt")
device_data = np.array_split(data1, num_devices)

num_qubits = 2
num_layers = 6
batch_size = 5

for i, data in enumerate(device_data):
  opt = NesterovMomentumOptimizer(0.01)
  X = data[:, 0:2]
  padding = 0.3 * np.ones((len(X), 1))
  X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
  normalization = np.sqrt(np.sum(X_pad ** 2, -1))
  X_norm = (X_pad.T / normalization).T
  features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
  Y = data[:, -1]
  np.random.seed(0)
  num_data1 = len(Y)
  num_train1 = int(0.75 * num_data1)
  index1 = np.random.permutation(range(num_data1))
  feats_train = features[index1[:num_train1]]
  Y_train = Y[index1[:num_train1]]
  feats_val = features[index1[num_train1:]]
  Y_val = Y[index1[num_train1:]]
  weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
  bias_init = np.array(0.0, requires_grad=True)
  device = Device(feats_train, Y_train, feats_val, Y_val, opt, weights_init, bias_init, num_train1, features, Y)
  device_list.append(device)

batch_size = 5

for it in range(20):
  start_time = time.time_ns()
  cost_array = []
  acc_train_array = []
  acc_val_array = []
  total_train_acc = 0
  total_val_acc = 0
  total_cost = 0

  for i,d in enumerate(device_list):
    batch_index = np.random.randint(0, d.num_train, (batch_size,))
    feats_train_batch = d.feats_train[batch_index]
    Y_train_batch = d.Y_train[batch_index]
    d.weights, d.bias, _, _ = d.opt.step(cost, d.weights, d.bias, feats_train_batch, Y_train_batch)
    predictions_train = [np.sign(variational_classifier(d.weights, d.bias, f)) for f in d.feats_train]
    predictions_val = [np.sign(variational_classifier(d.weights, d.bias, f)) for f in d.feats_val]
    acc_train = accuracy(d.Y_train, predictions_train)
    acc_val = accuracy(d.y_val, predictions_val)
    cost_value = cost(d.weights, d.bias, d.features, d.Y)
    cost_array.append(cost_value)
    acc_train_array.append(acc_train)
    acc_val_array.append(acc_val)
    print(
        "Comm: {:5d} | Device: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, i,  cost_value, acc_train, acc_val)
    )
    total_train_acc += acc_train
    total_val_acc += acc_val
    total_cost += cost_value

  acc_avg = total_train_acc/num_devices
  val_avg = total_val_acc/num_devices
  cost_avg = total_cost/num_devices
  print(
        "Average - Comm: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, cost_avg, acc_avg, val_avg)
    )
  total_time = time.time_ns() - start_time


# Reference:
# https://pennylane.ai/qml/demos/tutorial_variational_classifier/