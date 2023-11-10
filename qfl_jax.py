import numpy as np
import tensorflow as tf
import jax.numpy as jnp
import jax
import optax
import random
import tensorcircuit as tc
from tqdm import tqdm
from hashlib import sha256
import time

n = 8
n_node = 8
device_set = {}
n_class = 3
k = 12
readout_mode = 'softmax'
K = tc.set_backend('jax')  
key = jax.random.PRNGKey(42)
tf.random.set_seed(42)


def filter(x, y, class_list):
    keep = jnp.zeros(len(y)).astype(bool)
    for c in class_list:
        print("c", c)
        keep = keep | (y == c)
    x, y = x[keep], y[keep]
    y = jax.nn.one_hot(y, n_node)
    return x, y

def clf(params, c, k): 
    for j in range(k):
        for i in range(n - 1):  
            c.cnot(i, i + 1)  
        for i in range(n):   
            c.rx(i, theta=params[3 * j, i])  
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c 

def readout(c):
    if readout_mode == 'softmax':
        logits = []
        for i in range(n_node):
            logits.append(jnp.real(c.expectation([tc.gates.z(), [i,]])))
        logits = jnp.stack(logits, axis=-1) * 10
        probs = jax.nn.softmax(logits)
    elif readout_mode == 'sample':
        wf = jnp.abs(c.wavefunction()[:n_node])**2
        probs = wf / jnp.sum(wf)
    return probs

def loss(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))
loss = K.jit(loss, static_argnums=[3])

def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)
accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])

def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs
pred = K.vmap(pred, vectorized_argnums=[1])

class Device:
    def __init__(self, id, data, params, opt_state):
        self.id = id
        self.data = data
        self.params = params
        self.opt_state = opt_state
        self.sk = None
        self.params_hash = None
        self.pk = None
        self.train_list = []
        self.train_loss = []
        self.signature = None
        self.hash_signature = None

def prepareData(dataset, encoding_mode): 
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()  
    x_train = x_train / 255.0   

    if encoding_mode == 'vanilla':
        mean = 0
    elif encoding_mode == 'mean':
        mean = jnp.mean(x_train, axis=0)
    elif encoding_mode == 'half':
        mean = 0.5
    x_train = x_train - mean
    x_train = tf.image.resize(x_train[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_train = x_train / jnp.sqrt(jnp.sum(x_train ** 2, axis=-1, keepdims=True))
    x_test = x_test / 255.0  
    x_test = x_test - mean
    x_test = tf.image.resize(x_test[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_test = x_test / jnp.sqrt(jnp.sum(x_test ** 2, axis=-1, keepdims=True))
    y_test = jax.nn.one_hot(y_test, n_node)
    return x_train, y_train, x_test, y_test


for node in range(n_node-1):
    x_train, y_train, x_test, y_test = prepareData("mnist", "vanilla")
    deviceId = node
    x_train_node, y_train_node = filter(x_train, y_train, [(node + i) % n_node for i in range(n_class)])
    data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
    y_train_cat = np.argmax(y_train_node, axis=1)
    key, subkey = jax.random.split(key)
    params = jax.random.normal(subkey, (3 * k, n))
    opt = optax.adam(learning_rate=1e-2)
    opt_state = opt.init(params)
    a_device = Device(deviceId, data, params, opt_state)
    device_set[node] = a_device

devices_list = list(device_set.values())

def workerTask(device, local_epochs, b):
    for epoch in tqdm(range(local_epochs), leave=False):
        for i, (x, y) in enumerate(device.data):
            x = x.numpy()
            y = y.numpy()
            loss_val, grad_val = compute_loss(device.params, x, y, k)
            updates, device.opt_state = opt.update(grad_val, device.opt_state, device.params)
            device.params = optax.apply_updates(device.params, updates)  
            device.params_hash = int.from_bytes(sha256(str(device.params).encode('utf-8')).digest(), byteorder='big')
            loss_mean = jnp.mean(loss_val)
            if i % 20 == 0:
                acc = jnp.mean(compute_accuracy(device.params, x, y, k))
                tqdm.write(f'world {b}, epoch {epoch}, {i}/{len(device.data)}: loss={loss_mean:.4f}, acc={acc:.4f}')
        print(f"Device {device.id} training Epoch: {epoch} done...")
    print(f"Device {device.id} training ALL EPOCHS done...")
 

def device_training(local_epochs, b):
    for device in devices_list:
        workerTask(device, local_epochs, b) 

avg_params = None
def serverTask(b):
    params_list = []
    for device in devices_list:
        params_list.append(device.params)
    avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)
    for device in devices_list:
        device.params = avg_params
    test_acc = jnp.mean(pred(avg_params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
    test_loss = -jnp.mean(jnp.log(pred(avg_params, x_test[:1024], k)) * y_test[:1024])
    tqdm.write(f'world {b}: test acc={test_acc:.4f}, test loss={test_loss:.4f}')

loss_list = []
acc_list = []

print("Start Communication Rounds")
for b in range(10):
    current_time = time.time_ns()
    print("Communication Round: ", b)
    print(f"COMM ROUND: {b} - Start training Device")
    device_training(1, b)
    print(f"COMM ROUND: {b} - Start Server Task")
    serverTask(b)
    final_time = time.time_ns() - current_time

# Reference:
# https://github.com/haimengzhao/quantum-fed-infer