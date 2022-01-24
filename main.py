from dataloader import WICO
import tensorflow as tf
from model import Net
from spektral.data import DisjointLoader
from tensorflow import keras
from tqdm import tqdm
import numpy as np

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy

def train_step(inputs, target):

    with tf.GradientTape() as tape:
        predictions, _, _ = model(X[:2])  # drop edge features from inputs
        loss = loss_fn(target, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc

if __name__ == "__main__":
    data = WICO("./dataset/WICO/", root_edges=False, time_delay_edges=False)

    np.random.default_rng(0)
    np.random.shuffle(data)

    train_pct = 0.2
    val_pct = 0.1
    num_epochs = 5

    n = len(data)
    train_size, val_size = int(n*train_pct), int(val_pct*n)

    train_set = data[:train_size]
    val_set = data[train_size:(train_size + val_size)]
    test_set = data[(train_size + val_size):]

    train_loader = DisjointLoader(train_set, batch_size=1, epochs=num_epochs)
    val_loader = DisjointLoader(val_set, batch_size=1, epochs=1)
    test_loader = DisjointLoader(test_set, batch_size=1, epochs=1)

    model = Net(num_classes=3)
    loss_fn = CategoricalCrossentropy(from_logits=True)
    optimizer = Adam(learning_rate=0.001)

    epoch = step = 0
    results = []

    # TRAINING
    for batch in tqdm(train_loader):
        step += 1
        loss, acc = train_step(*batch)
        results.append((loss, acc))

        if step == train_loader.steps_per_epoch:
            step = 0
            epoch += 1
            print("Ep. {} - Loss: {}. Acc: {}".format(epoch, *np.mean(results, 0)))
            results = []
    
    results = []

    # TESTING
    for batch in test_loader:
        inputs, target = batch
        predictions, _, _ = model(inputs[:2], training=False)
        results.append(
            (
                loss_fn(target, predictions),
                tf.reduce_mean(categorical_accuracy(target, predictions)),
            )
        )
    print("Done. Test loss: {}. Test acc: {}".format(*np.mean(results, 0)))