from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#train_file = "EquimosisTrain.csv"
train_file = "EquimosisTrainingOrig.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_file),
                                           origin=train_file)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.],[0.],[0.],[0.],[0.],[0.],[0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 10 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(10,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=35)  # randomize
train_dataset = train_dataset.batch(5)

# View a single example entry from a batch
features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(6, activation="relu", input_shape=(10,)),  # input shape required
  tf.keras.layers.Dense(6, activation="relu"),
  tf.keras.layers.Dense(6)
])

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 20

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    # Track progress
    epoch_loss_avg(loss(model, x, y))  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  
  if epoch % 1 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

#test_file = "EquimosisTest.csv"
test_file = "EquimosisTestOrig.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_file),
                                  origin=test_file)

test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the function created earlier
test_dataset = test_dataset.shuffle(15)       # randomize
test_dataset = test_dataset.batch(5)           # use the same batch size as the training set

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

class_ids = ["Pocas horas a 2 días", "2 a 3 días", "3 a 6 días ","6 a 12 días","12 a 17 días","más de 17 días"]

predict_dataset = tf.convert_to_tensor([
    [0.672222222,0.7375,0.710227273,0.5,0.6,1,0,1,0,0.5,],
    [0.372222222,0.4375,0.510227273,0.5,0.6,1,0,1,0,0.5,],
    [0.616666667,0.7125,0.493181818,0,0.4,1,1,1,0,1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  #print (i,logits)
  class_idx = tf.argmax(logits).numpy()
  #print (class_idx)
  name = class_ids[class_idx]
  print("Example {} prediction: {}".format(i, name))
  
plt.show()