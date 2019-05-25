from keras.datasets import imdb
from keras import models, layers, optimizers, metrics, losses
import numpy as np
import matplotlib.pyplot as plt

def decode_to_english(train_data):
    word_index = imdb.get_word_index()
    #swap dictionary key-value ordering
    reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
    print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
  """
    perform one-hot encoding
  """
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

def build_model(input_shape=10000):
  """
    Build a network of input_shape ->(relu) 16 ->(relu) 16 ->(sigmoid) 1 
  """
  model = models.Sequential()
  model.add(layers.Dense(units=16, activation='relu', input_shape=(input_shape,)))
  model.add(layers.Dense(units=16, activation='relu'))
  model.add(layers.Dense(units=1, activation='sigmoid'))
  model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
  return model

def build_validation_set(x_train, y_train, samples=10000):
  return x_train[samples:], y_train[samples:], x_train[:samples], y_train[:samples]

def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=512):
  return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

def plot_loss(history_dict):
  epochs = range(1, len(history_dict['binary_accuracy'])+1)

  plt.plot(epochs, history_dict['loss'], 'bo', label='Training loss')
  plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  plt.clf()

def plot_accuracy(history_dict):
  epochs = range(1, len(history_dict['binary_accuracy'])+1)
  acc_values = history_dict['binary_accuracy']
  val_acc_values = history_dict['val_binary_accuracy']

  plt.plot(epochs, acc_values, 'bo', label='Training acc')
  plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()
  plt.clf()

def example1(x_train, x_test, y_train, y_test, x_val, y_val):
  model = build_model()

  history = train_model(model, x_train, y_train, x_val, y_val)
  history_dict = history.history
  plot_loss(history_dict)
  plot_accuracy(history_dict)
  results = model.evaluate(x=x_test, y=y_test)
  print("Example 1:", list(zip(('Loss', 'Accuracy'), results)))

def example2(x_train, x_test, y_train, y_test, x_val, y_val):
  model = build_model()

  history = train_model(model, x_train, y_train, x_val, y_val, epochs=4)
  history_dict = history.history
  plot_loss(history_dict)
  plot_accuracy(history_dict)
  results = model.evaluate(x=x_test, y=y_test)
  print("Example 2:", list(zip(('Loss', 'Accuracy'), results)))

def main():
  (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
  x_train = vectorize_sequences(train_data)
  x_test = vectorize_sequences(test_data)

  y_train = np.asarray(train_labels).astype('float32')
  y_test = np.asarray(test_labels).astype('float32')
  x_train, y_train, x_val, y_val = build_validation_set(x_train, y_train)
  decode_to_english((train_data))
  example1(x_train, x_test, y_train, y_test, x_val, y_val)
  example2(x_train, x_test, y_train, y_test, x_val, y_val)

if __name__ == "__main__":
  main()
