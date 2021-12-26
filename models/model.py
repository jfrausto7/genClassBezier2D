import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalMaxPool2D
import numpy as np
from typing import Counter

#TODO: set up hyperparams
NUM_OUTPUTS = 3
LEARNING_RATE = 0.01
LOSS_FUNCTION = tf.keras.losses.CategoricalCrossentropy()
BATCH_SIZE = 32

class BezierModel:
  """ The main model that we plan to train and test for guessing shapes """
  def __init__(self):
    super(BezierModel, self).__init__()
    self.loss = LOSS_FUNCTION
    self.model = tf.keras.Sequential() # start of model
    optimizerForModel = tf.keras.optimizers.Adam(LEARNING_RATE)

    # TODO: utilize resnet/vgg16

    # First block
    self.model.add(Conv2D(32, 2, 1, activation='relu', padding='same', input_shape=(256,256,3)))
    self.model.add(MaxPool2D(pool_size=(2,2), strides=2))
    self.model.add(Dropout(0.4))

    # Second block
    self.model.add(Conv2D(64, 2, 1, activation='relu', padding='valid'))
    self.model.add(MaxPool2D(pool_size=(2,2), strides=2))
    self.model.add(Dropout(0.4))

    # Final block
    self.model.add(GlobalMaxPool2D())
    self.model.add(Flatten())
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dropout(0.4))
    self.model.add(Dense(NUM_OUTPUTS, activation='softmax'))

    self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizerForModel, metrics=['categorical_accuracy'])

    # show summary of model architecture
    self.model.summary()
  
  def train(self, dataset, epochs, ckpt_callback, tb_callback):
    """ TODO: Train the model on input images with their corresponding shape labels """
    train_result = self.model.fit(dataset, epochs=epochs, callbacks=[ckpt_callback, tb_callback])
    accuracy = train_result.history['categorical_accuracy'][-1]
    loss = train_result.history['loss'][-1]
    return accuracy, loss

  def validate(self, dataset):
    """ TODO: Validate the model on unseen input images with their corresponding shape labels """
    valid_result = self.model.evaluate(dataset)
    accuracy = valid_result.history['categorical_accuracy'][-1]
    loss = valid_result.history['loss'][-1]
    return accuracy, loss

  def test(self, dataset):
    """ TODO: Test the model on unseen input images with their corresponding shape labels """
    preds = self.model.predict(dataset)

    # get the most occurring class
    classes = []
    for pred in preds:
      classes.append(np.argmax(pred, axis=0))
    counts = Counter(classes)
    predictions = counts.most_common()
    print(predictions)
    prediction = counts.most_common()[0][0]
    return prediction