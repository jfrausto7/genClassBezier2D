import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalMaxPool2D
import numpy as np

# hyperparams
NUM_OUTPUTS = 3
LEARNING_RATE = 1e-4
LOSS_FUNCTION = tf.keras.losses.categorical_crossentropy
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

class BezierModel:
  """ The main model that we plan to train and test for guessing shapes """
  def __init__(self):
    super(BezierModel, self).__init__()
    self.loss = LOSS_FUNCTION
    self.model = tf.keras.Sequential() # start of model
    optimizerForModel = tf.keras.optimizers.Adam(LEARNING_RATE)

    # TODO: utilize resnet/vgg16..?

    # First block
    self.model.add(Conv2D(32, 5, 2, activation='relu', padding='same', input_shape=(256,256,1)))
    self.model.add(MaxPool2D(pool_size=(2,2), strides=2))
    self.model.add(Dropout(0.4))

    # Second block
    self.model.add(Conv2D(64, 5, 1, activation='relu', padding='valid'))
    self.model.add(MaxPool2D(pool_size=(2,2), strides=2))
    self.model.add(Dropout(0.4))

    # Third block
    self.model.add(Conv2D(128, 5, 1, activation='relu', padding='valid'))
    self.model.add(MaxPool2D(pool_size=(2,2), strides=2))
    self.model.add(Dropout(0.4))

    # Fourth block
    self.model.add(Conv2D(256, 5, 1, activation='relu', padding='valid'))
    self.model.add(MaxPool2D(pool_size=(2,2), strides=2))
    self.model.add(Dropout(0.4))

    # Final block
    self.model.add(GlobalMaxPool2D())
    self.model.add(Dense(256, activation='relu'))
    self.model.add(Dropout(0.4))
    self.model.add(Flatten())
    self.model.add(Dense(NUM_OUTPUTS, activation='softmax'))

    self.model.compile(loss=self.loss, optimizer=optimizerForModel, metrics=['categorical_accuracy'])

    # show summary of model architecture
    self.model.summary()
  
  def train(self, test_dataset, valid_dataset, epochs, ckpt_callback, tb_callback):
    """ Train the model on input images with their corresponding shape labels """
    test_dataset = test_dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    valid_dataset = valid_dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    train_result = self.model.fit(test_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=[ckpt_callback, tb_callback])
    idx = np.argmax(train_result.history['categorical_accuracy'],axis=0)
    accuracy = train_result.history['categorical_accuracy'][idx]
    loss = train_result.history['loss'][idx]
    return accuracy, loss

  def test(self, dataset):
    """ Test the model on unseen input images with their corresponding shape labels """
    dataset = dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    valid_result = self.model.evaluate(dataset)
    accuracy = valid_result[1]
    loss = valid_result[0]
    return accuracy, loss