
#TODO: set up hyperparams
# NUM_OUTPUTS = 3
# LEARNING_RATE = ??
# LOSS_FUNCTION = ??
# BATCH_SIZE = 100

class BezierModel:
  """ The main model that we plan to train and test for guessing shapes """
  def __init__(self):
    
    super(BezierModel, self).__init__()
    # TODO: set up params

    # self.loss = LOSS_FUNCTION
    # self.model = tf.keras.Sequential() # start of model
    # self.intToState = intToState  # dictionary of corresponding states
    # optimizerForModel = tf.keras.optimizers.Adam(LEARNING_RATE)

    # TODO: build the trainable cnn architecture

    # self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizerForModel, metrics=['categorical_accuracy'])

    # show summary of model architecture
    self.model.summary()
  
  def train(self, inputs, labels, epochs, callback):
    """ TODO: Train the model on input images with their corresponding shape labels """
    pass

  def test(self, inputs, labels):
    """ TODO: Test the model on unseen input images with their corresponding shape labels """
    pass