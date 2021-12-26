import cv2
import os

class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass)
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derviced (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                                self.startX:self.endX]]


def detect_edges(img):
  # load our serialized edge detector from disk
  protoPath = "./models/hed_model/deploy.prototxt"
  modelPath = "./models/hed_model/hed_pretrained_bsds.caffemodel"

  net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)  # model

  image = img
  (H, W) = image.shape[:2]

  # construct a blob out of the input image for the Holistically-Nested
  # Edge Detector
  blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                              mean=(104.00698794, 116.66876762, 122.67891434),
                              swapRB=False, crop=False)

  # set the blob as the input to the network and perform a forward pass
  # to compute the edges
  net.setInput(blob)
  hed = net.forward()
  hed = cv2.resize(hed[0, 0], (W, H))
  hed = (255 * hed).astype("uint8")

  return hed