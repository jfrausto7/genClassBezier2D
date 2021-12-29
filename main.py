import os
import argparse
from types import TracebackType
from matplotlib import image
import tensorflow as tf
from tensorflow._api.v2 import data
import cv2

from models.model import BezierModel
tf.get_logger().setLevel('ERROR')   # errors only
import glob
import matplotlib.pyplot as plt
from statistics import mean

from shape_generation.generate_dataset import generate_dataset, generate_specific
from preprocess import preprocess, preprocess_with_hed
from hed import CropLayer

def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our BezierModel model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--train',
        help='Train',
        action='store_true',
        dest='train'
    )

    parser.add_argument(
        '--test',
        help='Test',
        action='store_true',
        dest='test'
    )

    parser.add_argument(
        '--dataset',
        help='Select a dataset to use',
        default='shapes',
        dest='dataset'
    )

    parser.add_argument(
        '--generate',
        help='Generate dataset of abstract shapes from Bezier curves',
        action='store_true',
        dest='generate'
    )

    parser.add_argument(
        '--weights',
        default='',
        help='Path for the weights to use in training/testing',
        dest='weights'
    )

    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='Number of training epochs',
        dest='epochs'
    )

    parser.add_argument(
        '--hed',
        help='Utilize holistically-nested edge detection over normal data augmentation',
        action='store_true',
        dest='hed'
    )

    parser.add_argument(
        '--specific',
        help='Special func',
        default='',
        dest='specific'
    )

    return parser.parse_args()

def load_dataset(path_to_dataset, subset=None, valid_split=None):
    if subset is "training":
        valid_split = 0.2
    else:
        valid_split = 0.1

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
    path_to_dataset,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=valid_split,
    subset=subset,
    interpolation="bilinear",
    crop_to_aspect_ratio=True
    )

    return dataset

def create_datasets(dir):
    train, valid, test = load_dataset(dir, subset = "training"), load_dataset(dir,subset = "validation"), load_dataset(dir, subset='validation')
    return train, valid, test

def process_dataset(dataset):
    image_tensors, label_tensors = [], []
    for image_batch, label_batch in dataset:
        if args.hed:
            print("Performing holistically-nested edge detection...")
            image_batch = tf.map_fn(preprocess_with_hed, image_batch)
            image_tensors.append(image_batch)
            label_tensors.append(label_batch)
        else:
            image_batch = tf.map_fn(preprocess, image_batch)
            image_tensors.append(image_batch)
            label_tensors.append(label_batch)

    image_tensors = tf.concat(image_tensors, axis=0)
    label_tensors = tf.concat(label_tensors, axis=0)
    image_tensors = tf.expand_dims(image_tensors, axis=-1)
    label_tensors = tf.expand_dims(label_tensors, axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((image_tensors,label_tensors))
    return dataset

def print_dataset(dataset):
    for img, lbl in dataset:
        plt.imshow(img)
        plt.show()
        

def assert_class_num_equiv(training_ds, validation_ds, test_ds):
    # ensure that these datasets have same class number
    equivalence_check = training_ds.class_names == validation_ds.class_names == test_ds.class_names
    assert_fail_message = "Training, Validation, and Test classes should match"
    assert(equivalence_check), assert_fail_message
    class_names = training_ds.class_names
    number_classes = len(class_names)
    print("There are " + str(number_classes) + " classes.")

def get_latest_dataset():
    main_dir = './data'
    assert(len(os.listdir(main_dir)) > 0), "No dataset generated!"
    latest = max(glob.glob(os.path.join(main_dir, '*/')), key=os.path.getmtime)
    return latest + 'shapes/', latest + 'textures/', latest + 'colors/'

def train(model, tr_dataset, vl_dataset, epochs, weightPath):

    print("Now training BezierModel")

    # use checkpoint if specified 
    if weightPath != '':
        if(os.path.exists(os.path.join(weightPath, "saved_model.pb"))):
            print("Loading old weights...")
            model.model = tf.keras.models.load_model(weightPath)
            print("Loaded old weights! Will continue training!")

    checkpoint_path = "./checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    # callback to save model weights
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False,
    save_freq='epoch'
    )

    tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=False, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )

    # Train
    acc, loss = model.train(tr_dataset, vl_dataset, epochs, model_checkpoint_callback, tb_callback)

    print(f"Best Accuracy: {(100*acc):>0.1f}%, Loss: {loss:>8f} \n")

    return model

def test(model, dataset, weightPath):

    print("Now testing BezierModel")

    # use checkpoint if specified 
    if weightPath != '':
        if(os.path.exists(os.path.join(weightPath, "saved_model.pb"))):
            print("Loading old weights...")
            model.model = tf.keras.models.load_model(weightPath)
            print("Loaded old weights! Will test on it!")

    accs,losses = [], []

    for _ in range(10):
        # Test
        acc, loss = model.test(dataset)
        accs.append(acc)
        losses.append(loss)

    avgAcc = mean(accs)
    avgLoss = mean(losses)

    print(f"Accuracy: {(100*avgAcc):>0.1f}%, Loss: {avgLoss:>8f} \n")
    

def main(args: argparse.Namespace) -> None:

    if args.specific != '':
        print("continuing")
        generate_specific(args.specific, 3235)

    if args.generate:
        shapes_dir, textures_dir, colors_dir = generate_dataset()
    else:
        shapes_dir, textures_dir, colors_dir = get_latest_dataset()

    if args.hed:
        # register our new layer with the CropLayer model (for HED)
        cv2.dnn_registerLayer("Crop", CropLayer)

    ## load respective datasets

    if args.dataset == 'shapes':
        # shapes only
        shape_train_ds, shape_valid_ds, shape_test_ds = create_datasets(shapes_dir)
        assert_class_num_equiv(shape_train_ds, shape_valid_ds, shape_test_ds)   # ensure that these datasets have same class number
        # data augmentation with cv2
        train_ds, val_ds, test_ds = process_dataset(shape_train_ds), process_dataset(shape_valid_ds), process_dataset(shape_test_ds)

    elif args.dataset == 'textures':
        # shapes w/ textures
        texture_train_ds, texture_valid_ds, texture_test_ds = create_datasets(textures_dir)
        assert_class_num_equiv(texture_train_ds, texture_valid_ds, texture_test_ds) # ensure that these datasets have same class number
        # data augmentation with cv2
        train_ds, val_ds, test_ds = process_dataset(texture_train_ds), process_dataset(texture_valid_ds), process_dataset(texture_test_ds)

    elif args.dataset == 'colors':
        # shapes w/ textures & colors
        color_train_ds, color_valid_ds, color_test_ds = create_datasets(colors_dir)
        assert_class_num_equiv(color_train_ds, color_valid_ds, color_test_ds)   # ensure that these datasets have same class number
        # data augmentation with cv2
        train_ds, val_ds, test_ds = process_dataset(color_train_ds), process_dataset(color_valid_ds), process_dataset(color_test_ds)
    
    else:
        print("Dataset specified is invalid.")
        return

    # print_dataset(train_ds)

    #instantiate model
    bezierModel = BezierModel()

    if args.train:
        # train the model
        bezierModel = train(bezierModel, train_ds, val_ds, args.epochs, args.weights)
    
    if args.test:
        # test the model
        test(bezierModel, test_ds, args.weights)

if __name__ == "__main__":
  args = parse_args()
  main(args)