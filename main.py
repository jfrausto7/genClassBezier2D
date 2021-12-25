import os
import argparse
import tensorflow as tf
tf.get_logger().setLevel('ERROR')   # errors only
import glob
import matplotlib.pyplot as plt

from shape_generation.generate_dataset import generate_dataset
from preprocess import preprocess


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
        '--validate',
        help='Validate',
        action='store_true',
        dest='validate'
    )

    parser.add_argument(
        '--generate',
        help='Generate dataset of abstract shapes from Bezier curves',
        action='store_true',
        dest='generate'
    )

    parser.add_argument(
        '--reset-data',
        help='Delete the scraped dataset images',
        action='store_true',
        dest='reset'
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

def augment_dataset(dataset):
    for image_batch, label_batch in dataset:
        image_batch = tf.map_fn(preprocess, image_batch)
        dataset = tf.data.Dataset.from_tensors((image_batch,label_batch))
    return dataset

def print_dataset(dataset):
    for i, l in dataset:
        for img in range(i.shape[0]):
            plt.imshow(i[img])
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
    assert(len(os.listdir(main_dir)) > 1), "No dataset generated!"
    latest = max(glob.glob(os.path.join(main_dir, '*/')), key=os.path.getmtime)
    return latest + 'shapes/', latest + 'textures/', latest + 'colors/'
    

def main(args: argparse.Namespace) -> None:

    if args.generate:
        shapes_dir, textures_dir, colors_dir = generate_dataset()
    else:
        shapes_dir, textures_dir, colors_dir = get_latest_dataset()

    ## load respective datasets

    # shapes only
    shape_train_ds, shape_valid_ds, shape_test_ds = create_datasets(shapes_dir)
    assert_class_num_equiv(shape_train_ds, shape_valid_ds, shape_test_ds)   # ensure that these datasets have same class number

    # shapes w/ textures
    texture_train_ds, texture_valid_ds, texture_test_ds = create_datasets(textures_dir)
    assert_class_num_equiv(texture_train_ds, texture_valid_ds, texture_test_ds) # ensure that these datasets have same class number

    # shapes w/ textures & colors
    color_train_ds, color_valid_ds, color_test_ds = create_datasets(colors_dir)
    assert_class_num_equiv(color_train_ds, color_valid_ds, color_test_ds)   # ensure that these datasets have same class number

    # augment data with cv2
    shape_train_ds, shape_valid_ds, shape_test_ds = augment_dataset(shape_train_ds), augment_dataset(shape_valid_ds), augment_dataset(shape_test_ds)
    texture_train_ds, texture_valid_ds, texture_test_ds = augment_dataset(texture_train_ds), augment_dataset(texture_valid_ds), augment_dataset(texture_test_ds)
    color_train_ds, color_valid_ds, color_test_ds = augment_dataset(color_train_ds), augment_dataset(color_valid_ds), augment_dataset(color_test_ds)

    # print_dataset(shape_train_ds)

    # TODO: instantiate model! test validate and train

if __name__ == "__main__":
  args = parse_args()
  main(args)