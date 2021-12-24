import os
import argparse
import tensorflow as tf
tf.get_logger().setLevel('ERROR')   # errors only
import glob

from shape_generation.generate_dataset import generate_dataset


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
    if subset:
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
    crop_to_aspect_ratio=True,
    )

    return dataset

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
        # TODO: call generate file
        shapes_dir, textures_dir, colors_dir = generate_dataset()
    else:
        shapes_dir, textures_dir, colors_dir = get_latest_dataset()

    
    # load respective datasets

    # shapes only
    shape_training_ds = load_dataset(
    shapes_dir,
    subset = "training"
    )
    shape_validation_ds = load_dataset(
    shapes_dir,
    subset = "validation"
    )
    shape_test_ds = load_dataset(
    shapes_dir
    )

    # ensure that these datasets have same class number
    assert_class_num_equiv(shape_training_ds, shape_validation_ds, shape_test_ds)

    # shapes w/ textures
    texture_training_ds = load_dataset(
    textures_dir,
    subset = "training"
    )
    texture_validation_ds = load_dataset(
    textures_dir,
    subset = "validation"
    )
    texture_test_ds = load_dataset(
    textures_dir
    )

    # ensure that these datasets have same class number
    assert_class_num_equiv(texture_training_ds, texture_validation_ds, texture_test_ds)

    # shapes w/ textures & colors
    color_training_ds = load_dataset(
    colors_dir,
    subset = "training"
    )
    color_validation_ds = load_dataset(
    colors_dir,
    subset = "validation"
    )
    color_test_ds = load_dataset(
    colors_dir
    )

    # ensure that these datasets have same class number
    assert_class_num_equiv(color_training_ds, color_validation_ds, color_test_ds)



if __name__ == "__main__":
  args = parse_args()
  main(args)