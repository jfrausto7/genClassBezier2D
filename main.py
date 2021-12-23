import argparse

def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our GeoGuessr model",
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


def main(args: argparse.Namespace) -> None:
  pass



if __name__ == "__main__":
  args = parse_args()
  main(args)