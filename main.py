from script.train import train_net
from script.test import test_net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train-net', default='n')
parser.add_argument('--test-net', default='y')
parser.add_argument('--generate-data', default='n')

args = parser.parse_args()


if __name__ == '__main__':
    if args.train_net == 'y':
        train_net()
    elif args.test_net == 'y':
        test_net()
    elif args.generate_data == 'y':
        pass