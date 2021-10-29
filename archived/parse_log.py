import pickle
import numpy as np
import argparse


def main(output_file):
    out = pickle.load(open(output_file, 'rb'))
    print('Exit code: %d, %d, %d.' % (
        np.sum(0 == np.array(out[0])), np.sum(1 == np.array(out[0])),
        np.sum(2 == np.array(out[0]))))
    idx = np.logical_or(1 == np.array(out[0]), 2 == np.array(out[0]))
    # print('Mean dist: %.4f' % np.mean(out[1]))
    # print('Init ub: %.4f' % np.mean(out[2]))
    print('Mean successful dist: %.4f' % np.mean(np.array(out[1])[idx]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse saved output file')
    parser.add_argument(
        'output_file', type=str, help='name of output file')
    args = parser.parse_args()
    main(args.output_file)
