import argparse

datasets = ['DAVIS2017', 'WAD']

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train mask refine module')
    parser.add_argument('dataset')

    args = parser.parse_args()

