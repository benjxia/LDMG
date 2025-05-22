#!/usr/bin/python3
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='train.py',
                    description='Trains the latent diffusion model',
                    epilog='Good luck running my shit code - ben')
    parser.add_argument('filename')
    parser.add_argument('-d', '--device')     # option that takes a value
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag
    args = parser.parse_args()
    print(args.filename, args.count, args.verbose)
    pass