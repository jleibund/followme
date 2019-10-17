#!../env/bin/python

"""
train.py

Trains a model

Usage:
    train.py --data <DIR> --name <NAME> --find <BOOL>

Options:
  --data <DIR>      data directory
  --name <NAME>   model name
  --find <BOOL>      find learning rate [default: False]
"""

import sys
import time

from docopt import docopt

import methods
import config

from trainers import train_categorical

def main():
    arguments = docopt(__doc__)
    print(arguments)
    data_dir = arguments['--data']
    model_name = arguments['--name']
    lr_find = arguments['--find']
    print(lr_find=='True')
    train_categorical(data_dir, model_name, lr_find=lr_find=='True')

if __name__ == "__main__":
    main()
