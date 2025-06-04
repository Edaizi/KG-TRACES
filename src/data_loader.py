import string
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset, concatenate_datasets, Dataset

        

def load_multiple_datasets(data_path_list, shuffle=False):
    '''
    Load multiple datasets from different paths.

    Args:
        data_path_list (_type_): _description_
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    '''
    dataset_list = [load_dataset('json', data_files=p, split="train")
                     for p in data_path_list]
    dataset = concatenate_datasets(dataset_list)
    if shuffle:
        dataset = dataset.shuffle()
    return dataset




