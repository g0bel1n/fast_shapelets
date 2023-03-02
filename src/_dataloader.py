import csv
import os
import pickle

import numpy as np
import requests

__supported_datasets__ = ["StarLightCurves"]


def download_dataset(dataset_name: str):
    """
    Download a dataset from the internet and save it to disk

    :param dataset_name: The name of the dataset to download
    :type dataset_name: str
    """

    os.mkdir(f"data/{dataset_name}")
    dl_url = f"http://www.timeseriesclassification.com/Downloads/{dataset_name}.zip"
    print(f"Downloading {dataset_name} dataset from {dl_url.split('/')[2]} \n")
    r = requests.get(dl_url, allow_redirects=True)
    open(f"data/{dataset_name}.zip", "wb").write(r.content)
    print(f"Unzipping {dataset_name} dataset \n")
    os.system(f"unzip data/{dataset_name}.zip -d data/{dataset_name}/")
    os.system(f"rm data/{dataset_name}.zip")

    print(f"Converting {dataset_name} dataset to pickle \n")
    with open(f"data/{dataset_name}/{dataset_name}_TRAIN.txt", "r") as train_file:
        train_data = csv.reader(train_file.readlines(), delimiter="\t")
        train_data = [
            [float(el_) for el_ in el[0].split(" ") if el_ != ""]
            for el in list(train_data)
        ]

    with open(f"data/{dataset_name}/{dataset_name}_TEST.txt", "r") as test_file:
        test_data = csv.reader(test_file.readlines(), delimiter="\t")
        test_data = [
            [float(el_) for el_ in el[0].split(" ") if el_ != ""]
            for el in list(test_data)
        ]

    with open(f"data/{dataset_name}/{dataset_name}_TEST.pkl", "wb") as test_file:
        pickle.dump(test_data, test_file)

    with open(f"data/{dataset_name}/{dataset_name}_TRAIN.pkl", "wb") as train_file:
        pickle.dump(train_data, train_file)

    os.system(f"rm data/{dataset_name}/{dataset_name}_TRAIN.txt")
    os.system(f"rm data/{dataset_name}/{dataset_name}_TEST.txt")
    os.system(f"rm data/{dataset_name}/{dataset_name}_TRAIN.ts")
    os.system(f"rm data/{dataset_name}/{dataset_name}_TEST.ts")
    os.system(f"rm data/{dataset_name}/{dataset_name}_TRAIN.arff")
    os.system(f"rm data/{dataset_name}/{dataset_name}_TEST.arff")

    print(f"Done converting {dataset_name} dataset to pickle \n")


def assert_root_dir(root_dir: str = "fast_shapelets"):
    """
    It checks that the current working directory is the root directory of the project

    :param root_dir: The directory where the data is stored, defaults to fast_shapelets
    :type root_dir: str (optional)
    """

    if root_dir in os.path.abspath(os.curdir).split("/")[-1]:
        pass
    elif root_dir in os.path.abspath(os.curdir).split("/"):
        os.chdir("..")
        assert_root_dir()

    else:
        raise Exception("Please run this script from the root directory of the project")


def get_dataset(dataset_name: str) -> tuple:
    """
    > It downloads the dataset if it doesn't exist, and then loads the train and test data from the
    pickle files

    :param dataset_name: The name of the dataset you want to download
    :type dataset_name: str
    :return: The train and test data for the dataset.
    """

    assert_root_dir()

    assert dataset_name in __supported_datasets__, "Dataset not supported"

    if not os.path.exists(f"data/{dataset_name}"):
        download_dataset(dataset_name)
    else:
        print(f"Dataset {dataset_name} loading from cache \n")

    with open(f"data/{dataset_name}/{dataset_name}_TRAIN.pkl", "rb") as train_file:
        train_data = np.array(pickle.load(train_file))

    with open(f"data/{dataset_name}/{dataset_name}_TEST.pkl", "rb") as test_file:
        test_data = np.array(pickle.load(test_file))

    return train_data[:, 1:], train_data[:, 0], test_data[:, 1:], test_data[:, 0]
