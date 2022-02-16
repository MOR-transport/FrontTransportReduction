from os.path import dirname, join
import os

MAIN_DIRECTORY = dirname(dirname(__file__))
def get_full_path(*path):
    return join(MAIN_DIRECTORY, *path)

def init_dirs(*paths):
    """
    :example data_dir, pic_dir = init_dirs('data','imgs')
    :param paths: arguments of paths to create
    :return: absolute path to the created directory
    """
    path_list = []
    for path in paths:
        full_path=get_full_path(path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        path_list.append(full_path)

    return path_list