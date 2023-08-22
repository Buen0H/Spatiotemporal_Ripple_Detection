'''
Data Handling
'''
from os import listdir
from PIL import Image
from numpy import array, int64
def getArraysFromTIFs(folderpath):
    ''' Return arrays from spatiotemproal maps inside folder
    INPUT
    folderpath  - string specifying directory of folder containing data
    OUTPUT
    data        - dictionary with keys corresponding to name of image and value containing a numpy array (N, T)
    '''
    DESIRED_SUFFIX = '.tif'
    files = [file for file in listdir(folderpath)   # get all tif files inside folder
             if file.endswith(DESIRED_SUFFIX)]
    files.sort()                                    # sort them
    data = {}
    for file in files:
        label = file[:-len(DESIRED_SUFFIX)]         # subject label
        with Image.open(folderpath+file) as im:     # proper way of opening a file; closes automatically
            data[label] = array(im.convert('L'), dtype=int64).T  # convert to greyscale
            # Old way of importing images and greyscaling by taking a sum
            # imarray = array(im, dtype=int64)        # import tif file into an (T, N, 3) array
            # data[label] = imarray.sum(axis=2).T     # returning sum based upon previous work; (N, T)
    return data
