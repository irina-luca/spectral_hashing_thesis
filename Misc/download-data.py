import numpy as np
from sklearn.datasets import fetch_mldata, fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def get_mnist_data(input_file):
    print("Download MNIST dataset")
    mnist = fetch_mldata('MNIST original')
    np.savetxt(input_file, mnist.data, delimiter=' ')


def get_sift_data(input_file):
    sift_data = fvecs_read(input_file)
    return sift_data


# -- Function to extract data vectors from fvecs format. All credit goes to: https://gist.github.com/danoneata/49a807f47656fedbb389 -- #
def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def save_data_to_file(output_file, data_set, delimiter):
    np.savetxt(output_file, data_set, delimiter=delimiter)


def main_download_datasets():
    # -- MNIST dataset: The MNIST database contains a total of 70000 examples of handwritten digits of size 28x28 pixels, labeled from 0 to 9; MNIST === bitmap dataset -- #
    # get_mnist_data("../Data/MNIST/mnist.txt")
    # mnist_data = np.genfromtxt("../Data/MNIST/mnist.data", delimiter=' ', dtype=np.float)
    # print(mnist_data[201])


    # -- SIFT dataset: sift descriptors -- #
    # sift_data = get_sift_data("../Data/SIFT/gist_base.fvecs")
    # print(sift_data.shape)
    # print(sift_data[1029])
    # save_data_to_file("../Data/SIFT/ANN_GIST1M_960D.data", sift_data, " ")


    # -- The 20 newsgroups text dataset -- #
    categories = ['alt.atheism',
                   'comp.graphics',
                   'comp.os.ms-windows.misc',
                   'comp.sys.ibm.pc.hardware',
                   'comp.sys.mac.hardware',
                   'comp.windows.x',
                   'misc.forsale',
                   'rec.autos',
                   'rec.motorcycles',
                   'rec.sport.baseball',
                   'rec.sport.hockey',
                   'sci.crypt',
                   'sci.electronics',
                   'sci.med',
                   'sci.space',
                   'soc.religion.christian',
                   'talk.politics.guns',
                   'talk.politics.mideast',
                   'talk.politics.misc',
                   'talk.religion.misc']
    newsgroups_vectors = fetch_20newsgroups_vectorized(subset='all', remove=('headers', 'footers', 'quotes'))
    newsgroups_data = newsgroups_vectors.data.todense()
    # print(newsgroups_data)
    # save_data_to_file("../Data/20-NewsGroups-Text/20-NewsGroups-Text.data", newsgroups_data, " ")
    # print(newsgroups_data[0].sum())
    print(newsgroups_data.shape)

    # Maybe do JL on this dataset #











    # vectorizer = TfidfVectorizer()
    # vectors = vectorizer.fit_transform(newsgroups_train.data)
    # print(vectors[0:10, :])
    # print(vectors.shape)
    # save_data_to_file("../Data/20-NewsGroups-Text/...    .data", vectors, " ")



main_download_datasets()











# def fvecs_read(filename, c_contiguous=True):
#     fv = np.fromfile(filename, dtype=np.float32)
#     if fv.size == 0:
#         return np.zeros((0, 0))
#     dim = fv.view(np.int32)[0]
#     assert dim > 0
#     fv = fv.reshape(-1, 1 + dim)
#     if not all(fv.view(np.int32)[:, 0] == dim):
#         raise IOError("Non-uniform vector sizes in " + filename)
#     fv = fv[:, 1:]
#     if c_contiguous:
#         fv = fv.copy()
#     return fv
#
#
# # data = fvecs_read("downloaded-data/mnist")
# # data = fvecs_read("downloaded-data/mnist-with-awgn")
# # data = fvecs_read("downloaded-data/siftsmall_base.fvecs")
# # data = fvecs_read("downloaded-data/siftsmall_groundtruth.ivecs")
# print(data)
# print(data.shape)
#
