import gzip
import numpy as np
from urllib import request

train_images = "train-images-idx3-ubyte.gz"
train_labels = "train-labels-idx1-ubyte.gz"

test_images = "t10k-images-idx3-ubyte.gz"
test_labels = "t10k-labels-idx1-ubyte.gz"


base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

request.urlretrieve(base_url+train_images, train_images)
request.urlretrieve(base_url+train_labels, train_labels)

request.urlretrieve(base_url+test_images, test_images)
request.urlretrieve(base_url+test_labels, test_labels)


with gzip.open(train_images, 'rb') as f:
    np_train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)

with gzip.open(test_images, 'rb') as f:
    np_test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)

with gzip.open(train_labels, 'rb') as f:
    np_train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open(test_labels, 'rb') as f:
    np_test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

np.save("train_images.npy", np_train_images)
np.save("test_images.npy", np_test_images)
np.save("train_labels.npy", np_train_labels)
np.save("test_labels.npy", np_test_labels)
