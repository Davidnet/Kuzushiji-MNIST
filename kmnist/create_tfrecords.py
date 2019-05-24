# TODO: Remove complexity related to the use of py_function and to the eager execution
import os
import argparse
import numpy as np
import tensorflow as tf 
tf.enable_eager_execution()


DATA_DIR = os.path.join("/data/tf_records")
DATA_IMAGES = "/data/kmnist-train-imgs.npz"
DATA_LABELS = "/data/kmnist-train-labels.npz"

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label):
    """Create a tf.Example message ready to be written to a file.
    """

    image_str = tf.serialize_tensor(image)

    feature = {
        "image": _bytes_feature(image_str.numpy()),
        "label": _int64_feature(label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(image, label):
    tf_string = tf.py_function(
        serialize_example,
        (image, label),
        tf.string)
    return tf.reshape(tf_string, ())

def main(data_dir=DATA_DIR):

    with np.load(DATA_IMAGES) as raw_images, np.load(DATA_LABELS) as raw_labels:
        features = raw_images['arr_0']
        labels = raw_labels['arr_0']

    kmnist_dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dev_dataset = kmnist_dataset.take(1000)
    train_dataset = kmnist_dataset.skip(1000)
    print(dev_dataset)


    serialized_dev_dataset = dev_dataset.map(tf_serialize_example)
    serialized_train_dataset = train_dataset.map(tf_serialize_example)

    assert os.path.isdir(data_dir), "data directory is not a directory"

    tfrdev_path = os.path.join(data_dir, "kmnist-dev.tfrecord")
    tfrtrain_path = os.path.join(data_dir, "kmnist-train.tfrecord")


    writer_dev = tf.data.experimental.TFRecordWriter(tfrdev_path)
    writer_train = tf.data.experimental.TFRecordWriter(tfrtrain_path)

    writer_dev.write(serialized_dev_dataset)
    writer_train.write(serialized_train_dataset)

    # writer = tf.data.experimental.TFRecordWriter(filename)

    # filename = '/data/tf_records/test.tfrecord'
    # writer = tf.data.experimental.TFRecordWriter(filename)
    # write_op = writer.write(serialized_test_dataset)





if __name__ == "__main__":
    main()