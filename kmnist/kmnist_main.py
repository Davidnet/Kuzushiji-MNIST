import os
import tensorflow as tf 



def parse_records(example_proto):
    """Parse the input tf.Example proto using the dictionary above."""
    image_feature_description = {
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    return tf.parse_single_example(example_proto, image_feature_description)

def parse_into_image(features_dict):
    serialized_image = features_dict["image"]
    image = tf.parse_tensor(serialized_image, tf.uint8)
    label = features_dict["label"]
    image = tf.reshape(image, [28, 28])
    return dict(images = image), label

def train_input_fn(filename, batch_size):
    raw_dataset = tf.data.TFRecordDataset(filename)
    raw_dataset = raw_dataset.map(parse_records)
    raw_dataset = raw_dataset.map(parse_into_image)
    raw_dataset = raw_dataset.shuffle(buffer_size=100000).repeat().batch(batch_size)

    return raw_dataset

def eval_input_fn(filename, batch_size):
    raw_dataset = tf.data.TFRecordDataset(filename)
    raw_dataset = raw_dataset.map(parse_records)
    raw_dataset = raw_dataset.map(parse_into_image)
    raw_dataset = raw_dataset.batch(batch_size)
    return raw_dataset


feature_columns = [tf.feature_column.numeric_column("images", shape=[28, 28])]

classifier = tf.estimator.DNNClassifier(
    hidden_units=[256, 32],
    feature_columns=feature_columns,
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.3,
    model_dir="/models/canned_estimator",
    batch_norm=True
)

serving_feature_spec = tf.feature_column.make_parse_example_spec(
    feature_columns
)

serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      serving_feature_spec
)

best_exporter = tf.estimator.BestExporter(
    serving_input_receiver_fn=serving_input_receiver_fn,
)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda : train_input_fn("/data/tf_records/kmnist-train.tfrecord", 32), 
    max_steps=100000,
    )


eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda : eval_input_fn("/data/tf_records/kmnist-dev.tfrecord", 32),
    exporters=best_exporter,
    start_delay_secs=0,
    throttle_secs=5
)


tf.estimator.train_and_evaluate(
    classifier,
    train_spec,
    eval_spec
)

