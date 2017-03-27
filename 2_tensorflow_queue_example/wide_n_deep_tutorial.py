# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-bad-import-order

#
#
# curl -k -L -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
#
# curl -k -L -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
#
# # Convert CSV to TFRecord: adult.data -> adult_data.tfrecords
# python tfrecord.py --data_file adult.data
#
# # Convert CSV to TFRecord: adult.test -> adult_test.tfrecords
# python tfrecord.py --data_file adult.test --skip_header
#
# python wide_n_deep_tutorial.py \
#     --model_type=wide_n_deep \
#     --train_data=adult_data.tfrecords \
#     --test_data=adult_test.tfrecords \
#     --model_dir=model/

"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data as data

import tempfile
# import urllib
# import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "model", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "adult_data.tfrecords",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "adult_data.tfrecords",
    "Path to the test data.")


COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

tf.logging.set_verbosity(tf.logging.INFO)


# def maybe_download():
#   """May be downloads training data and returns train and test file names."""
#   if FLAGS.train_data:
#     train_file_name = FLAGS.train_data
#   else:
#     train_file = tempfile.NamedTemporaryFile(delete=False)
#     urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)  # pylint: disable=line-too-long
#     train_file_name = train_file.name
#     train_file.close()
#     print("Training data is downloaded to %s" % train_file_name)
#
#   if FLAGS.test_data:
#     test_file_name = FLAGS.test_data
#   else:
#     test_file = tempfile.NamedTemporaryFile(delete=False)
#     urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)  # pylint: disable=line-too-long
#     test_file_name = test_file.name
#     test_file.close()
#     print("Test data is downloaded to %s" % test_file_name)
#
#   return train_file_name, test_file_name


def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                     keys=["female", "male"])
  race = tf.contrib.layers.sparse_column_with_keys(column_name="race",
                                                   keys=["Amer-Indian-Eskimo",
                                                         "Asian-Pac-Islander",
                                                         "Black", "Other",
                                                         "White"])

  education = tf.contrib.layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000)
  marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(
      "marital_status", hash_bucket_size=100)
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100)
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100)
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000)
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000)

  # Continuous base columns.
  age = tf.contrib.layers.real_valued_column("age")
  education_num = tf.contrib.layers.real_valued_column("education_num")
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

  # Transformations.
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        18, 25, 30, 35, 40, 45,
                                                        50, 55, 60, 65
                                                    ])

  # Wide columns and deep columns.
  wide_columns = [gender, native_country, education, occupation, workclass,
                  marital_status, relationship, age_buckets,
                  tf.contrib.layers.crossed_column([education, occupation],
                                                   hash_bucket_size=int(1e4)),
                  tf.contrib.layers.crossed_column(
                      [age_buckets, race, occupation],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([native_country, occupation],
                                                   hash_bucket_size=int(1e4))]
  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(marital_status,
                                         dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(race, dimension=8),
      tf.contrib.layers.embedding_column(native_country,
                                         dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


# def input_fn(df):
#   """Input builder function."""
#   # Creates a dictionary mapping from each continuous feature column name (k) to
#   # the values of that column stored in a constant Tensor.
#   continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
#   # Creates a dictionary mapping from each categorical feature column name (k)
#   # to the values of that column stored in a tf.SparseTensor.
#   categorical_cols = {k: tf.SparseTensor(
#       indices=[[i, 0] for i in range(df[k].size)],
#       values=df[k].values,
#       shape=[df[k].size, 1])
#                       for k in CATEGORICAL_COLUMNS}
#   # Merges the two dictionaries into one.
#   feature_cols = dict(continuous_cols)
#   feature_cols.update(categorical_cols)
#   # Converts the label column into a constant Tensor.
#   label = tf.constant(df[LABEL_COLUMN].values)
#   # Returns the feature columns and the label.
#   return feature_cols, label


def train_and_eval():
  """Train and evaluate the model."""
  # train_file_name, test_file_name = maybe_download()
  # df_train = pd.read_csv(
  #     tf.gfile.Open(train_file_name),
  #     names=COLUMNS,
  #     skipinitialspace=True)
  # df_test = pd.read_csv(
  #     tf.gfile.Open(test_file_name),
  #     names=COLUMNS,
  #     skipinitialspace=True,
  #dnn / input_from_feature_columns / input_from_feature_columns / education_embedding / education_embeddingweights

  #     skiprows=1)
  #dnn / input_from_feature_columns / input_from_feature_columns / education_embedding / lookup
  #
  # df_train[LABEL_COLUMN] = (
  #     df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
  # df_test[LABEL_COLUMN] = (
  #     df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
  #dnn / input_from_feature_columns / input_from_feature_columns / hours_per_week
  #dnn/input_from_feature_columns/input_from_feature_columns/education_num/ToFloat
  #read_batch_features_train
  #dnn/input_from_feature_columns/input_from_feature_columns/age/ToFloa
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)
  #tensor_names = ['dnn/input_from_feature_columns/input_from_feature_columns/hours_per_week']
  #read_batch_features_train / fifo_queue_Dequeue
  tensor_names = ['dnn/input_from_feature_columns/input_from_feature_columns/education_num/ToFloat']
  #tensor_names = ['dnn/input_from_feature_columns/input_from_feature_columns/education_embedding/InnerFlatten/SparseReshape/Identity','dnn/input_from_feature_columns/input_from_feature_columns/education_embedding/lookup']
  #tensor_names = ['read_batch_features_train/fifo_queue_Dequeue']#x
  #tensor_names = ['read_batch_features_train/read/ReaderReadUpToV2']
  #tensor_names = ['read_batch_features_train/read/TFRecordReaderV2'] # 뭔가 느림
  #tensor_names = ['read_batch_features_train/cond/random_shuffle_queue_EnqueueMany/Switch_2']  # 뭔가 느림= ['read_batch_features_train/read/TFRecordReaderV2']  # ?? ??


  #read_batch_features_train / cond / random_shuffle_queue_EnqueueMany / Switch_1

  print_monitor = tf.contrib.learn.monitors.PrintTensor(tensor_names, every_n=50)

  m = build_estimator(model_dir)
  # m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
  # results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  m.fit(input_fn=lambda: data.input_fn(tf.contrib.learn.ModeKeys.TRAIN, FLAGS.train_data, 128), steps=FLAGS.train_steps,monitors=[print_monitor] )
  results = m.evaluate(input_fn=lambda: data.input_fn(tf.contrib.learn.ModeKeys.EVAL, FLAGS.test_data, 128), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)
  input_features = data.create_feature_columns()
  features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)

  _features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features= features)
  age = tf.cast(_features['age'],tf.int64)
#
# # Convert from a scalar string tensor (whose single string has
# # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
# # [mnist.IMAGE_PIXELS].
# image = tf.decode_raw(features['image_raw'], tf.uint8)
# annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
#
# height = tf.cast(features['height'], tf.int32)
# width = tf.cast(features['width'], tf.int32)
#
# image_shape = tf.pack([height, width, 3])
# annotation_shape = tf.pack([height, width, 1])
#
# image = tf.reshape(image, image_shape)
# annotation = tf.reshape(annotation, annotation_shape)
#
# image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
# annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)
#
# # Random transformations can be put here: right before you crop images
# # to predefined size. To get more information look at the stackoverflow
# # question linked above.
#
# resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
#                                                        target_height=IMAGE_HEIGHT,
#                                                        target_width=IMAGE_WIDTH)
#
# resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
#                                                             target_height=IMAGE_HEIGHT,
#                                                             target_width=IMAGE_WIDTH)
#
  age = tf.train.shuffle_batch([age],
                                              batch_size=3,
                                              capacity=30,
                                              num_threads=2,
                                              min_after_dequeue=10)

  return age

def read_and_decode2(filename_queue):
    try:
        input_features = data.create_feature_columns()
        features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)

        feature_map = tf.contrib.learn.io.read_batch_record_features(
            file_pattern=[filename_queue],
            batch_size=2,
            features=features,
            name="read_batch_features")
        # sess = tf.InteractiveSession()
        # print(feature_map["age"].eval())
        # sess.close()
        #  with tf.Session() as sess:
        #    print(sess.run(feature_map["age"]))
        #     print(feature_map["age"].eval())

        # a = feature_map
        # sess = tf.Session()
        # sess.run(a)
        x = tf.Print(feature_map['age'], [feature_map['age']])
        print(x)

        target = feature_map.pop("label")
    except Exception as e:
        raise e

    return feature_map, target

def multi_queue():
    filename = 'adult_data.tfrecords'

    #Queue 는 이런식으로 설정 여기서는 쓰지 않음
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=1)

    #꼭 local variable initial  해야함
    init_op =  tf.local_variables_initializer()

    #Multi Thread로 들고옴
    feature_map, target = data.input_fn(tf.contrib.learn.ModeKeys.EVAL, FLAGS.test_data, 128)


    with tf.Session() as sess:
        # Start populating the filename queue.
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tfrecord_list_row = list() # 출력을 위한 List
        print_column = True

        for i in range(3):
            # Multi Thread에서 넣을것을 Session으로 실행
            example, label = sess.run([feature_map, target])

            _row = ""
            if print_column == True: # Header를 위한 Column Key 설정 첫줄만
                tfrecord_list_key = [col for col in example.keys()]
                tfrecord_list_key.append('label')
                print_column = False
                tfrecord_list_row.append(tfrecord_list_key)

            for i in range(len(example[list(example.keys())[0]])): # Row를 들고 오기 뭔가 지저분함

                tfrecord_list_col = list()
                for _k in example.keys():
                    if str(type(example[_k])).find('Sparse') > -1: #Sparse는 Bytes로 나와서 Bytes를 String 으로 처리
                        tfrecord_list_col.append(str(example[_k].values[i].decode()))
                    else:
                        # numpy도 ndarray로 나와서 [0]을 붙여 정리함
                        tfrecord_list_col.append(str(example[_k][i][0]))
                tfrecord_list_col.append(str(label[i][0]))
                columns_value = tfrecord_list_col
                tfrecord_list_row.append(columns_value)

            #이쁘게 출력하기 위해 Print 함수 설정
            for item in tfrecord_list_row:
                print(str(item[0:])[1:-1])

        coord.request_stop()
        coord.join(threads)


def main(_):
  multi_queue()
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
