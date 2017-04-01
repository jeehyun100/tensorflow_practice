import tensorflow as tf

def create_feature_columns():


    # Continuous base columns.
    data_1 = tf.contrib.layers.real_valued_column("data_1", dtype=tf.int64)
    # Sparse base columns.
    data_2 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "data_2", hash_bucket_size=1000)




    return set([
        data_1,
        data_2
    ])

def input_fn(mode, data_file, batch_size):
    try:
        input_features = create_feature_columns()
        features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)

        feature_map = tf.contrib.learn.io.read_batch_record_features(
            file_pattern=data_file,
            batch_size=batch_size,
            randomize_input = False,
            features=features,
            reader_num_threads = 2,
            name="read_batch_features_{}".format(mode))
        #sess = tf.InteractiveSession()
        #print(feature_map["age"].eval())
        #sess.close()
      #  with tf.Session() as sess:
        #    print(sess.run(feature_map["age"]))
       #     print(feature_map["age"].eval())

        # a = feature_map
        # sess = tf.Session()
        # sess.run(a)
       # x = tf.Print(feature_map['age'], [feature_map['age']])
       # print(batch_size)




        #target = feature_map.pop("label")
    except Exception as e:
        raise e

    return feature_map