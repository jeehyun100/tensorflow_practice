import tensorflow as tf

def create_feature_columns():
    # Sparse base columns.
    gender = tf.contrib.layers.sparse_column_with_keys(
        column_name="gender", keys=["female", "male"])
    race = tf.contrib.layers.sparse_column_with_keys(
        column_name="race", keys=[
            "Amer-Indian-Eskimo",
            "Asian-Pac-Islander",
            "Black",
            "Other",
            "White"
        ])

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
    age = tf.contrib.layers.real_valued_column("age", dtype=tf.int64)
    education_num = tf.contrib.layers.real_valued_column("education_num", dtype=tf.int64)
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain", dtype=tf.int64)
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss", dtype=tf.int64)
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week", dtype=tf.int64)

    label = tf.contrib.layers.real_valued_column("label", dtype=tf.int64)

    return set([
        workclass,
        education,
        marital_status,
        occupation,
        relationship,
        race,
        gender,
        native_country,
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        label,
    ])

def input_fn(mode, data_file, batch_size):
    try:
        input_features = create_feature_columns()
        features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)

        feature_map = tf.contrib.learn.io.read_batch_record_features(
            file_pattern=[data_file],
            batch_size=batch_size,
            features=features,
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
        x = tf.Print(feature_map['age'], [feature_map['age']])
        print(x)




        target = feature_map.pop("label")
    except Exception as e:
        raise e

    return feature_map, target