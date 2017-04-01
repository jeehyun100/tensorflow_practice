import tensorflow as tf
import input_data as data


def ilen(it):
    return len(list(it))


def multi_queue():
    filenames = ['11_csv.tfrecords','22_csv.tfrecords','33_csv.tfrecords','44_csv.tfrecords']

    #Queue 는 이런식으로 설정 여기서는 쓰지 않음
    filename_queue = tf.train.string_input_producer(
        filenames , num_epochs=1)

    #꼭 local variable initial  해야함
    init_op =  tf.local_variables_initializer()


    file_name = "11_csv.tfrecords"

    iterrrrr = tf.python_io.tf_record_iterator(file_name)
    print("tfrecord count " + str(ilen(iterrrrr)))
        # for s_example in tf.python_io.tf_record_iterator(file_name):
        #     example = tf.parse_single_example(s_example, features=features)
        #     data.append(tf.expand_dims(example['x'], 0))

    #reader = tf.TFRecordReader()
    #index, row = reader.read(filename_queue)

    #Multi Thread로 들고옴
    feature_map = data.input_fn(tf.contrib.learn.ModeKeys.EVAL, filenames, 40)


    with tf.Session() as sess:
        # Start populating the filename queue.
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tfrecord_list_row = list() # 출력을 위한 List
        print_column = True

        for i in range(5):
            # Multi Thread에서 넣을것을 Session으로 실행
            example = sess.run(feature_map)

            _row = ""
            if print_column == True: # Header를 위한 Column Key 설정 첫줄만
                tfrecord_list_key = [col for col in example.keys()]
                #tfrecord_list_key.append('label')
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
                #tfrecord_list_col.append(str(label[i][0]))
                columns_value = tfrecord_list_col
                tfrecord_list_row.append(columns_value)

            #이쁘게 출력하기 위해 Print 함수 설정
            for item in tfrecord_list_row:
                print(str(item[0:])[1:-1])
                #tfrecord_list_row.clear()
            tfrecord_list_row.clear()

        coord.request_stop()
        coord.join(threads)

def main(_):
  multi_queue()


if __name__ == "__main__":
  tf.app.run()