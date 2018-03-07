# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
from PIL import Image

def writer_TFRecord():
    """
    生成TFRecords文件
    """
    with tf.python_io.TFRecordWriter("./data/train.tfrecords") as writer:
        root_path = "./data/train/"
        for img_name in os.listdir(root_path):
            img_path = root_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            if 'cat' in img_name:
                img_target = 0
            else:
                img_target = 1
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_target])),
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
            writer.write(example.SerializeToString())   # 序列化为字符串


def read_TFRecord():
    """
    读取TFRecords文件
    """
    for serialized_example in tf.python_io.tf_record_iterator("./data/train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature["image"].bytes_list.value
        label = example.features.feature["label"].int64_list.value

        print(type(image), label)
        break

def queue_read_TFRecord():
    """
    使用队列读取TFRecords文件
    """

    filename_queue = tf.train.string_input_producer(["./data/train.tfrecords"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "label": tf.FixedLenFeature([], tf.int64),
        "image": tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features["image"], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features["label"], tf.int32)

    return image,label

def main():
    
    image, label = queue_read_TFRecord()
    img_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=32, capacity=2000, min_after_dequeue=1000)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l= sess.run([img_batch, label_batch])
            print(val.shape, l)



if __name__ == '__main__':
    # writer_TFRecord()

    # read_TFRecord()

    # queue_read_TFRecord()

    main()
