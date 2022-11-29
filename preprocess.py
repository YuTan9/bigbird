# import sentencepiece as spm
# import numpy as np
# sp = spm.SentencePieceProcessor()
# sp.load('en.model')

# print(sp.encode_as_ids(['Hello world', 'This is a test']))
# print(sp.encode_as_pieces(['Hello world', 'This is a test']))
with open('./data/UM/en.txt', 'r') as f:
    en = f.read()
#     en = en.split('\n')
#     print(len(en))
#     print(en[0])
#     np.save('./data/UM/tokenized_en.npy', np.array(sp.encode_as_ids(en), dtype = 'object'))

# sp = spm.SentencePieceProcessor()
# sp.load('zh.model')
with open('./data/UM/zh.txt', 'r') as f:
    zh = f.read()
#     zh = zh.split('\n')
#     print(len(zh))
#     print(zh[0])
#     np.save('./data/UM/tokenized_zh.npy', np.array(sp.encode_as_ids(zh), dtype = 'object'))

import os
import tensorflow as tf
import numpy as np

# en = np.load('./data/UM/tokenized_en.npy', allow_pickle = True)
# zh = np.load('./data/UM/tokenized_zh.npy', allow_pickle = True)


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(english, chinese):
    feature = {
        "en": bytes_feature(english),
        "zh": bytes_feature(chinese),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "en-encoding": tf.io.FixedLenFeature([], tf.string),
        "zh-encoding": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

with tf.io.TFRecordWriter('./data/UM/train.tfrecord') as writer:
    for idx in range(int(len(en) * 0.7)):
        # print(en[idx], zh[idx])
        example = create_example(en[idx], zh[idx])
        writer.write(example.SerializeToString())
with tf.io.TFRecordWriter('./data/UM/test.tfrecord') as writer:
    for idx in range(int(len(en) * 0.7), len(en)):
        # print(en[idx], zh[idx])
        example = create_example(en[idx], zh[idx])
        writer.write(example.SerializeToString())