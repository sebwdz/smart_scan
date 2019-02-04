import glob
import PIL.Image
import random

import numpy as np
from skimage.transform import resize
import scipy.ndimage.interpolation

import tensorflow as tf

import contextlib2
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util


def read_images(directory):

    files = glob.glob(directory)
    for file in files:
        image = np.array(PIL.Image.open(file).convert('RGB'))
        if image.shape[0] > 250 and image.shape[1] > 250:
            yield image


def get_fixed_element():
    return np.array(PIL.Image.open('data/fixed.png').convert('RGB').resize((200, 150)))


def randomize_image(shape):
    x_ = tf.placeholder(shape=shape, dtype=tf.float32)
    image = tf.image.random_brightness(x_, 0.8, seed=None)
    image = tf.image.random_contrast(image, 0.05, 0.95, seed=None)
    image = tf.image.random_hue(image, 0.4, seed=None)
    session = tf.Session()
    return lambda img: session.run(image, feed_dict={x_: img})


def image_to_jpeg(shape):
    x_ = tf.placeholder(shape=shape, dtype=tf.uint8)
    image_data = tf.image.encode_jpeg(
        x_, format='', quality=95, progressive=False, optimize_size=False, chroma_downsampling=True, density_unit='in',
        x_density=300, y_density=300, xmp_metadata='', name=None
    )
    session = tf.Session()
    return lambda img: session.run(image_data, feed_dict={x_: img})


def randomly_add_fixed(image, fixed):

    x_rate = float(random.randint(30, 100) / 100)
    y_rate = float(random.randint(30, 100) / 100)

    fixed_resized = resize(fixed, (int(fixed.shape[0] * y_rate), int(fixed.shape[1] * x_rate)),
                           mode='constant', preserve_range=True)

    position_y = random.randint(1, image.shape[0] - (fixed_resized.shape[0]) - 1)
    position_x = random.randint(1, image.shape[1] - (fixed_resized.shape[1]) - 1)

    image[position_y:(fixed_resized.shape[0] + position_y), position_x:(fixed_resized.shape[1] + position_x)] = fixed_resized

    return image, [position_y, position_x, fixed_resized.shape[0] + position_y, fixed_resized.shape[1] + position_x]


def resize_bounding_box(initial_shape, bbox):

    ry = initial_shape[0]
    rx = initial_shape[1]

    bbox[0] = bbox[0] / ry
    bbox[1] = bbox[1] / rx

    bbox[2] = bbox[2] / ry
    bbox[3] = bbox[3] / rx

    return bbox


def to_tf_example(image_data, bbox, shape):

    height = shape[0]
    width = shape[1]

    xmins = [bbox[1]]
    xmaxs = [bbox[3]]
    ymins = [bbox[0]]
    ymaxs = [bbox[2]]

    classes_text = [b'object']
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(b'test'),
        'image/source_id': dataset_util.bytes_feature(b'test'),
        'image/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(image_data)),
        'image/format': dataset_util.bytes_feature(b'jpg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def get_next_batch(directory, batch_size=1000, shape=(200, 200), randomize=1):
    fixed = get_fixed_element()

    x_ = []

    fixed_randomizer = randomize_image(fixed.shape)
    image_randomizer = randomize_image((*shape, 3))
    to_jpeg = image_to_jpeg((*shape, 3))

    image_reader = read_images(directory)

    for r_image in image_reader:

        for i in range(randomize):

            image = r_image

            if random.randint(0, 10) >= 3:
                fixed_randomized = fixed_randomizer(fixed)
            else:
                fixed_randomized = fixed

            if random.randint(0, 10) >= 3:
                fixed_randomized = scipy.ndimage.interpolation.rotate(fixed_randomized, random.randint(0, 30) - 15)

            image, bbox = randomly_add_fixed(image, fixed_randomized)
            bbox = resize_bounding_box(image.shape, bbox)

            if random.randint(0, 10) >= 3:
                resize_shape = random.randint(10, 40) / 10
                image = resize(image, (int(shape[0] / resize_shape), int(shape[1] / resize_shape)), mode='constant', preserve_range=True)

            image = resize(image, shape, mode='constant', preserve_range=True)
            if random.randint(0, 10) >= 3:
                image = image_randomizer(image).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            image = to_jpeg(image)

            x_.append(to_tf_example(image, bbox, shape))

            if len(x_) > batch_size:
                yield x_
                x_ = []

    if len(x_):
        yield x_


def simple_get_next_batch(directory, batch_size=1000, shape=(200, 200)):
    fixed = get_fixed_element()

    x_ = []
    y_ = []

    fixed_randomizer = randomize_image(fixed.shape)
    image_randomizer = randomize_image((*shape, 3))

    image_reader = read_images(directory)

    for image in image_reader:

        fixed_randomized = fixed_randomizer(fixed)

        image, bbox = randomly_add_fixed(image, fixed_randomized)

        #bbox = resize_bounding_box(image.shape, bbox)

        mask = np.zeros((image.shape[0], image.shape[1], 1))
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3], 0] = 1

        image = resize(image, shape, mode='constant', preserve_range=True)
        mask = resize(mask, shape, mode='constant', preserve_range=True)
        image = image_randomizer(image).astype(np.uint8)

        x_.append(image)
        y_.append(mask)

        if len(x_) > batch_size:
            yield x_, y_
            x_ = []
            y_ = []

    if len(x_):
        yield x_, y_


def get_next_image(directory, batch_size=1000, shape=(200, 200)):
    fixed = get_fixed_element()

    x_ = []

    fixed_randomizer = randomize_image(fixed.shape)
    image_randomizer = randomize_image((*shape, 3))

    image_reader = read_images(directory)

    for image in image_reader:

        if random.randint(0, 10) >= 3:
            fixed_randomized = fixed_randomizer(fixed)
        else:
            fixed_randomized = fixed

        if random.randint(0, 10) >= 3:
            fixed_randomized = scipy.ndimage.interpolation.rotate(fixed_randomized, random.randint(0, 20) - 10)

        image, bbox = randomly_add_fixed(image, fixed_randomized)

        if random.randint(0, 10) >= 3:
            resize_shape = random.randint(10, 20) / 10
            image = resize(image, (int(shape[0] / resize_shape), int(shape[1] / resize_shape)), mode='constant', preserve_range=True)

        image = resize(image, shape, mode='constant', preserve_range=True)
        if random.randint(0, 10) >= 3:
            image = image_randomizer(image).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        x_.append(image)
        if len(x_) > batch_size:
            yield x_
            x_ = []

    if len(x_):
        yield x_


def run():

    num_shards = 10
    output_filebase = './data/training.record'
    index = 0
    random_count = 10

    with contextlib2.ExitStack() as tf_record_close_stack:

        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)

        for r in range(random_count):

            print('random = ', r)
            examples_reader = get_next_batch('./data/INRIAPerson/Train/i_*/*', batch_size=num_shards, shape=(600, 600), randomize=4)

            for examples in examples_reader:

                for example in examples:
                    output_shard_index = index % num_shards
                    output_tfrecords[output_shard_index].write(example.SerializeToString())
                    index += 1
                    print('...', index, '...', end='\r')

    writer = tf.python_io.TFRecordWriter('./data/eval.record')

    examples = get_next_batch('./data/INRIAPerson/Test/i_*/*', batch_size=1000, shape=(600, 600)).__next__()

    for example in examples:
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    run()
