import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os


class DataLoader(object):
    def __init__(self, root):
        super(DataLoader, self).__init__()
        self.root = root

    def get_image(self, img_path, img_height=512, img_width=512, mask=False):
        img = tf.io.read_file(img_path)

        if not mask:
            img = tf.image.decode_png(img, 3)
            img = tf.image.resize(images=img, size=[img_height, img_width]) / 255.

        else:
            img = tf.image.decode_png(img, 1)

            img = tf.cast(tf.image.resize(images=img, size=[img_height, img_width]), dtype=tf.uint8)

        return img

    def load_data(self, img_path, mask_path):
        img, mask = self.get_image(img_path), self.get_image(mask_path, mask=True)

        return img, mask

    def configure_for_performance(self, ds, cnt, shuffle=False):
        if not shuffle:
            ds = ds.batch(1)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        else:
            ds = ds.shuffle(buffer_size=cnt)
            ds = ds.batch(1)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def mk_ds(self, type):
        img_path = glob.glob(f'{self.root}/{type}/*.png')
        mask_path = glob.glob(f'{self.root}/{type}_labels/*png')

        ds = tf.data.Dataset.from_tensor_slices((img_path, mask_path))
        ds = ds.map(self.load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds


