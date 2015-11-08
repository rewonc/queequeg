import cv2
import os
import dslib
from dslib import dataset
import json
import re
import numpy as np
import random
from functools import partial


def get_records_from_json(path):
    with open(path) as data_file:
        annotations = json.load(data_file)
    return annotations


def separate_annotations(record):
    return ({'filename': record['filename'], 'annotation': ann} for ann in record['annotations'])


def get_filename(path):
    filename_regex = re.compile("w_[0-9]*.jpg")
    return filename_regex.search(path).group()


def load_image(path, name):
    return cv2.imread(path + name)


def add_label_image(record):
    mask = np.zeros((record['image'].shape[0], record['image'].shape[1]))
    ann = record['annotation']
    y, height = int(ann['y']), int(ann['height'])
    x, width = int(ann['x']), int(ann['width'])
    mask[y:y + height, x:x + width] = 1
    record['mask'] = mask
    return record


def center_crop(output_size, im):
    '''
    Crop a 2d image to the intended size by scaling one dimension
    and center cropping in the other
    '''
    if im is None:
        raise TypeError
    x0, y0 = im.shape[0], im.shape[1]
    x1, y1 = output_size
    aspect_ratio = x0 / float(y0)
    resize_via_y = y1 * aspect_ratio > x1
    if resize_via_y:
        scale_ratio = y1 / float(y0)
        scaled_x = int(x0 * scale_ratio)
        scaled = cv2.resize(im, (y1, scaled_x))
        diff = (scaled_x - x1) / 2
        cropped = scaled[diff:diff + x1, :]
    else:
        scale_ratio = x1 / float(x0)
        scaled_y = int(y0 * scale_ratio)
        scaled = cv2.resize(im, (scaled_y, x1))
        diff = (scaled_y - y1) / 2
        cropped = scaled[:, diff:diff + y1]
#     print im.shape, output_size, cropped.shape
    return cropped


def augment_data(in_data,
                 zoom_max_multiplier=1.5,
                 stretch_max_multiplier=1.5,
                 translation_max_distance=10,
                 rotation_max=0.5,
                 shear_max=0.05,
                 horizontal_flip=0.5,
                 vertical_flip=0.5,
                 affine_mode="reflect",
                 contrast_enhance_max=1.6,
                 brightness_enhance_max=1.6,
                 color_enhance_max=1.3,
                 ):


    train_affine_params = dict(
        zoom_range=(1.0 / zoom_max_multiplier,
                    zoom_max_multiplier),
        translation_range=(-translation_max_distance,
                           translation_max_distance),
        rotation_range=(-rotation_max,
                        rotation_max),
        stretch_range=(1.0 / stretch_max_multiplier,
                       stretch_max_multiplier),
        shear_range=(-shear_max,
                     shear_max),
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        mode=affine_mode,
    )

    def same_transform(im, mask):
        fn = dslib.data_augmentation.image2d.random_affine_fn(im.shape,
                **train_affine_params)
        return [fn(im), fn(mask)]


    def transpose_axes(im):
        return im.transpose(2, 0, 1)


    def color_jitter(im, multiplier_range=(-0.7, 0.7)):
        if np.max(im) > 1.0:
            raise TypeError("Must be between 01")
        if np.min(im) < 0.0:
            raise TypeError("Must be between 01")
        std = np.std(im)

        for i in xrange(im.shape[2]):
            multiplier = random.uniform(multiplier_range[0],
                                        multiplier_range[1])
            im[:, :, i] += std * multiplier
            return np.clip(im, 0.0, 1.0)

        return im

    enhance_params = dict(
        contrast_range=(1.0 / contrast_enhance_max, contrast_enhance_max),
        brightness_range=(1.0 / brightness_enhance_max, brightness_enhance_max),
    )

    augmented_data = in_data.map(
        key=['image', 'mask'],
        out=['image', 'mask'],
        fn=same_transform
    ).map_key(
        key='image',
        fn=dslib.cv2_utils.to_01
    ).map_key(
        key='image',
        fn=dslib.data_augmentation.image2d.random_enhancement_augmentation,
        kwargs=enhance_params
    ).map_key(
        key='image',
        fn=color_jitter
    ).map_key(
        key='image',
        fn=transpose_axes
    )
    return augmented_data


def batchify(dset, batch_size=16):
    pass


def get_train_test_gens(anno_type='Head', rel_img_path='../imgs/',
                        desired_output_size=(300, 400),
                        test_split_percentage=0.20,
                        annotations_dir='../code/right_whale_hunt/annotations/',
                        chunk_size=16):

    annotations_filenames = os.listdir(annotations_dir)
    random.seed = 42

    records = dataset.from_list(
        annotations_filenames
    ).mapcat(
        fn=lambda name: get_records_from_json(annotations_dir + name)
    ).mapcat(
        fn=separate_annotations
    ).filter(
        fn=lambda x: x['annotation']['class'] == anno_type
    ).map_key(
        key='filename',
        fn=get_filename
    ).to_list()

    random.shuffle(records)

    total_num_records = len(records)
    test_split_idx = int(total_num_records * test_split_percentage)
    test_records = records[:test_split_idx]
    train_records = records[test_split_idx:]

    train_gen = dataset.from_list(train_records).random_sample(
    ).map(
        key='filename',
        out='image',
        fn=partial(load_image, rel_img_path)
    ).map(
        fn=add_label_image
    ).map_each_key(
        keys=['image', 'mask'],
        fn=partial(center_crop, desired_output_size)
    )

    test_gen = dataset.from_list(train_records).random_sample(
    ).map(
        key='filename',
        out='image',
        fn=partial(load_image, rel_img_path)
    ).map(
        fn=add_label_image
    ).map_each_key(
        keys=['image', 'mask'],
        fn=partial(center_crop, desired_output_size)
    )

    out_train = augment_data(train_gen).numpy_chunk(
        chunk_size=chunk_size,
        keys=['image', 'mask']
    )

    out_test = test_gen.numpy_chunk(
        chunk_size=chunk_size,
        keys=['image', 'mask']
    )

    return out_train, out_test
