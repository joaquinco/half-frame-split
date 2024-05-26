#!/bin/python3

import argparse
import logging
import os
import sys
import random
import math
import statistics as stats

# pip install Pillow
from PIL import Image

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def norm(values):
    return math.sqrt(sum(map(lambda x: x ** 2, values)))


def get_subimage_path(image_path, suffix, output_dir=None):
    filename, ext = os.path.splitext(image_path)

    if output_dir:
        base_filename = os.path.basename(filename)
        filename = os.path.join(output_dir, base_filename)

    return '{}_{}{}'.format(filename, suffix, ext)


def clean_separator_values(values):
    """
    Given a list of autodetected separation values, return the two
    values that are probably correct.

    TODO: Allow to define detection strategies and use this method.

    Remove outliers that can make the stripe wider causing loss of information.
    Assumes values have a normal distribution, remove outliers and return mean.
    """
    def remove_outliers(numbers, sign=1):
        mean = stats.mean(numbers)
        stdev = stats.stdev(numbers)

        return [n for n in numbers if sign * n <= sign * (mean + sign * stdev)]

    left, right = zip(*values)
    cleaned_values = [remove_outliers(left, sign=-1), remove_outliers(right)]
    return tuple(map(stats.mean, cleaned_values))


def clean_separator_values_minmax(values):
    """
    Filter left greater to avg(rigth), the other way round and finally return
    max(left), min(right)
    """
    left, right = zip(*values)
    lmean, rmean = stats.mean(left), stats.mean(right)

    return (
        max(filter(lambda x: x < rmean, left)),
        min(filter(lambda x: x > lmean, right))
    )


class StripDetector:
    def __init__(
        self,
        black_threshold=50,
        pixel_increment=1,
        detect_samples=100
    ):
        self.threshold = black_threshold
        self.increment = pixel_increment
        self.samples = detect_samples

    def get_separator(self, image, left, right, x):
        """
        Adjust left and right values until a dark (black) color
        is detected.

        A pixel if black if it's norm is <= threshold
        """
        def is_black(pixel):
            if not isinstance(pixel, tuple):
                pixel = (pixel,)

            return norm(pixel) <= self.threshold

        while left < right:
            pixel = image.getpixel((left, x))

            if is_black(pixel):
                break

            left += self.increment

        while right > left:
            pixel = image.getpixel((right, x))

            if is_black(pixel):
                break

            right -= self.increment

        return left, right

    def detect(self, image, samples=None):
        """
        Detects separation between images subframes buy searching for a black
        strip in the middle of the image vertically.

        Return a tuple (left, right) with the start and ending of the strip
        """
        left_offset = image.width * 4 / 10
        right_offset = image.width * 6 / 10
        samples = samples or self.samples

        values = [
            self.get_separator(
                image, left_offset, right_offset,
                int(random.uniform(0, image.height))
            )
            for _ in range(samples)
        ]

        separator = clean_separator_values_minmax(values)
        logger.debug('Detected separator %s', separator)

        return separator


def split_frame(image_path, sep_finder, output_dir):
    """
    Given an image file which is assumed to hold
    a two half frames images, split them into two
    different files.
    """
    logger.debug('Loading %s', image_path)
    image = Image.open(image_path)

    width, height = image.size
    sep_left, sep_right = sep_finder.detect(image)

    left_image = image.crop(
        (0, 0, sep_left, height)
    )
    right_image = image.crop(
        (sep_right, 0, width, height)
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for index, subimage in enumerate([left_image, right_image], start=1):
        subimage_path = get_subimage_path(image_path, index, output_dir)
        with open(subimage_path, 'wb') as fp:
            logger.info('Saving image %s', subimage_path)
            subimage.save(fp)


def main():
    parser = argparse.ArgumentParser(description='Half Frame Photo Splitter')
    parser.add_argument(
        'images_path', help='Half frame image to cut', nargs='+')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument(
        '--black-threshold',
        type=int,
        default=50,
        help='Tweak black strip detection threshold.'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Strip detection samples. Increase for dark images'
    )
    parser.add_argument(
        '-d', '--output-dir',
        help='Output directory',
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    sep_finder = StripDetector(
        black_threshold=args.black_threshold,
        detect_samples=args.samples
    )
    for image_path in args.images_path:
        split_frame(image_path, sep_finder, args.output_dir)


if __name__ == '__main__':
    main()
