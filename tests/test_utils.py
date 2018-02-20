#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for nussl/utils.py
"""

import os

import unittest
import nussl
import numpy as np
from scipy import signal
import constants
from six.moves.urllib_parse import urljoin

class TestUtils(unittest.TestCase):
    """

    """

    def test_find_peak_indices(self):
        array = np.arange(0, 100)
        peak = nussl.utils.find_peak_indices(array, 1)[0]
        assert peak == 99

        array = np.arange(0, 100).reshape(10, 10)
        peak = nussl.utils.find_peak_indices(array, 3, min_dist=0)
        assert peak == [[9, 9], [9, 8], [9, 7]]

    def test_find_peak_values(self):
        array = np.arange(0, 100)
        peak = nussl.utils.find_peak_values(array, 1)[0]
        assert peak == 99

        array = np.arange(0, 100).reshape(10, 10)
        peak = nussl.utils.find_peak_values(array, 3, min_dist=0)
        assert peak == [99, 98, 97]

    def test_add_mismatched_arrays(self):
        long_array = np.ones((20,))
        short_array = np.arange(10)
        expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

        # Test basic cases
        result = nussl.utils.add_mismatched_arrays(long_array, short_array)
        assert all(np.equal(result, expected_result))

        result = nussl.utils.add_mismatched_arrays(short_array, long_array)
        assert all(np.equal(result, expected_result))

        expected_result = expected_result[:len(short_array)]

        result = nussl.utils.add_mismatched_arrays(long_array, short_array, truncate=True)
        assert all(np.equal(result, expected_result))

        result = nussl.utils.add_mismatched_arrays(short_array, long_array, truncate=True)
        assert all(np.equal(result, expected_result))

        # Test complex casting
        short_array = np.arange(10, dtype=complex)
        expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=complex)

        result = nussl.utils.add_mismatched_arrays(long_array, short_array)
        assert all(np.equal(result, expected_result))

        result = nussl.utils.add_mismatched_arrays(short_array, long_array)
        assert all(np.equal(result, expected_result))

        expected_result = expected_result[:len(short_array)]

        result = nussl.utils.add_mismatched_arrays(long_array, short_array, truncate=True)
        assert all(np.equal(result, expected_result))

        result = nussl.utils.add_mismatched_arrays(short_array, long_array, truncate=True)
        assert all(np.equal(result, expected_result))

        # Test case where arrays are equal length
        short_array = np.ones((15,))
        expected_result = short_array * 2

        result = nussl.utils.add_mismatched_arrays(short_array, short_array)
        assert all(np.equal(result, expected_result))

        result = nussl.utils.add_mismatched_arrays(short_array, short_array, truncate=True)
        assert all(np.equal(result, expected_result))

    def test_download_audio_example(self):
        example_name = 'dev1_female3_inst_mix.wav'
        nussl.utils.print_available_audio_files()
        # nussl.utils.download_audio_example(example_name, '')

    def test_download_hashing(self):
        # example_name = 'dev1_female3_inst_mix.wav'
        # url = 'https://ethman.github.io/nussl-extras/audio/'
        example_name = 'torch-0.3.1-cp27-none-macosx_10_6_x86_64.whl'
        url = 'http://download.pytorch.org/whl/'
        file_url = urljoin(url, example_name)

        # check to make sure downloaded file is removed because of a mismatched hash
        nussl.utils._download_file(example_name, file_url, '', '', file_hash='foobar')
        assert not os.path.isfile(os.path.expanduser('~/.nussl/' + example_name))

        # make sure file is downloaded regardless because no hash was provided
        nussl.utils._download_file(example_name, file_url, '', '')
        assert os.path.isfile(os.path.expanduser('~/.nussl/' + example_name))

        # test ability to provide local_folder to change download location
        nussl.utils._download_file(example_name, file_url, '~/.nussl/local_dir', '')
        assert os.path.isfile(os.path.expanduser('~/.nussl/local_dir' + example_name))

        # check to make sure file isn't downloaded and file already there is used because correct hash is provided
        correct_hash = 'fc0894f970693fcdb369d887c1662ff96a069690747d79a43d18f6115808026b'
        nussl.utils._download_file(example_name, file_url, '', '', file_hash=correct_hash)
        assert os.path.isfile(os.path.expanduser('~/.nussl/' + example_name))




if __name__ == '__main__':
    unittest.main()

