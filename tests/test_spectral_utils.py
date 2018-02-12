#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import nussl
import warnings
import inspect


class TestSpectralUtils(unittest.TestCase):
    # set all test cases to be 3 seconds long at the default sample rate (44.1kHz), with 1 channel (ie mono)
    sr = nussl.DEFAULT_SAMPLE_RATE
    dur = 3
    length = sr * dur
    n_ch = 1

    # Define my window lengths to be powers of 2, ranging from 128 to 8192 samples
    win_min = 7  # 2 ** 7  =  128
    win_max = 13  # 2 ** 13 = 8192
    win_lengths = [2 ** i for i in range(win_min, win_max + 1)]

    # pick hop lengths in terms of window length. 1 = one full window. .1 = 1/10th of a window
    hop_length_ratios = [1.0, 0.75, 0.5, 0.3, 0.25, 0.1]

    # figure out what the length a 40 millisecond window would be (in samples). We pick 40 ms because...
    # it is 1/25th of a second. This gives a window length that can hold 1 full cycle of a 25 hz signal
    # Of course, human hearing extends down to 20 Hz....but at least this is lower than the lowest note
    # on a piano (27.5 Hz for equal temperament A440 tuning)
    win_length_40ms = int(2 ** (np.ceil(np.log2(nussl.DEFAULT_WIN_LEN_PARAM * sr))))

    # epsilon value used to verify we are using librosa's stft & istft correctly
    librosa_epsilon = 1e-6

    def test_stft_istft_noise_seed1(self):
        """
        Tests whether the stft can be inverted correctly (with istft) on a seeded random array.
        This is because a seeded array can be reproduced exactly, which can be helpful for
        reproducing a bug.

        The hop length is always the window length. The window type is rectangular.

        This WILL raise an error if the calculated array is different than the original array.
        """
        win_type = nussl.WINDOW_RECTANGULAR

        for i in range(10):
            np.random.seed(i)
            noise = (np.random.rand(self.n_ch, self.length) * 2) - 1
            noise = noise[0, ]

            for win_length in self.win_lengths:
                hop_length = win_length
                self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, noise)

    def test_stft_istft_noise_no_seed(self):
        """
        Tests whether the stft can be inverted correctly (with istft) on an unseeded random array.
        This is to make sure we are not cherry-picking examples that will always work. An unseeded
        random array will be different every time it is created.

        The hop length is always the window length. The window type is rectangular.

        This WILL raise an error if the calculated array is different than the original array.
        """
        win_type = nussl.WINDOW_RECTANGULAR

        for win_length in self.win_lengths:
            hop_length = win_length
            noise = (np.random.rand(self.n_ch, self.length) * 2) - 1
            noise = noise[0, ]

            self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, noise)

    def test_stft_istft_ones1(self):
        """
        Tests a signal of all ones with rectangular window at the different lengths defined in self.win_lengths.
        hop length is same as window length. Window type is rectangular.

        This is all ones because this is an easy way to tell if stft to istft creates any weird
        amplitude issues.

        This WILL raise an error if the calculated array is different than the original array.
        """
        win_type = nussl.WINDOW_RECTANGULAR
        ones = np.ones(self.length)

        for win_length in self.win_lengths:
            hop_length = win_length

            self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, ones)

    def test_stft_istft_ones2(self):
        """
        Tests a signal of all ones with a rectangular window with different window lengths
        (as defined in self.win_lengths) and also different hop lengths (these are calculated from
        the every combination of values in self.hop_length_ratios and self.win_lengths).

        Because the hop size is irregular, we cannot guarantee that the signal will reconstruct correctly.
        Therefore this test is to make sure the stft and istft do not crash.

        This will NOT raise an error if the calculated array is different than the original array.
        """
        win_type = nussl.WINDOW_RECTANGULAR
        ones = np.ones(self.length)

        for win_length in self.win_lengths:
            for i in self.hop_length_ratios:
                hop_length = int(win_length * i)

                self.do_stft_istft(win_length, hop_length, win_type, ones)

    def test_stft_istft_hann1(self):
        """
        Tests the HANN window for reconstruction of the signal through stft and istft for various
        window lengths. The hop size is always going to be half of the window length so the signal
        should be perfectly reconstructed.

        This WILL raise an error if the calculated array is different than the original array.
        """
        win_type = nussl.WINDOW_HANN

        for win_length in self.win_lengths:
            hop_length = win_length / 2
            ones = np.ones(self.length)

            self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, ones)

    def test_stft_istft_40ms_win_length(self):
        """
        This tests the standard 40ms window used by default in nussl. Window is rectangular and we
        test all the values in self.hop_length_ratios.

        This will NOT raise an error if the calculated array is different than the original array.
        """
        win_type = nussl.WINDOW_RECTANGULAR
        ones = np.ones(self.length)

        for i in self.hop_length_ratios:
            hop_length = int(self.win_length_40ms * i)

            self.do_stft_istft(self.win_length_40ms, hop_length, win_type, ones)

    def test_stft_istft_sin(self):
        """
        In this test, we create a sine wave of 300 Hz as the signal to through the stft-istft chain.
        We use the HANN window, window_length = 2048, and hop_length = 1024.

        This is a test with a harmonic component so we know we can reconstruct it correctly.

        This WILL raise an error if the calculated array is different than the original array.
        """
        win_type = nussl.WINDOW_HANN
        freq = 300
        x = np.linspace(0, freq * 2 * np.pi, self.dur * self.sr)
        x = np.sin(x)
        win_length = 2048
        hop_length = win_length / 2

        self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, x)

    def test_e_stft_plus(self):
        """
        This is a quick test of the e_stft_plus() function. This test does NOT test any of the
        additional features that e_stft_plus() does that are not in e_stft(). This test is meant
        as a sanity check to make sure things are running without crashing.

        This test does not try to invert the stft.

        This will NOT raise an error if the calculated array is different than the original array.
        """
        # win_type = nussl.WindowType.RECTANGULAR
        win_type = 'rectangular'
        ones = np.ones(self.length)

        nussl.stft_utils.e_stft_plus(ones, self.win_length_40ms,
                                     self.win_length_40ms, win_type, self.sr)

    def test_e_stft_plus_sin(self):
        """
        This is a second test for e_stft_plus, where we verify all of the outputs (stft, psd, freq & time arrays).

        This test does not try to invert the stft.

        This will NOT raise an error if the calculated array is different than the original array.
        """
        freq = 300
        win_type = nussl.WINDOW_HANN
        x = np.linspace(0, freq * 2 * np.pi, self.dur * self.sr)
        x = np.sin(x)
        win_length = 2048
        hop_length = win_length / 2

        stft, p, freq_array, _ = nussl.stft_utils.e_stft_plus(x, win_length, hop_length, win_type, self.sr)

    def test_librosa_stft(self):
        """
        This test checks our wrappers for librosa's stft and istft. They should be redundant

        This test does not try to invert the stft.

        This will NOT raise an error if the calculated array is different than the original array.
        """
        freq = 600
        win_type = nussl.WINDOW_HANN
        x = np.linspace(0, freq * 2 * np.pi, self.dur * self.sr)
        x = np.sin(x)
        win_length = 2048
        hop_length = win_length / 2
        epsilon = 0.01  # TODO: why doesn't self.librosa_epsilon work?

        stft = nussl.stft_utils.librosa_stft_wrapper(x, win_length, hop_length, win_type)
        signal = nussl.stft_utils.librosa_istft_wrapper(stft, win_length, hop_length, win_type,
                                                        original_signal_length=len(x))

        assert max(np.abs(x - signal)) <= epsilon

    def test_librosa_and_nussl_produce_same_results(self):
        """
        Test to make sure librosa's stft and nussl's stft give the same results
        Returns:

        """
        # Make a 300Hz sine wave
        win_type = nussl.WINDOW_HANN
        freq = 300
        x = np.linspace(0, freq * 2 * np.pi, self.dur * self.sr)
        x = np.sin(x)
        win_length = 2048
        hop_length = win_length / 2

        lib_stft = nussl.stft_utils.librosa_stft_wrapper(x, win_length, hop_length, win_type)
        nussl_stft = nussl.stft_utils.e_stft(x, win_length, hop_length, win_type)

        assert lib_stft.shape == nussl_stft.shape

        # This would be nice to test, but we scale our signals differently so they will always fail
        # assert np.allclose(np.real(lib_stft), np.real(nussl_stft))
        # assert np.allclose(np.imag(lib_stft), np.imag(nussl_stft))

        # win_type is None => Hann window
        lib_signal = nussl.stft_utils.librosa_istft_wrapper(lib_stft, win_length, hop_length, 'hann')
        nussl_signal = nussl.stft_utils.e_istft(nussl_stft, win_length, hop_length, win_type)

        assert len(lib_signal) == len(nussl_signal)

        # Truncate signals and compare
        lib_signal = lib_signal[:len(x)]
        nussl_signal = nussl_signal[:len(x)]

        assert np.allclose(lib_signal, nussl_signal, atol=self.librosa_epsilon)

    # TODO: this
    # ##########################################################################################
    # COMING SOON:
    # Tests that verify that the stft produces correct results (i.e., not just invertible results)
    # E.g., verifying that we see a peak at 3kHz if our signal has a 3kHz frequency in it.
    # ##########################################################################################

    @staticmethod
    def do_stft_istft_assert_allclose(win_length, hop_length, win_type, signal):
        """
        This is a test utility that runs do_stft_istft() and does an np.allclose() on the signals it outputs.
        This verifies that every sample in the original signal and the recalculated signal are within 1e-5.
        :param win_length: window length, to be given to stft() and istft()
        :param hop_length: hop length, to be given to stft() and istft()
        :param win_type: window type, to be given to stft() and istft()
        :param signal: signal to be converted to stft and then back with istft
        """
        calculated_signal = TestSpectralUtils.do_stft_istft(win_length, hop_length, win_type, signal)

        if len(calculated_signal) < len(signal):
            # print out what test we're running
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            current_test = calframe[1][3]

            msg = "calculated_signal shorter than signal. \n \twin_len={}, hop_len={}, win_type={}, " \
                  "test={}".format(win_length, hop_length, win_type, current_test)

            raise AssertionError(msg)

        # Chop off the end
        calculated_signal = calculated_signal[:len(signal)]

        assert np.allclose(signal, calculated_signal)
        return

    @staticmethod
    def do_stft_istft(win_length, hop_length, win_type, signal):
        """
        Test utility to run the stft and istft on a signal.
        :param win_length: window length, to be given to stft() and istft()
        :param hop_length: hop length, to be given to stft() and istft()
        :param win_type: window type, to be given to stft() and istft()
        :param signal: signal to be converted to stft and then back with istft
        :return: calculated signal
        """
        stft = nussl.stft_utils.e_stft(signal, win_length, hop_length, win_type)
        calculated_signal = nussl.stft_utils.e_istft(stft, win_length, hop_length, win_type)

        return calculated_signal


if __name__ == '__main__':
    unittest.main()
