#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AudioSignal object
"""

from __future__ import division

import copy
import json
import numbers
import os.path
import warnings

import audioread
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

import constants
import stft_utils
import utils

__all__ = ['AudioSignal']


class AudioSignal(object):
    """AudioSignal is the main entry point for the user or source separation algorithm to manipulate audio.

    The AudioSignal class is a container for all things related to audio data. It contains utilities for
    I/O, time-series and frequency domain manipulation, plotting, and much more. The AudioSignal class is used
    in all source separation objects in *nussl*.

    Parameters:
        path_to_input_file (str, optional): Path to an input file to open upon initialization. Audio gets loaded
            into :attr:`audio_data`.
        audio_data_array (:obj:`np.ndarray`, optional): Numpy array containing a real-valued, time-series representation
            of the audio.
        offset (int, optional): Starting point of the section to be extracted in seconds.
            Defaults to 0
        duration (int, optional): Length of the signal to be extracted. Defaults to full length of the signal
        sample_rate (int, optional): sampling rate to read audio file at. Defaults to Constants.DEFAULT_SAMPLE_RATE
        stft (:obj:`np.ndarray`, optional): Optional pre-computed complex spectrogram data.
        stft_params (:obj:`StftParams`, optional):

    Examples:
        * create a new signal object:     ``signal = nussl.AudioSignal('sample_audio_file.wav')``
        * compute the spectrogram of the new signal object:   ``signal.stft()``
        * compute the inverse stft of a spectrogram:          ``sig.istft()``

    See Also:
        For a walk-through of AudioSignal features, see :ref:`audio_signal_basics` and :ref:`audio_signal_stft`.

    Attributes:
        audio_data (:obj:`np.ndarray`):
            Real-valued, uncompressed, time-domain representation of the audio.
            2D numpy array with shape `(n_channels, n_samples)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats if not already.
        path_to_input_file (str): Path to the input file. ``None`` if this AudioSignal never loaded a file, i.e.,
            initialized with a np.array.
        sample_rate (int): Sample rate of this AudioSignal object.
        stft_data (:obj:`np.ndarray`): Complex-valued, frequency-domain representation of audio calculated by the
            Short-Time Fourier Transform (STFT).
            3D numpy array with shape `(n_frequency_bins, n_hops, n_channels)`.
            ``None`` by default, this can be initialized at instantiation.
        stft_params (:obj:`StftParams`): Container for all settings for doing a STFT. Has same lifespan as AudioSignal
            object.
  
    """

    def __init__(self, path_to_input_file=None, audio_data_array=None, stft=None,
                 sample_rate=None, stft_params=None, offset=0, duration=None):

        self.path_to_input_file = path_to_input_file
        self._audio_data = None
        self._stft_data = None
        self._sample_rate = None
        self._active_start = None
        self._active_end = None

        # Assert that this object was only initialized in one way
        got_path = path_to_input_file is not None
        got_audio_array = audio_data_array is not None
        got_stft = stft is not None
        init_inputs = np.array([got_path, got_audio_array, got_stft])

        # noinspection PyPep8
        if len(init_inputs[init_inputs == True]) > 1:  # ignore inspection for clarity
            raise ValueError('Can only initialize AudioSignal object with one of [path, audio, stft]!')

        if path_to_input_file is not None:
            self.load_audio_from_file(self.path_to_input_file, offset, duration, sample_rate)
        elif audio_data_array is not None:
            self.load_audio_from_array(audio_data_array, sample_rate)
        else:
            self._sample_rate = constants.DEFAULT_SAMPLE_RATE

        # stft data
        if stft is not None:
            self.stft_data = stft  # complex spectrogram data

        self.stft_params = stft_utils.StftParams(self.sample_rate) if stft_params is None else stft_params
        self.use_librosa_stft = constants.USE_LIBROSA_STFT

    def __str__(self):
        return self.__class__.__name__

    ##################################################
    #                 Properties
    ##################################################

    # Constants for accessing _audio_data np.array indices
    _LEN = 1
    _CHAN = 0

    _STFT_BINS = 0
    _STFT_LEN = 1
    _STFT_CHAN = 2

    @property
    def signal_length(self):
        """ (int): Number of samples in the active region of `self.audio_data`
            The length of the audio signal represented by this object in samples
        """
        if self.audio_data is None:
            return None
        return self.audio_data.shape[constants.LEN_INDEX]

    @property
    def signal_duration(self):
        """ (float): Duration of audio in seconds.
            The length of the audio signal represented by this object in seconds
        """
        if self.signal_length is None:
            return None
        return self.signal_length / self.sample_rate

    @property
    def num_channels(self):
        """ (int): Number of channels this AudioSignal has.
            Defaults to returning number of channels in :attr:`audio_data`. If that is ``None``, returns number of
            channels in :attr:`stft_data`. If both are ``None`` then returns ``None``.
        """
        # TODO: what about a mismatch between audio_data and stft_data??
        if self.audio_data is not None:
            return self.audio_data.shape[constants.CHAN_INDEX]
        if self.stft_data is not None:
            return self.stft_data.shape[constants.STFT_CHAN_INDEX]
        return None

    @property
    def is_mono(self):
        """
        PROPERTY
        Returns:
            (bool): Whether or not this signal is mono (i.e., has exactly `one` channel).

        """
        return self.num_channels == 1

    @property
    def is_stereo(self):
        """
        PROPERTY
        Returns:
            (bool): Whether or not this signal is stereo (i.e., has exactly `two` channels).

        """
        return self.num_channels == 2

    @property
    def audio_data(self):
        """ (:obj:`np.ndarray`): Real-valued, uncompressed, time-domain representation of the audio.
            2D numpy array with shape `(n_channels, n_samples)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats if not already.
        """
        if self._audio_data is None:
            return None

        start = 0
        end = self._audio_data.shape[constants.LEN_INDEX]

        if self._active_end is not None and self._active_end < end:
            end = self._active_end

        if self._active_start is not None and self._active_start > 0:
            start = self._active_start

        return self._audio_data[:, start:end]

    @audio_data.setter
    def audio_data(self, value):

        if value is None:
            self._audio_data = None
            return

        elif not isinstance(value, np.ndarray):
            raise ValueError('Type of self.audio_data must be of type np.ndarray!')

        if not np.isfinite(value).all():
            raise ValueError('Not all values of audio_data are finite!')

        if value.ndim > 1 and value.shape[constants.CHAN_INDEX] > value.shape[constants.LEN_INDEX]:
            warnings.warn('self.audio_data is not as we expect it. Transposing signal...')
            value = value.T

        if value.ndim > 2:
            raise ValueError('self.audio_data cannot have more than 2 dimensions!')

        if value.ndim < 2:
            value = np.expand_dims(value, axis=constants.CHAN_INDEX)

        self._audio_data = value

        self.set_active_region_to_default()

    @property
    def stft_data(self):
        """ (:obj:`np.ndarray`): Complex-valued, time-frequency representation of the audio.
            2D numpy array with shape `(n_frequency_bins, n_time_bins)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats if not already.
        """

        return self._stft_data

    @stft_data.setter
    def stft_data(self, value):

        if value is None:
            self._stft_data = None
            return

        elif not isinstance(value, np.ndarray):
            raise ValueError('Type of self.stft_data must be of type np.ndarray!')

        if value.ndim == 1:
            raise ValueError('Cannot support arrays with less than 2 dimensions!')

        if value.ndim == 2:
            value = np.expand_dims(value, axis=constants.STFT_CHAN_INDEX)

        if value.ndim > 3:
            raise ValueError('Cannot support arrays with more than 3 dimensions!')

        self._stft_data = value

    @property
    def file_name(self):
        """ (str): The name of the file wth extension, NOT the full path.
        
        Notes:
            This will return ``None`` if this :class:`AudioSignal` object was not loaded from a file.
        
        See Also:
            :attr:`self.path_to_input_file`
        """
        if self.path_to_input_file is not None:
            return os.path.split(self.path_to_input_file)[1]
        return None

    @property
    def sample_rate(self):
        """
        Sample rate for this audio signal.
        Returns:


        """
        return self._sample_rate

    @property
    def time_vector(self):
        """ (:obj:`np.ndarray`): A 1D np array with timestamps (in seconds) for each sample in the time domain.
        """
        if self.signal_duration is None:
            return None
        return np.linspace(0.0, self.signal_duration, num=self.signal_length)

    @property
    def freq_vector(self):
        """ (:obj:`np.ndarray`): A 1D numpy array with frequency values that correspond
        to each frequency bin (vertical axis) for the STFT.
            
        Raises:
            AttributeError: If :attr:`stft_data` is ``None``. Run :func:`stft` before accessing this.
            
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate freq_vector until self.stft() is run')
        return np.linspace(0.0, self.sample_rate // 2, num=self.stft_data.shape[constants.STFT_VERT_INDEX])

    @property
    def time_bins_vector(self):
        """(:obj:`np.ndarray`): A 1D numpy array with time values that correspond
        to each time bin (horizontal axis) in the STFT.
            
        Raises:
            AttributeError: If :attr:`stft_data` is ``None``. Run :func:`stft` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate time_bins_vector until self.stft() is run')
        return np.linspace(0.0, self.signal_duration, num=self.stft_data.shape[constants.STFT_LEN_INDEX])

    @property
    def stft_length(self):
        """ (int): The number of time windows the STFT has.
        Raises:
            AttributeError: If ``self.stft_dat``a is ``None``. Run :func:`stft` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate stft_length until self.stft() is run')
        return self.stft_data.shape[constants.STFT_LEN_INDEX]

    @property
    def num_fft_bins(self):
        """ (int): Number of FFT bins in self.stft_data
        Raises:
            AttributeError: If :attr:`stft_data` is ``None``. Run :func:`stft` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate num_fft_bins until self.stft() is run')
        return self.stft_data.shape[constants.STFT_VERT_INDEX]

    @property
    def active_region_is_default(self):
        """ (bool): True if active region is the full length of :attr:`audio_data`.
        
        See Also:
            * :func:`set_active_region` for a full description of active regions in :class:`AudioSignal`
            * :func:`set_active_region_to_default`

        """
        return self._active_start == 0 and self._active_end == self._signal_length

    @property
    def _signal_length(self):
        """ (int): This is the length of the full signal, not just the active region.
        """
        if self._audio_data is None:
            return None
        return self._audio_data.shape[constants.LEN_INDEX]

    @property
    def power_spectrogram_data(self):
        """ (:obj:`np.ndarray`): Returns a real valued ``np.array`` with power spectrogram data.
        The power spectrogram is defined as (STFT)^2, where ^2 is element-wise squaring
        of entries of the STFT. Same shape as :attr:`stft_data`.
        
        Raises:
            AttributeError: if :attr:`stft_data` is ``None``. Run :func:`stft` before accessing this.
            
        See Also:
            * :attr:`stft_data` complex-valued Short-time Fourier Transform data.
            * :attr:`power_magnitude_data`
            * :func:`get_power_spectrogram_channel`
            
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate power_spectrogram_data because self.stft_data is None')
        return np.abs(self.stft_data) ** 2

    @property
    def magnitude_spectrogram_data(self):
        """ (:obj:`np.ndarray`): Returns a real valued ``np.array`` with magnitude spectrogram data.
        
        The power spectrogram is defined as Abs(STFT), the element-wise absolute value of every item in the STFT. 
        Same shape as :attr:`stft_data`.
        
        Raises:
            AttributeError: if :attr:`stft_data` is ``None``. Run :func:`stft` before accessing this.
            
        See Also:
            * :attr:`stft_data` complex-valued Short-time Fourier Transform data.
            * :attr:`power_spectrogram_data`
            * :func:`get_magnitude_spectrogram_channel`
            
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate magnitude_spectrogram_data because self.stft_data is None')
        return np.abs(self.stft_data)

    @property
    def has_data(self):
        """ Returns False if :attr:`audio_data` and :attr:`stft_data` are empty. Else, returns True.
        
        Returns:
            Returns False if :attr:`audio_data` and :attr:`stft_data` are empty. Else, returns True.
            
        """
        return self.has_audio_data or self.has_stft_data

    @property
    def has_stft_data(self):
        """ Returns False if :attr:`stft_data` is empty. Else, returns True.

        Returns:
            Returns False if :attr:`stft_data` is empty. Else, returns True.

        """
        return self.stft_data is not None and self.stft_data.size != 0

    @property
    def has_audio_data(self):
        """ Returns False if :attr:`audio_data` is empty. Else, returns True.

        Returns:
            Returns False if :attr:`audio_data` is empty. Else, returns True.

        """
        return self.audio_data is not None and self.audio_data.size != 0

    ##################################################
    #                     I/O
    ##################################################

    def load_audio_from_file(self, input_file_path, offset=0, duration=None, new_sample_rate=None):
        # type: (str, float, float) -> None
        """Loads an audio signal from a file

        Parameters:
            input_file_path (str): Path to input file.
            offset (float, optional): The starting point of the section to be extracted (seconds).
                Defaults to 0 seconds.
            duration (float, optional): Length of signal to load in second.
                signal_length of 0 means read the whole file. Defaults to the full length of the signal.
            new_sample_rate (int):

        """
        try:
            with audioread.audio_open(os.path.realpath(input_file_path)) as input_file:
                file_length = input_file.duration

            if offset > file_length:
                raise ValueError('offset is longer than signal!')

            if duration is not None and offset + duration >= file_length:
                warnings.warn('offset + duration are longer than the signal. Reading until end of signal...',
                              UserWarning)

            audio_input, self._sample_rate = librosa.load(input_file_path,
                                                          sr=None,
                                                          offset=offset,
                                                          duration=duration,
                                                          mono=False)

            # Change from fixed point to floating point
            if not np.issubdtype(audio_input.dtype, float):
                audio_input = audio_input.astype('float') / (np.iinfo(audio_input.dtype).max + 1.0)

            self.audio_data = audio_input

            if new_sample_rate is not None and new_sample_rate != self._sample_rate:
                warnings.warn("Input sample rate is different than the sample rate read from the file! Resampling...",
                              UserWarning)
                self.resample(new_sample_rate)

        except Exception as e:
            if isinstance(e, ValueError):  # This is the error we just raise, re-raise it
                raise e
            else:
                raise IOError("Cannot read from file, {file}".format(file=input_file_path))

        self.path_to_input_file = input_file_path
        self.set_active_region_to_default()

    def load_audio_from_array(self, signal, sample_rate=constants.DEFAULT_SAMPLE_RATE):
        """Loads an audio signal from a numpy array.

        Notes:
            Only accepts float arrays and int arrays of depth 16-bits.

        Parameters:
            signal (:obj:`np.ndarray`): Array containing the audio file signal sampled at sampleRate
            sample_rate (int, optional): the sample rate of signal. Default is Constants.DEFAULT_SAMPLE_RATE (44.1kHz)

        """
        assert (type(signal) == np.ndarray)

        self.path_to_input_file = None

        # Change from fixed point to floating point
        if not np.issubdtype(signal.dtype, float):
            if np.max(signal) > np.iinfo(np.dtype('int16')).max:
                raise ValueError('Please convert your array to 16-bit audio.')

            signal = signal.astype('float') / (np.iinfo(np.dtype('int16')).max + 1.0)

        self.audio_data = signal
        self._sample_rate = sample_rate if sample_rate is not None else constants.DEFAULT_SAMPLE_RATE

        self.set_active_region_to_default()

    def write_audio_to_file(self, output_file_path, sample_rate=None, verbose=False):
        """Outputs the audio signal to a file

        Parameters:
            output_file_path (str): Filename where output file will be saved.
            sample_rate (int, optional): The sample rate to write the file at. Default is ``self.sample_rate``.
            verbose (bool, optional): Print out a message if writing the file was successful.
        """
        if self.audio_data is None:
            raise Exception("Cannot write audio file because there is no audio data.")

        try:
            self.peak_normalize()

            if sample_rate is None:
                sample_rate = self.sample_rate

            audio_output = np.copy(self.audio_data)

            # TODO: better fix
            # convert to fixed point again
            if not np.issubdtype(audio_output.dtype, int):
                audio_output = np.multiply(audio_output, 2 ** (constants.DEFAULT_BIT_DEPTH - 1)).astype('int16')

            wav.write(output_file_path, sample_rate, audio_output.T)
        except Exception as e:
            print("Cannot write to file, {file}.".format(file=output_file_path))
            raise e
        if verbose:
            print("Successfully wrote {file}.".format(file=output_file_path))

    ##################################################
    #                Active Region
    ##################################################

    def set_active_region(self, start, end):
        """
        Determines the bounds of what gets returned when you access :attr:`audio_data`.
        None of the data in :attr:`audio_data` is discarded when you set the active region, it merely becomes
        inaccessible until the active region is set back to default (i.e., the full length of the signal).

        This is useful for reusing a single :class:`AudioSignal` object to do multiple operations on only
        select parts of the audio data.

        Warnings:
            Many functions will raise exceptions while the active region is not default. Be aware that adding,
            subtracting, concatenating, truncating, and other utilities may not be available.

        See Also:
            * :func:`set_active_region_to_default`
            * :attr:`active_region_is_default`

        Args:
            start (int): Beginning of active region (in samples). Cannot be less than 0.
            end (int): End of active region (in samples). Cannot be larger than self.signal_length.

        """
        start, end = int(start), int(end)
        self._active_start = start if start >= 0 else 0
        self._active_end = end if end < self._signal_length else self._signal_length

    def set_active_region_to_default(self):
        """
        Resets the active region of this :class:`AudioSignal` object to it default value of the entire
        :attr:`audio_data` array.
        
        See Also:
            * :func:`set_active_region` for an explanation of active regions within the :class:`AudioSignal`.

        """
        self._active_start = 0
        self._active_end = self._signal_length

    def next_window_generator(self, window_size, hop_size, convert_to_samples=False):
        """
        Not Implemented
        
        Raises:
            NotImplemented
            
        Args:
            window_size:
            hop_size:
            convert_to_samples:

        Returns:

        """
        raise NotImplemented
        # start = self._active_start
        # end = self.signal_length
        # if convert_to_samples:
        #     start /= self.sample_rate
        #     end = self.signal_duration
        # old_start = self._active_start
        # self.set_active_region_to_default()
        #
        # while old_start + window_size < self.signal_length:
        #     start = old_start + hop_size
        #     end = start + window_size
        #     self.set_active_region(start, end)
        #     yield start, end

    ##################################################
    #               STFT Utilities
    ##################################################

    def stft(self, window_length=None, hop_length=None, window_type=None, n_fft_bins=None, remove_reflection=True,
             overwrite=True, use_librosa=constants.USE_LIBROSA_STFT):
        """ Computes the Short Time Fourier Transform (STFT) of :attr:`audio_data`.
            The results of the STFT calculation can be accessed from :attr:`stft_data`
            if :attr:`stft_data` is ``None`` prior to running this function or ``overwrite == True``

        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`stft_data`!

        Args:
            window_length (int, optional): Amount of time (in samples) to do an FFT on
            hop_length (int, optional): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str, optional): Type of scaling to apply to the window.
            n_fft_bins (int, optional): Number of FFT bins per each hop
            remove_reflection (bool, optional): Should remove reflection above Nyquist
            overwrite (bool, optional): Overwrite :attr:`stft_data` with current calculation
            use_librosa (bool, optional): Use *librosa's* stft function

        Returns:
            (:obj:`np.ndarray`) Calculated, complex-valued STFT from :attr:`audio_data`, 3D numpy array
            with shape `(n_frequency_bins, n_hops, n_channels)`.

        """
        if self.audio_data is None or self.audio_data.size == 0:
            raise ValueError("No time domain signal (self.audio_data) to make STFT from!")

        window_length = self.stft_params.window_length if window_length is None else int(window_length)
        hop_length = self.stft_params.hop_length if hop_length is None else int(hop_length)
        window_type = self.stft_params.window_type if window_type is None else window_type
        n_fft_bins = self.stft_params.n_fft_bins if n_fft_bins is None else int(n_fft_bins)

        calculated_stft = self._do_stft(window_length, hop_length, window_type,
                                        n_fft_bins, remove_reflection, use_librosa)

        if overwrite:
            self.stft_data = calculated_stft

        return calculated_stft

    def _do_stft(self, window_length, hop_length, window_type, n_fft_bins, remove_reflection, use_librosa):
        if self.audio_data is None or self.audio_data.size == 0:
            raise ValueError('Cannot do stft without signal!')

        stfts = []

        stft_func = stft_utils.librosa_stft_wrapper if use_librosa else stft_utils.e_stft

        for chan in self.get_channels():
            stfts.append(stft_func(signal=chan, window_length=window_length,
                                   hop_length=hop_length, window_type=window_type,
                                   n_fft_bins=n_fft_bins, remove_reflection=remove_reflection))

        return np.array(stfts).transpose((1, 2, 0))

    def istft(self, window_length=None, hop_length=None, window_type=None, overwrite=True,
              use_librosa=constants.USE_LIBROSA_STFT, truncate_to_length=None):
        """ Computes and returns the inverse Short Time Fourier Transform (iSTFT).

        The results of the iSTFT calculation can be accessed from :attr:`audio_data`
        if :attr:`audio_data` is ``None`` prior to running this function or ``overwrite == True``

        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`audio_data`!

        Args:
            window_length (int, optional): Amount of time (in samples) to do an FFT on
            hop_length (int, optional): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str, optional): Type of scaling to apply to the window.
            overwrite (bool, optional): Overwrite :attr:`stft_data` with current calculation
            use_librosa (bool, optional): Use *librosa's* stft function
            truncate_to_length (int, optional): truncate resultant signal to specified length. Default None.

        Returns:
            (:obj:`np.ndarray`) Calculated, real-valued iSTFT from :attr:`stft_data`, 2D numpy array
            with shape `(n_channels, n_samples)`.

        """
        if self.stft_data is None or self.stft_data.size == 0:
            raise ValueError('Cannot do inverse STFT without self.stft_data!')

        window_length = self.stft_params.window_length if window_length is None else int(window_length)
        hop_length = self.stft_params.hop_length if hop_length is None else int(hop_length)
        # TODO: bubble up center
        window_type = self.stft_params.window_type if window_type is None else window_type

        calculated_signal = self._do_istft(window_length, hop_length, window_type, use_librosa)

        # Make sure it's shaped correctly
        calculated_signal = np.expand_dims(calculated_signal, -1) if calculated_signal.ndim == 1 else calculated_signal

        # if truncate_to_length isn't provided
        if truncate_to_length is None:
            if self.signal_length is not None:
                truncate_to_length = self.signal_length

        if truncate_to_length is not None and truncate_to_length > 0:
            calculated_signal = calculated_signal[:, :truncate_to_length]

        if overwrite or self.audio_data is None:
            self.audio_data = calculated_signal

        return calculated_signal

    def _do_istft(self, window_length, hop_length, window_type, use_librosa):
        if self.stft_data.size == 0:
            raise ValueError('Cannot do inverse STFT without self.stft_data!')

        signals = []

        istft_func = stft_utils.librosa_istft_wrapper if use_librosa else stft_utils.e_istft

        for stft in self.get_stft_channels():
            calculated_signal = istft_func(stft=stft, window_length=window_length,
                                           hop_length=hop_length, window_type=window_type)

            signals.append(calculated_signal)

        return np.array(signals)

    def apply_mask(self, mask):
        """
        Applies the input mask to the time-frequency representation in this AudioSignal object and returns a new
        AudioSignal object with the mask applied.
        
        Args:
            mask (:obj:`MaskBase`-derived object): A :ref:`mask_base`-derived object containing a mask

        Returns:
            A new :class:`AudioSignal` object with the input mask applied to the STFT

        """
        # Lazy load to prevent a circular reference upon initialization
        from ..separation.masks import mask_base

        if not isinstance(mask, mask_base.MaskBase):
            raise ValueError('mask is {} but is expected to be a MaskBase-derived object!'.format(type(mask)))

        if mask.shape != self.stft_data.shape:
            raise ValueError('Input mask and self.stft_data are not the same shape! '
                             'mask: {}, self.stft_data: {}'.format(mask.shape, self.stft_data.shape))

        masked_stft = self.stft_data * mask.mask
        return self.make_copy_with_stft_data(masked_stft, verbose=False)

    ##################################################
    #                   Plotting
    ##################################################
    #

    _NAME_STEM = 'audio_signal'

    def plot_time_domain(self, channel=None, x_label_time=True, title=None, file_path_name=None):
        """
        Plots a graph of the time domain audio signal
        Parameters:
            channel (int): The index of the single channel to be plotted
            x_label_time (True): Label the x axis with time (True) or samples (False)
            title (str): The title of the audio signal plot
            file_path_name (str): The output path of where the plot is saved, including the file name

        """

        if self.audio_data is None:
            raise ValueError('Cannot plot with no audio data!')

        if channel > self.num_channels - 1:
            raise ValueError('Channel selected does not exist!')

        # Mono or single specific channel selected for plotting
        if self.num_channels == 1 or channel is not None:
            plot_channels = channel if channel else self.num_channels - 1
            if x_label_time is True:
                plt.plot(self.time_vector, self.audio_data[plot_channels])
                plt.xlim(self.time_vector[0], self.time_vector[-1])
            else:
                plt.plot(self.audio_data[plot_channels])
                plt.xlim(0, self.signal_length)
            channel_num_plot = 'Channel {}'.format(plot_channels)
            plt.ylabel(channel_num_plot)

        # Stereo signal plotting
        elif self.num_channels == 2 and channel is None:
            top_plot = abs(self.audio_data[0])
            bottom_plot = -abs(self.audio_data[1])
            if x_label_time is True:
                plt.plot(self.time_vector, top_plot)
                plt.plot(self.time_vector, bottom_plot, 'C0')
                plt.xlim(self.time_vector[0], self.time_vector[-1])
            else:
                plt.plot(top_plot)
                plt.plot(bottom_plot, 'C0')
                plt.xlim(0, self.signal_length)

        # Plotting more than 2 channels each on their own plots in a stack
        elif self.num_channels > 2 and channel is None:
            f, axarr = plt.subplots(self.num_channels, sharex=True)
            for i in range(self.num_channels):
                if x_label_time is True:
                    axarr[i].plot(self.time_vector, self.audio_data[i])
                    axarr[i].set_xlim(self.time_vector[0], self.time_vector[-1])
                else:
                    axarr[i].plot(self.audio_data[i], sharex=True)
                    axarr[i].set_xlim(0, self.signal_length)
                channel_num_plot = 'Ch {}'.format(i)
                axarr[i].set_ylabel(channel_num_plot)

        if title is None:
            title = self.file_name if self.file_name is not None else self._NAME_STEM

        plt.suptitle(title)

        if file_path_name:
            file_path_name = file_path_name if self._check_if_valid_img_type(file_path_name) \
                                            else file_path_name + '.png'
            plt.savefig(file_path_name)

    def plot_spectrogram(self, file_name=None, ch=None):
        # TODO: use self.stft_data if not None
        # TODO: flatten to mono be default
        # TODO: make other parameters adjustable
        if file_name is None:
            name = self.file_name if self.file_name is not None else self._NAME_STEM + '_spectrogram'
        else:
            name = os.path.splitext(file_name)[0]

        name = name if self._check_if_valid_img_type(name) else name + '.png'

        if ch is None:
            stft_utils.plot_stft(self.to_mono(), name, sample_rate=self.sample_rate)
        else:
            stft_utils.plot_stft(self.get_channel(ch), name, sample_rate=self.sample_rate)

    @staticmethod
    def _check_if_valid_img_type(name):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        result = any([name[-len(k):] == k for k in fig.canvas.get_supported_filetypes().keys()])
        plt.close()
        return result

    ##################################################
    #                  Utilities
    ##################################################

    def concat(self, other):
        """ Concatenate two :class:`AudioSignal` objects (by concatenating :attr:`audio_data`).

        Puts ``other.audio_data`` after :attr:`audio_data`.

        Raises:
            AssertionError: If ``self.sample_rate != other.sample_rate``,
            ``self.num_channels != other.num_channels``, or ``self.active_region_is_default`` is ``False``.

        Args:
            other (:class:`AudioSignal`): :class:`AudioSignal` to concatenate with the current one.
            
        """
        self._verify_audio(other)

        self.audio_data = np.concatenate((self.audio_data, other.audio_data), axis=constants.LEN_INDEX)

    def truncate_samples(self, n_samples):
        """ Truncates the signal leaving only the first ``n_samples`` samples.
        This can only be done if ``self.active_region_is_default`` is True.

        Raises:
            Exception: If ``n_samples > self.signal_length`` or `self.active_region_is_default`` is ``False``.

        Args:
            n_samples: (int) number of samples that will be left.

        """
        if n_samples > self.signal_length:
            raise ValueError('n_samples must be less than self.signal_length!')

        if not self.active_region_is_default:
            raise Exception('Cannot truncate while active region is not set as default!')

        self.audio_data = self.audio_data[:, 0: n_samples]

    def truncate_seconds(self, n_seconds):
        """ Truncates the signal leaving only the first n_seconds.
        This can only be done if self.active_region_is_default is True.

        Raises:
            Exception: If ``n_seconds > self.signal_duration`` or `self.active_region_is_default`` is ``False``.

        Args:
            n_seconds: (float) number of seconds to truncate :attr:`audio_data`.

        """
        if n_seconds > self.signal_duration:
            raise Exception('n_seconds must be shorter than self.signal_duration!')

        if not self.active_region_is_default:
            raise Exception('Cannot truncate while active region is not set as default!')

        n_samples = n_seconds * self.sample_rate
        self.truncate_samples(n_samples)

    def crop_signal(self, before, after):
        """
        Get rid of samples before and after the signal on all channels. Contracts the length
        of self.audio_data by before + after. Useful to get rid of zero padding after the fact.
        Args:
            before: (int) number of samples to remove at beginning of self.audio_data
            after: (int) number of samples to remove at end of self.audio_data

        """
        if not self.active_region_is_default:
            raise Exception('Cannot crop signal while active region is not set as default!')
        num_samples = self.signal_length
        self.audio_data = self.audio_data[:, before:num_samples - after]
        self.set_active_region_to_default()

    def zero_pad(self, before, after):
        """ Adds zeros before and after the signal to all channels.
        Extends the length of self.audio_data by before + after.

        Raises:
            Exception: If `self.active_region_is_default`` is ``False``.

        Args:
            before: (int) number of zeros to be put before the current contents of self.audio_data
            after: (int) number of zeros to be put after the current contents fo self.audio_data

        """
        if not self.active_region_is_default:
            raise Exception('Cannot zero-pad while active region is not set as default!')

        for ch in range(self.num_channels):
            self.audio_data = np.lib.pad(self.get_channel(ch), (before, after), 'constant', constant_values=(0, 0))

    def peak_normalize(self, overwrite=True):
        """ Normalizes ``abs(self.audio_data)`` to 1.0.

            Warnings:
                If :attr:`audio_data` is not represented as floats this will convert the representation to floats!
        """
        max_val = 1.0
        max_signal = np.max(np.abs(self.audio_data))
        if max_signal > max_val:
            normalized = self.audio_data.astype('float') / max_signal
            if overwrite:
                self.audio_data = normalized
            return normalized

    def add(self, other):
        """Adds two audio signal objects.

        This does element-wise addition on the :attr:`audio_data` array.

        Raises:
            AssertionError: If ``self.sample_rate != other.sample_rate``,
            ``self.num_channels != other.num_channels``, or ``self.active_region_is_default`` is ``False``.

        Parameters:
            other (:class:`AudioSignal`): Other :class:`AudioSignal` to add.

        Returns:
            (:class:`AudioSignal`): New :class:`AudioSignal` object with the sum of ``self`` and ``other``.
        """
        self._verify_audio_arithmetic(other)

        new_signal = copy.deepcopy(self)
        new_signal.audio_data = self.audio_data + other.audio_data

        return new_signal

    def subtract(self, other):
        """Subtracts two audio signal objects.

        This does element-wise subtraction on the :attr:`audio_data` array.

        Raises:
            AssertionError: If ``self.sample_rate != other.sample_rate``,
            ``self.num_channels != other.num_channels``, or ``self.active_region_is_default`` is ``False``.

        Parameters:
            other (:class:`AudioSignal`): Other :class:`AudioSignal` to subtract.

        Returns:
            (:class:`AudioSignal`): New :class:`AudioSignal` object with the difference between ``self`` and ``other``.
        """
        other_copy = copy.deepcopy(other)
        other_copy *= -1
        return self.add(other_copy)

    def audio_data_as_ints(self, bit_depth=constants.DEFAULT_BIT_DEPTH):
        """ Returns :attr:`audio_data` as a numpy array of signed ints with a specified bit-depth.

        Available bit-depths are: 8-, 16-, 24-, or 32-bits.

        Raises:
            TypeError: If ``bit_depth`` is not one of the above bit-depths.

        Notes:
            :attr:`audio_data` is regularly stored as an array of floats. This will not affect :attr:`audio_data`.
        Args:
            bit_depth (int, optional): Bit depth of the integer array that will be returned.

        Returns:
            (:obj:`np.ndarray`): Integer representation of :attr:`audio_data`.

        """
        if bit_depth not in [8, 16, 24, 32]:
            raise TypeError('Cannot convert self.audio_data to integer array of bit depth = {}'.format(bit_depth))

        int_type = 'int' + str(bit_depth)

        return np.multiply(self.audio_data, 2 ** (constants.DEFAULT_BIT_DEPTH - 1)).astype(int_type)

    def make_empty_copy(self, verbose=True):
        """ Makes a copy of this :class:`AudioSignal` object with :attr:`audio_data` and :attr:`stft_data`
        initialized to ``np.ndarray``s of the same size, but populated with zeros.

        Returns:
            (:class:`AudioSignal`):

        """
        if not self.active_region_is_default and verbose:
            warnings.warn('Making a copy when active region is not default!')

        new_signal = copy.deepcopy(self)
        new_signal.audio_data = np.zeros_like(self.audio_data)
        new_signal.stft_data = np.zeros_like(self.stft_data)
        return new_signal

    def make_copy_with_audio_data(self, audio_data, verbose=True):
        """ Makes a copy of this `AudioSignal` object with `self.audio_data` initialized to the input `audio_data`
        numpy array.

        Args:
            audio_data:
            verbose (bool): If ``True`` prints warnings. If ``False``

        Returns:

        """
        if verbose:
            if not self.active_region_is_default:
                warnings.warn('Making a copy when active region is not default.')

            if audio_data.shape != self.audio_data.shape:
                warnings.warn('Shape of new audio_data does not match current audio_data.')

        new_signal = copy.deepcopy(self)
        new_signal.audio_data = audio_data
        new_signal.stft_data = None
        return new_signal

    def make_copy_with_stft_data(self, stft_data, verbose=True):
        """ Makes a copy of this `AudioSignal` object with `self.stft_data` initialized to the input `stft_data`
        numpy array.

        Args:
            stft_data:
            verbose:

        Returns:

        """
        if verbose:
            if not self.active_region_is_default:
                warnings.warn('Making a copy when active region is not default.')

            if stft_data.shape != self.stft_data.shape:
                warnings.warn('Shape of new stft_data does not match current stft_data.')

        new_signal = copy.deepcopy(self)
        new_signal.stft_data = stft_data
        new_signal.audio_data = None
        return new_signal

    def to_json(self):
        """ Converts this :class:`AudioSignal` object to JSON.

        See Also:
            :func:`from_json`

        Returns:
            (str): JSON representation of the current :class:`AudioSignal` object.

        """
        return json.dumps(self, default=AudioSignal._to_json_helper)

    @staticmethod
    def _to_json_helper(o):
        if not isinstance(o, AudioSignal):
            raise TypeError

        d = copy.copy(o.__dict__)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = utils.json_ready_numpy_array(v)
        d['__class__'] = o.__class__.__name__
        d['__module__'] = o.__module__
        d['stft_params'] = o.stft_params.to_json()
        return d

    @staticmethod
    def from_json(json_string):
        """ Creates a new :class:`AudioSignal` object from a JSON encoded :class:`AudioSignal` string.

        For best results, ``json_string`` should be created from ``AudioSignal.to_json()``.

        See Also:
            :func:`to_json`

        Args:
            json_string (string): a json encoded :class:`AudioSignal` string

        Returns:
            (obj:`AudioSignal`): an :class:`AudioSignal` object based on the parameters in JSON string

        """
        return json.loads(json_string, object_hook=AudioSignal._from_json_helper)

    @staticmethod
    def _from_json_helper(json_dict):
        if '__class__' in json_dict and '__module__' in json_dict:
            class_name = json_dict.pop('__class__')
            module = json_dict.pop('__module__')
            if class_name != AudioSignal.__name__ or module != AudioSignal.__module__:
                raise TypeError('Expected {}.{} but got {}.{} from json!'.format(AudioSignal.__module__,
                                                                                 AudioSignal.__name__,
                                                                                 module, class_name))

            a = AudioSignal()

            if 'stft_params' not in json_dict:
                raise TypeError('JSON string must contain StftParams object!')

            stft_params = json_dict.pop('stft_params')
            a.stft_params = stft_utils.StftParams.from_json(stft_params)

            for k, v in json_dict.items():
                if isinstance(v, dict) and constants.NUMPY_JSON_KEY in v:
                    a.__dict__[k] = utils.json_numpy_obj_hook(v[constants.NUMPY_JSON_KEY])
                else:
                    a.__dict__[k] = v if not isinstance(v, unicode) else v.encode('ascii')
            return a
        else:
            return json_dict

    def rms(self):
        """ Calculates the root-mean-square of :attr:`audio_data`.
        
        Returns:
            (float): Root-mean-square of :attr:`audio_data`.

        """
        return np.sqrt(np.mean(np.square(self.audio_data)))

    def get_closest_frequency_bin(self, freq):
        """
        Returns index of the closest element to freq
        
        Args:
            freq: (int) frequency to retrieve in Hz

        Returns: 
            (int) index of closest frequency to input freq

        Example:
            
            .. code-block:: python
                :linenos:
                # Make a low pass filter starting around 1200 Hz
                signal = nussl.AudioSignal('path_to_song.wav')
                signal.stft()
                idx = signal.get_closest_frequency_bin(1200)  # 1200 Hz
                signal.stft_data[idx:, :, :] = 0.0  # eliminate everything above idx

        """
        if self.freq_vector is None:
            raise ValueError('Cannot get frequency bin until self.stft() is run!')
        return (np.abs(self.freq_vector - freq)).argmin()

    def apply_gain(self, value):
        """
        Apply a gain to self.audio_data
        Args:
            value: (float) amount to multiply self.audio_data by

        """
        if not isinstance(value, numbers.Real):
            raise ValueError('Can only multiply/divide by a scalar!')

        self.audio_data = self.audio_data * value
        return self

    def resample(self, new_sample_rate):
        """
        Resample an audio signal using resample
        Args:
            new_sample_rate: (int) The new sample rate to apply to the audio signal

        """

        if new_sample_rate == self.sample_rate:
            warnings.warn('Cannot resample to the same sample rate.')
            return

        resampled_signal = []

        for channel in self.get_channels():
            resampled_channel = librosa.resample(channel, self.sample_rate, new_sample_rate)
            resampled_signal.append(resampled_channel)

        self.audio_data = np.array(resampled_signal)
        self._sample_rate = new_sample_rate

    ##################################################
    #              Channel Utilities
    ##################################################

    def _verify_get_channel(self, n):
        if n >= self.num_channels:
            raise ValueError('Cannot get channel {0} when this object only has {1} channels! (0-based)'
                             .format(n, self.num_channels))

        if n < 0:
            raise ValueError('Cannot get channel {}. This will cause unexpected results'.format(n))

    def get_channel(self, n):
        """Gets audio data of n-th channel from :attr:`audio_data` as a 1D ``np.ndarray`` of shape (n_samples,).

        Raises:
            Exception: If not ``0 <= n < self.num_channels``.

        Parameters:
            n (int): index of channel to get. **0-based**
            
        Returns:
            (:obj:`np.array`): The audio data in the n-th channel of the signal, 1D
            
        See Also:
            * :func:`get_channels`: Generator for looping through channels of :attr:`audio_data`.
            * :func:`get_stft_channel`: Gets stft data from a specific channel.
            * :func:`get_stft_channels`: Generator for looping through channels from :attr:`stft_data`.
        """
        self._verify_get_channel(n)

        return utils._get_axis(self.audio_data, constants.CHAN_INDEX, n)

    def get_channels(self):
        """Generator that will loop through channels of :attr:`audio_data`.

        Yields:
            (:obj:`np.array`): The audio data in the next channel of this signal as a 1D ``np.ndarray``.
            
        See Also:
            * :func:`get_channel`: Gets audio data from a specific channel.
            * :func:`get_stft_channel`: Gets stft data from a specific channel.
            * :func:`get_stft_channels`: Generator for looping through channels from :attr:`stft_data`.

        """
        for i in range(self.num_channels):
            yield self.get_channel(i)

    def get_stft_channel(self, n):
        """Returns STFT data of n-th channel from :attr:`stft_data` as a 2D ``np.ndarray``.

        Raises:
            Exception: If not ``0 <= n < self.num_channels``.

        Args:
            n: (int) index of stft channel to get. **0-based**

        Returns:
            (:obj:`np.array`): the stft data in the n-th channel of the signal, 2D
            
        See Also:
            * :func:`get_stft_channels`: Generator for looping through channels from :attr:`stft_data`.
            * :func:`get_channel`: Gets audio data from a specific channel.
            * :func:`get_channels`: Generator for looping through channels of :attr:`audio_data`.
        """
        self._verify_get_channel(n)

        return utils._get_axis(self.stft_data, constants.STFT_CHAN_INDEX, n)

    def get_stft_channels(self):
        """Generator that will loop through channels of :attr:`stft_data`.

        Yields:
            (:obj:`np.array`): The STFT data in the next channel of this signal as a 2D ``np.ndarray``.
            
        See Also:
            * :func:`get_stft_channel`: Gets stft data from a specific channel.
            * :func:`get_channel`: Gets audio data from a specific channel.
            * :func:`get_channels`: Generator for looping through channels of :attr:`audio_data`.

        """
        for i in range(self.num_channels):
            yield self.get_stft_channel(i)

    def make_audio_signal_from_channel(self, n):
        """
        Makes a new :class:`AudioSignal` object from with data from channel ``n``.
        
        Args:
            n (int): index of channel to make a new signal from. **0-based**

        Returns:
            (:class:`AudioSignal`) new :class:`AudioSignal` object with only data from channel ``n``.

        """
        new_signal = copy.copy(self)
        new_signal.audio_data = self.get_channel(n)
        new_signal.stft_data = self.get_stft_channel(n)
        return new_signal

    def get_power_spectrogram_channel(self, n):
        """ Returns the n-th channel from ``self.power_spectrogram_data``.

        Raises:
            Exception: If not ``0 <= n < self.num_channels``.

        Args:
            n: (int) index of power spectrogram channel to get **0-based**

        Returns:
            (:obj:`np.array`): the power spectrogram data in the n-th channel of the signal, 1D
        """
        self._verify_get_channel(n)

        # np.array helps with duck typing
        return utils._get_axis(np.array(self.power_spectrogram_data), constants.STFT_CHAN_INDEX, n)

    def get_magnitude_spectrogram_channel(self, n):
        """ Returns the n-th channel from ``self.magnitude_spectrogram_data``.

        Raises:
           Exception: If not ``0 <= n < self.num_channels``.

        Args:
            n: (int) index of magnitude spectrogram channel to get **0-based**

        Returns:
            (:obj:`np.array`): the magnitude spectrogram data in the n-th channel of the signal, 1D
        """
        self._verify_get_channel(n)

        # np.array helps with duck typing
        return utils._get_axis(np.array(self.magnitude_spectrogram_data), constants.STFT_CHAN_INDEX, n)

    def to_mono(self, overwrite=False, keep_dims=True):
        """ Converts :attr:`audio_data` to mono by averaging every sample.

        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`audio_data`!

        Args:
            overwrite (bool): If `True` this function will overwrite :attr:`audio_data`.
            keep_dims (bool): If `True` this function will return a 1D array,
            else will return array with shape `(1, n_samples)`.
        Returns:
            (:obj:`np.array`): Mono-ed version of :attr:`audio_data`.

        """
        mono = np.mean(self.audio_data, axis=constants.CHAN_INDEX, keepdims=keep_dims)

        if overwrite:
            self.audio_data = mono
        return mono

    def stft_to_one_channel(self, overwrite=False):
        """ Converts :attr:`stft_data` to a single channel by averaging every sample.
        Shape: stft_data.shape will be (num_freq, num_time, 1) where the last axis is the channel number

        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`stft_data`!

        Args:
            overwrite (bool, optional): If ``True`` this function will overwrite :attr:`stft_data`.

        Returns:
            (:obj:`np.array`): Single channel version of :attr:`stft_data`.

        """
        one_channel_stft = np.mean(self.stft_data, axis=constants.CHAN_INDEX)
        if overwrite:
            self.stft_data = one_channel_stft
        return one_channel_stft

    ##################################################
    #              Operator overloading
    ##################################################

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def _verify_audio(self, other):
        if self.num_channels != other.num_channels:
            raise Exception('Cannot do operation with two signals that have a different number of channels!')

        if self.sample_rate != other.sample_rate:
            raise Exception('Cannot do operation with two signals that have different sample rates!')

        if not self.active_region_is_default:
            raise Exception('Cannot do operation while active region is not set as default!')

    def _verify_audio_arithmetic(self, other):
        self._verify_audio(other)

        if self.signal_length != other.signal_length:
            raise ValueError('Cannot do arithmetic with signals of different length!')

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __mul__(self, value):
        if not isinstance(value, numbers.Real):
            raise ValueError('Can only multiply/divide by a scalar!')

        return self.make_copy_with_audio_data(np.multiply(self.audio_data, value), verbose=False)

    def __div__(self, value):
        if not isinstance(value, numbers.Real):
            raise ValueError('Can only multiply/divide by a scalar!')

        return self.make_copy_with_audio_data(np.divide(self.audio_data, float(value)), verbose=False)

    def __truediv__(self, value):
        return self.__div__(value)

    def __itruediv__(self, value):
        return self.__idiv__(value)

    def __imul__(self, value):
        return self.apply_gain(value)

    def __idiv__(self, value):
        return self.apply_gain(1 / float(value))

    def __len__(self):
        return self.signal_length

    def __eq__(self, other):
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, other.__dict__[k]):
                    return False
            elif v != other.__dict__[k]:
                return False
        return True

    def __ne__(self, other):
        return not self == other
