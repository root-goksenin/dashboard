
import os 

import numpy as np 

from scipy.signal import fftconvolve 

from .loaders import load_rir_from_npy


def convolve_with_rir(rir, audio):
    source = np.array(np.array([fftconvolve(audio, rir[i], mode = "full") for i in range(rir.shape[0])]))
    return source 

def add_noise(source, noise, snr):
    r"""Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    """


    L = source.shape[-1]

    if L != noise.shape[-1]:
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.shape[-1]}).")

    else:
        masked_waveform = source
        masked_noise = noise

    energy_signal = np.linalg.norm(masked_waveform, ord=2, axis=-1) ** 2  # (*,)
    energy_noise = np.linalg.norm(masked_noise, ord=2, axis=-1) ** 2  # (*,)
    original_snr_db = 10 * (np.log10(energy_signal) - np.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = np.expand_dims(scale,-1) * noise  # (*, 1) * (*, L) = (*, L)

    return (source + scaled_noise) # (*, L)


def process_audio(sampled_rirs, labels, audio, noise_):
   assert sampled_rirs is not None and (len(sampled_rirs) > 0), "There are no sampled RIRs"
   noise = []
   source = None
   for rir, label in zip(sampled_rirs, labels):
      rir = load_rir_from_npy(f"/projects/0/prjs1338/RIRs/{os.path.basename(rir)}")
      if "noise" in label: 
         convolved_audio = convolve_with_rir(rir, noise_)
         noise.append(convolved_audio)
      elif "source" in label: 
         convolved_audio = convolve_with_rir(rir, audio)
         source = convolved_audio
   
   # If there are noise and sources, add noise to the source and return the audio
   if (source is not None) and (len(noise) > 0):
        max_len = max([x.shape[-1] for x in noise])
        agg_noise = np.array([np.pad(x, ((0,0),(0, max_len - x.shape[-1])), 'constant') for x in noise]).sum(axis = 0)
        agg_noise = agg_noise[:, :source.shape[1]]
        # If audio is bigger than noise, then circular pad to continue playing the noise!
        if agg_noise.shape[1] < source.shape[1]:
            # Calculate the padding amount for the last dimension (columns)
            pad_amount = source.shape[1] - agg_noise.shape[1]
            # Apply circular padding using numpy.pad
            agg_noise = np.pad(
                agg_noise, 
                pad_width=((0, 0), (0, pad_amount)),  # Pad only the last dimension (columns)
                mode='wrap'
            )
        return source, agg_noise
   # If there is no source, just return the first noise.
   elif source is None:
      return noise[0], None
   # If there is source and no noise return the source.
   else:
      return source, None
