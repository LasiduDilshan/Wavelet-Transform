# Noise Removal and Amplification Using Wavelet Transform

This repository contains the code and resources for removing noise from audio samples using Wavelet Transform. The project utilizes the Wavelet Transform to effectively denoise audio signals, providing a clearer and amplified output.

## Table of Contents

1. [Overview](#overview)
2. [Wavelet Transform](#wavelet-transform)
3. [Signal Denoising using Wavelet Transform](#signal-denoising-using-wavelet-transform)
4. [Python Implementation](#python-implementation)
5. [Code Example](#code-example)
    - [Single File Denoising](#single-file-denoising)
    - [FFT Comparison](#fft-comparison)
    - [Batch Processing for Multiple Files](#batch-processing-for-multiple-files)
6. [References](#references)
7. [Contact](#contact)

## Overview

Wavelet transform is a powerful tool for analyzing signals with non-stationary or time-varying characteristics. Unlike the Fourier transform, which represents a signal purely in the frequency domain, the wavelet transform provides both frequency and time information. This makes it particularly useful for denoising audio signals where noise characteristics vary across different frequency bands.

## Wavelet Transform

The wavelet transform decomposes a signal into a set of basis functions called wavelets. These wavelets are small, localized functions used to analyze different components of a signal at various scales and positions. The two common types of wavelet transforms are:

1. **Continuous Wavelet Transform (CWT)**:
   - Involves convolving the signal with a scaled and translated version of the mother wavelet function at every point in time.
   - Provides a continuous view of the signal's frequency content over time.

2. **Discrete Wavelet Transform (DWT)**:
   - A sampled version of CWT and more computationally efficient.
   - Involves iteratively decomposing a signal into approximation and detail coefficients at different scales.

### Signal Denoising using Wavelet Transform

Wavelet transform is widely used for denoising signals. The general process involves the following steps:

1. **Decomposition**:
   - Apply the DWT to decompose the signal into approximation and detail coefficients at multiple scales.
   - Approximation coefficients represent the coarse, low-frequency components, and detail coefficients represent the high-frequency details.

2. **Thresholding**:
   - Threshold the detail coefficients to remove or attenuate noise using soft or hard thresholding techniques.
   - Soft thresholding sets coefficients below a certain threshold to zero and attenuates others, providing a smoother denoised signal.
   - Hard thresholding sets coefficients below a threshold to zero, effectively removing noise.

3. **Reconstruction**:
   - Reconstruct the denoised signal using the modified coefficients.
   - The approximation coefficients from the highest scale can be omitted if high-frequency details are not crucial.

4. **Iterative Processing (Optional)**:
   - Iteratively apply the above steps to further enhance denoising performance.

## Python Implementation

The denoising process using wavelet transform with Python involves several steps:

1. **Import Libraries**:
   - Import necessary libraries for applying wavelet transform, handling sound files, visualizing waves, and file handling.

2. **Load and Normalize .wav Files**:
   - Normalize the signal to amplify it.

3. **Apply Wavelet Transform to Denoise Signals**:
   - Use the `denoise_wavelet` function from the `skimage.restoration` module to denoise the signals.

4. **Visualize Signals**:
   - Plot the original and denoised signals for comparison.

5. **Save Denoised .wav Files**:
   - Save the denoised signals to new .wav files.

### Code Example

Here's a detailed example demonstrating the process for a single .wav file and a folder containing multiple .wav files:

#### Single File Denoising

```python
from scipy.io import wavfile 
import numpy as np 
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
import soundfile as sf 

# Applying Wavelet transform to remove noise of 1 .wav file as an example
# The example file = 00a49d6b07.wav
Fs, x = wavfile.read("00a49d6b07.wav")
x = x / max(x)  # Normalizing

x_denoise = denoise_wavelet(x, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')

plt.figure(figsize=(20, 10), dpi=100)
plt.plot(x)
plt.plot(x_denoise)
plt.show()

print(len(x_denoise))
print(len(x))
print("Sample rate = ", Fs)  # Sample rate

sf.write('denoise.wav', x_denoise, 16000)  # Writing denoised file into .wav form
```

#### FFT Comparison

```python
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

wav_file_1_path = "00a49d6b07.wav"
wav_file_2_path = "denoise.wav"

sample_rate_1, audio_data_1 = wavfile.read(wav_file_1_path)
sample_rate_2, audio_data_2 = wavfile.read(wav_file_2_path)

if sample_rate_1 != sample_rate_2:
    raise ValueError("Sample rates of both files should be equal!")

num_samples_1 = len(audio_data_1)
num_samples_2 = len(audio_data_2)

fft_data_1 = np.fft.fft(audio_data_1)
fft_data_2 = np.fft.fft(audio_data_2)

fft_frequencies = np.fft.fftfreq(num_samples_1, 1/sample_rate_1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(fft_frequencies[:num_samples_1//2], np.abs(fft_data_1[:num_samples_1//2]))
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("FFT Amplitude", fontsize=12)
plt.title("FFT of Noise Audio", fontsize=14)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(fft_frequencies[:num_samples_1//2], np.abs(fft_data_2[:num_samples_1//2]))
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("FFT Amplitude", fontsize=12)
plt.title("FFT of Denoised Audio", fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### Batch Processing for Multiple Files

```python
import os
import soundfile as sf
from scipy.io import wavfile
from skimage.restoration import denoise_wavelet

folder_path = "C:/Users/HP/Desktop/SP_CUP/Noise Removing/Wavelet Transform/noise audio"  # Replace with the actual path to your folder

wav_file_paths = []

for root, directories, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            wav_file_paths.append(file_path)

output_folder = "C:/Users/HP/Desktop/SP_CUP/Noise Removing/Wavelet Transform/denoise audio"
os.makedirs(output_folder, exist_ok=True)

for file_path in wav_file_paths:
    Fs, x = wavfile.read(file_path)
    x_denoise = denoise_wavelet(x, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')
    x_denoise = x_denoise / max(x_denoise)

    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_folder, filename + ".wav")
    sf.write(output_file_path, x_denoise, 16000)
```

## References

- [In order to de-noise, the coefficients at different frequency levels are used.](https://dergipark.org.tr/tr/download/articlefile/496722#:~:text=In%20order%20to%20de%2Dnoise,coefficients%20at%20different%20frequency%20levels.)
- [Wavelet-based denoising in the IEEE Explore library.](https://ieeexplore.ieee.org/document/8404418)
- [Wavelet denoising tutorial on MathWorks.](https://www.mathworks.com/help/wavelet/ug/wavelet-denoising.html)

## Contact

For any questions or suggestions, please contact the project maintainer.
