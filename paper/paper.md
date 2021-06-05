---
title: 'pyAudioProcessing: Audio Processing, Feature Extraction and building Machine Learning Models from Audio Data'
tags:
  - Python
  - audio
  - audio processing
  - feature extraction
  - machine learning
  - gfcc
  - mfcc
  - cepstral coefficients
  - spectral coefficients
authors:
  - name: Jyotika Singh
    orcid: 0000-0002-5442-3004
    affiliation: 1
affiliations:
 - name: Independent researcher
   index: 1
date: 2 June 2021
bibliography: paper.bib

---

# Summary

PyAudioProcessing is a Python based library for processing audio data, forming and extracting numerical features from audio and further building and testing machine learning models. This library allows you to extract features such as MFCC, GFCC, spectral features, chroma features and other beat based and cepstrum based features from audio to use with one's own classification backend or popular scikit-learn classifiers.

# Statement of need

PyAudioProcessing is a Python based library for processing audio data into features and building Machine Learning models. Audio processing and feature extraction research is popular in MATLAB. There are comparatively fewer resources for audio processing and classification in Python. This tool contains implementation of popular and different audio feature extraction that can be use in combination with most scikit-learn classifiers. Audio data can be trained, tested and classified using pyAudioProcessing. The output consists of cross validation scores and results of testing on custom audio files.

The library lets the user extract aggregated data features calculated per audio file. Unique feature extractions such as Mel Frequency Cepstral Coefficients (MFCC) [@6921394], Gammatone Frequency Cepstral Coefficients (GFCC) [@inbook], spectral coefficients, chroma features and others are available to extract and use in combination with different backend classifiers. While MFCC features find use in most commonly encountered audio processing tasks such as audio type classification, speech classification, GFCC features have been found to have application in speaker identification or speaker diarization. Many such applications, comparisons and uses can be found in this IEEE paper [@6639061]. All these features are also helpful for a variety of other audio classification tasks.

# Audio features

Information about getting started with audio processing is described in [@opensource]. 

![MFCC from audio spectrum.\label{fig:mfcc}](mfcc.png)

The Mel scale relates perceived frequency, or pitch, of a pure tone to its actual measured frequency. Humans are much better at discerning small changes in pitch at low frequencies than they are at high frequencies. Incorporating this scale makes our features match more closely what humans hear. The Mel-frequency scale is approximately linear for frequencies below 1 kHz and logarithmic for frequencies above 1 kHz, as shown in \autoref{fig:mfcc}. This is motivated by the fact that the human auditory system becomes less frequency-selective as frequency increases above 1 kHz.
Passing a spectrum through the Mel filter bank, followed by taking the log magnitude and a discrete cosine transform (DCT) produces the Mel cepstrum. DCT extracts the signal's main information and peaks. It is also widely used in JPEG and MPEG compressions. The peaks are the gist of the audio information. Typically, the first 13 coefficients extracted from the Mel cepstrum are called the MFCCs. These hold very useful information about audio and are often used to train machine learning models. This can be further seen in the form of an illustration in \autoref{fig:mfcc}.

![GFCC from audio spectrum.\label{fig:gfcc}](gfcc.png)

Another filter inspired by human hearing is the Gammatone filter bank. Gammatone filters are conceived to be a good approximation to the human auditory filters and are used as a front-end simulation of the cochlea. Since a human ear is the perfect receiver and distinguisher of speakers in the presence of noise or no noise, construction of gammatone filters that mimic auditory filters became desirable. Thus, it has many applications in speech processing because it aims to replicate how we hear. GFCCs are formed by passing the spectrum through Gammatone filter bank, followed by loudness compression and DCT, as seen in \autoref{fig:gfcc}. The first (approximately) 22 features are called GFCCs. GFCCs have a number of applications in speech processing, such as speaker identification.

Other features useful in audio processing tasks (especially speech) include Linear prediction coefficients and Linear prediction cepstral coefficients (LPCC), Bark frequency cepstral coefficients (BFCC), Power normalized cepstral coefficients (PNCC), and spectral features like spectral flux, entropy, roll off, centroid, spread, and energy entropy.

# References
