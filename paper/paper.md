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

PyAudioProcessing is a Python based library for processing audio data, constructing and extracting numerical features from audio, building and testing machine learning models and classifying data with existing pre-trained audio classification models or custom user-built models. It in an end-to-end solution for building features from raw audio samples and training a model that can then be used to classify unseen raw audio samples. This library allows the user to extract features such as MFCC, GFCC, spectral features, chroma features and other beat based and cepstrum based features from audio to use with one's own classification backend or popular scikit-learn classifiers that have been built into pyAudioProcessing. This software contributes to the available open-source software by enabling users to use Python based machine learning backend with highly researched audio features such as GFCC and others that are actively user for many audio classification based applications but are not readily available in Python due to primary popularity of research in MATLAB.  

This software aims to provide machine learning engineers, data scientists, researchers and students with a set of baseline models to classify audio, the ability to use this library to build features on custom training data, the ability to automatically train on a scikit-learn classifier and perform hyperparameter tuning using this library, the ability to export the built features for integration with any machine learning backend and the ability to classify audio files. This software furthers aims to aid users in addressing research efforts using GFCC and other evolving and actively researched audio features possible with Python.

# Statement of need

PyAudioProcessing is a Python based library for processing audio data into features and building Machine Learning models. Audio processing and feature extraction research is popular in MATLAB. There are comparatively fewer resources for audio processing and classification in Python. This tool contains implementation of popular and different audio feature extraction that can be use in combination with most scikit-learn classifiers. Audio data can be trained, tested and classified using pyAudioProcessing. The output consists of cross validation scores and results of testing on custom audio files.

The library lets the user extract aggregated data features calculated per audio file. Unique feature extractions such as Mel Frequency Cepstral Coefficients (MFCC) [@6921394], Gammatone Frequency Cepstral Coefficients (GFCC) [@inbook], spectral coefficients, chroma features and others are available to extract and use in combination with different backend classifiers. While MFCC features find use in most commonly encountered audio processing tasks such as audio type classification, speech classification, GFCC features have been found to have application in speaker identification or speaker diarization. Many such applications, comparisons and uses can be found in this IEEE paper [@6639061]. All these features are also helpful for a variety of other audio classification tasks.

Some other popular libraries for the domain of audio processing include librosa [@mcfee2015librosa] and pyAudioAnalysis [@giannakopoulos2015pyaudioanalysis]. Librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems. PyAudioAnalysis is a python library for audio feature extraction, classification, segmentation and applications. It allows the user to train scikit-learn models for mfcc, spectral and chroma features.

PyAudioProcessing adds multiple additional features. The library includes the implementation of GFCC features converted from MATLAB based research to allow users to leverage Python with features for speech classification and speaker identification tasks in addition to MFCC and spectral features that are useful for music and other audio classification tasks. It allows the user to choose from the different feature options and use single or combinations of different audio features. The features can be run through a variety of scikit-learn models including a grid search for best model and Hyperparameters, along with a final confusion matrix and cross validation performance statistics. It further allows for saving and exporting the different audio features per audio file for the user to be able to leverage those while using a different custom classifier backend that is not a part of scikit-learn's models. 

The library further provides some pre-build audio classification models such as `speechVSmusic`, `speechVSmusicVSbirds` sound classifier and `music genre` classifier for give the users a baseline of pre-trained models for their common audio classification tasks. The user can use the library to build custom classifiers with the help of the instructions in the README.

Given the use of this software in the community today inspires the need and growth of this software. It is referenced in a text book titled `Artificial Intelligence with Python Cookbook` published by Packt Publishing in October 2020 [@packt]. Additionally, pyAudioProcessing is a part of specific admissions requirement for a funded PhD project at University of Portsmouth <sup id="portsmouth">[1](#footnote_portsmouth)</sup>. It is further referenced in this thesis paer titled "Master Thesis AI Methodologies for Processing Acoustic Signals AI Usage for Processing Acoustic Signals" [@phdthesis].

<b id="footnote_portsmouth">1</b> https://www.port.ac.uk/study/postgraduate-research/research-degrees/phd/explore-our-projects/detection-of-emotional-states-from-speech-and-text [↩](#portsmouth)

# Audio features

Information about getting started with audio processing is described in [@opensource]. 

![MFCC from audio spectrum.\label{fig:mfcc}](mfcc.png)

The Mel scale relates perceived frequency, or pitch, of a pure tone to its actual measured frequency. Humans are much better at discerning small changes in pitch at low frequencies than they are at high frequencies. Incorporating this scale makes our features match more closely what humans hear. The Mel-frequency scale is approximately linear for frequencies below 1 kHz and logarithmic for frequencies above 1 kHz, as shown in \autoref{fig:mfcc}. This is motivated by the fact that the human auditory system becomes less frequency-selective as frequency increases above 1 kHz.
Passing a spectrum through the Mel filter bank, followed by taking the log magnitude and a discrete cosine transform (DCT) produces the Mel cepstrum. DCT extracts the signal's main information and peaks. It is also widely used in JPEG and MPEG compressions. The peaks are the gist of the audio information. Typically, the first 13 coefficients extracted from the Mel cepstrum are called the MFCCs. These hold very useful information about audio and are often used to train machine learning models. This can be further seen in the form of an illustration in \autoref{fig:mfcc}.

![GFCC from audio spectrum.\label{fig:gfcc}](gfcc.png)

Another filter inspired by human hearing is the Gammatone filter bank. Gammatone filters are conceived to be a good approximation to the human auditory filters and are used as a front-end simulation of the cochlea. Since a human ear is the perfect receiver and distinguisher of speakers in the presence of noise or no noise, construction of gammatone filters that mimic auditory filters became desirable. Thus, it has many applications in speech processing because it aims to replicate how we hear. GFCCs are formed by passing the spectrum through Gammatone filter bank, followed by loudness compression and DCT, as seen in \autoref{fig:gfcc}. The first (approximately) 22 features are called GFCCs. GFCCs have a number of applications in speech processing, such as speaker identification.

Other features useful in audio processing tasks (especially speech) include Linear prediction coefficients and Linear prediction cepstral coefficients (LPCC), Bark frequency cepstral coefficients (BFCC), Power normalized cepstral coefficients (PNCC), and spectral features like spectral flux, entropy, roll off, centroid, spread, and energy entropy.

# References
