# pyAudioProcessing

![pyaudioprocessing](https://user-images.githubusercontent.com/16875926/63388515-8e66fe00-c35d-11e9-98f5-a7ad0478a353.png)

A Python based library for processing audio data into features and building Machine Learning models.  
This was written using `Python 3.7.6`, and should work with python 3.6+.  


## Getting Started  

1. One way to install pyAudioProcessing and it's dependencies is from PyPI using pip
```
pip install pyAudioProcessing
```  
To upgrade to the latest version of pyAudioProcessing, the following pip command can be used.  
```
pip install -U pyAudioProcessing
```  

2. Or, you could also clone the project and get it setup  

```
git clone git@github.com:jsingh811/pyAudioProcessing.git
cd pyAudioProcessing
pip install -e .
```
and then, get the requirements by running

```
pip install -r requirements/requirements.txt
```

## Training and Classifying Audio files  

### Choices  

Feature options :  
You can choose between features `mfcc`, `gfcc`, `spectral`, `chroma` or a comma separated combination of those, example `gfcc,mfcc,spectral,chroma`, to extract from your audio files.  
Classifier options :  
You can choose between `svm`, `svm_rbf`, `randomforest`, `logisticregression`, `knn`, `gradientboosting` and `extratrees`.  
Hyperparameter tuning is included in the code for each using grid search.  


### Examples  

Command line example of using `gfcc,spectral,chroma` feature and `svm` classifier.   

Training:  
```
python pyAudioProcessing/run_classification.py -f "data_samples/training" -clf "svm" -clfname "svm_clf" -t "train" -feats "gfcc,spectral,chroma"
```  
Classifying:   

```
python pyAudioProcessing/run_classification.py -f "data_samples/testing" -clf "svm" -clfname "svm_clf" -t "classify" -feats "gfcc,spectral,chroma"
```  
Classification results get saved in `classifier_results.json`.  


Code example of using `gfcc,spectral,chroma` feature and `svm` classifier.  
```
from pyAudioProcessing.run_classification import train_and_classify
# Training
train_and_classify("data_samples/training", "train", ["gfcc", "spectral", "chroma"], "svm", "svm_clf")
# Classify data
train_and_classify("data_samples/testing", "classify", ["gfcc", "spectral", "chroma"], "svm", "svm_clf")
```

## Extracting features from audios  

This feature lets the user extract data features calculated on audio files.   

### Choices  

Feature options :  
You can choose between features `mfcc`, `gfcc`, `spectral`, `chroma` or a comma separated combination of those, example `gfcc,mfcc,spectral,chroma`, to extract from your audio files.  
To use your own audio files for feature extraction and pass in the directory containing .wav files as the `-d` argument. Please refer to the format of directory `data_samples/testing`.  

### Examples  

Command line example of for `gfcc` and `mfcc` feature extractions.  

```
python pyAudioProcessing/extract_features.py -f "data_samples/testing"  -feats "gfcc,mfcc"
```  
Features extracted get saved in `audio_features.json`.  

Code example of performing `gfcc` and `mfcc` feature extraction.   
```
from pyAudioProcessing.extract_features import get_features
# Feature extraction
features = get_features("data_samples/testing", ["gfcc", "mfcc"])
```  


## Author  

Jyotika Singh  
Data Scientist  
https://twitter.com/jyotikasingh_/
https://www.linkedin.com/in/jyotikasingh/  
