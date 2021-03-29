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

## Choices  

### Feature options  

You can choose between features `mfcc`, `gfcc`, `spectral`, `chroma` or any combination of those, example `gfcc,mfcc,spectral,chroma`, to extract from your audio files for classification or just saving extracted feature for other uses.  

### Classifier options   

You can choose between `svm`, `svm_rbf`, `randomforest`, `logisticregression`, `knn`, `gradientboosting` and `extratrees`.  
Hyperparameter tuning is included in the code for each using grid search.  


## Training and Testing Data structuring  

Let's say you have 2 classes that you have training data for (music and speech), and you want to use pyAudioProcessing to train a model using available feature options. Save each class as a directory and all the training audio .wav files under the respective class directories. Example:  

```bash
.
├── training_data
├── music
│   ├── music_sample1.wav
│   ├── music_sample2.wav
│   ├── music_sample3.wav
│   ├── music_sample4.wav
├── speech
│   ├── speech_sample1.wav
│   ├── speech_sample2.wav
│   ├── speech_sample3.wav
│   ├── speech_sample4.wav
```  

Similarly, for any test data (with known labels) you want to pass through the classifier, structure it similarly as  

```bash
.
├── testing_data
├── music
│   ├── music_sample5.wav
│   ├── music_sample6.wav
├── speech
│   ├── speech_sample5.wav
│   ├── speech_sample6.wav
```  
If you want to classify audio samples without any known labels, structure the data similarly as  

```bash
.
├── data
├── unknown
│   ├── sample1.wav
│   ├── sample2.wav
```  

## Training and Classifying Audio files  

Audio data can be trained, tested and classified using pyAudioProcessing. Please see [feature options](https://github.com/jsingh811/pyAudioProcessing#feature-options) and [classifier model options](https://github.com/jsingh811/pyAudioProcessing#classifier-options) for more information.   

### Examples  

Code example of using `gfcc,spectral,chroma` feature and `svm` classifier. Sample data can be found [here](https://github.com/jsingh811/pyAudioProcessing/tree/master/data_samples). Please refer to the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring) to use your own data instead.   
```
from pyAudioProcessing.run_classification import train_and_classify
# Training
train_and_classify("data_samples/training", "train", ["gfcc", "spectral", "chroma"], "svm", "svm_clf")
```
The above logs files analyzed, hyperparameter tuning results for recall, precision and F1 score, along with the final confusion matrix.

To classify audio samples with the classifier you created above,
```
# Classify data
train_and_classify("data_samples/testing", "classify", ["gfcc", "spectral", "chroma"], "svm", "svm_clf")
```  
The above logs the filename where the classification results are saved along with the details about testing files and the classifier used.


If you cloned the project via git, the following command line example of training and classification with `gfcc,spectral,chroma` features and `svm` classifier can be used as well. Sample data can be found [here](https://github.com/jsingh811/pyAudioProcessing/tree/master/data_samples). Please refer to the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring) to use your own data instead.   

Training:  
```
python pyAudioProcessing/run_classification.py -f "data_samples/training" -clf "svm" -clfname "svm_clf" -t "train" -feats "gfcc,spectral,chroma"
```  
Classifying:   

```
python pyAudioProcessing/run_classification.py -f "data_samples/testing" -clf "svm" -clfname "svm_clf" -t "classify" -feats "gfcc,spectral,chroma"
```  
Classification results get saved in `classifier_results.json`.  


## Extracting features from audios  

This feature lets the user extract aggregated data features calculated per audio file. See [feature options](https://github.com/jsingh811/pyAudioProcessing#feature-options) for more information on choices of features available.  

### Examples  

Code example for performing `gfcc` and `mfcc` feature extraction can be found below. To use your own audio data for feature extraction, pass the path to `get_features` in place of `data_samples/testing`. Please refer to the format of directory `data_samples/testing` or the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring).  

```
from pyAudioProcessing.extract_features import get_features
# Feature extraction
features = get_features("data_samples/testing", ["gfcc", "mfcc"])
# features is a dictionary that will hold data of the following format
"""
{
  subdir1_name: {file1_path: {"features": <list>, "feature_names": <list>}, ...},
  subdir2_name: {file1_path: {"features": <list>, "feature_names": <list>}, ...},
  ...
}
"""
```  
To save features in a json file,
```
from pyAudioProcessing import utils
utils.write_to_json("audio_features.json",features)
```  

If you cloned the project via git, the following command line example of for `gfcc` and `mfcc` feature extractions can be used as well. The features argument should be a comma separated string, example `gfcc,mfcc`.  
To use your own audio files for feature extraction, pass in the directory path containing .wav files as the `-f` argument. Please refer to the format of directory `data_samples/testing` or the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring).  

```
python pyAudioProcessing/extract_features.py -f "data_samples/testing"  -feats "gfcc,mfcc"
```  
Features extracted get saved in `audio_features.json`.  


## Author  

Jyotika Singh  
Data Scientist  
https://twitter.com/jyotikasingh_/
https://www.linkedin.com/in/jyotikasingh/  
