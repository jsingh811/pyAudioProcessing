# pyAudioProcessing

![pyaudioprocessing](https://user-images.githubusercontent.com/16875926/131924198-e34abe7e-12d8-41f9-926d-db199734dcaa.png)

A Python based library for processing audio data into features (GFCC, MFCC, spectral, chroma) and building Machine Learning models.  
This was written using `Python 3.7.6`, and has been tested to work with Python >= 3.6, <4.  

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
You can also get the requirements by running

```
pip install -r requirements/requirements.txt
```


## Contents  
[Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring)  
[Feature and Classifier model options](https://github.com/jsingh811/pyAudioProcessing#options)  
[Pre-trained models](https://github.com/jsingh811/pyAudioProcessing#classifying-with-pre-trained-models)  
[Extracting numerical features from audio](https://github.com/jsingh811/pyAudioProcessing#extracting-features-from-audios)  
[Building custom classification models](https://github.com/jsingh811/pyAudioProcessing#training-and-classifying-audio-files)  
[Audio cleaning](https://github.com/jsingh811/pyAudioProcessing#audio-cleaning)  
[Audio format conversion](https://github.com/jsingh811/pyAudioProcessing#audio-format-conversion)  
[Audio visualization](https://github.com/jsingh811/pyAudioProcessing#audio-visualization)  

Please refer to the [Wiki](https://github.com/jsingh811/pyAudioProcessing/wiki) for more details.    

## Citation

Using pyAudioProcessing in your research? Please cite as follows.


```
Jyotika Singh. (2021, July 22). jsingh811/pyAudioProcessing: Audio processing, feature extraction and classification (Version v1.2.0). Zenodo. http://doi.org/10.5281/zenodo.5121041
```
[![DOI](https://zenodo.org/badge/197088356.svg)](https://zenodo.org/badge/latestdoi/197088356)


Bibtex
```
@software{jyotika_singh_2021_5121041,
  author       = {Jyotika Singh},
  title        = {{jsingh811/pyAudioProcessing: Audio processing,
                   feature extraction and classification}},
  month        = jul,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.2.0},
  doi          = {10.5281/zenodo.5121041},
  url          = {https://doi.org/10.5281/zenodo.5121041}
}
```


## Options

### Feature options  

You can choose between features `gfcc`, `mfcc`, `spectral`, `chroma` or any combination of those, example `gfcc,mfcc,spectral,chroma`, to extract from your audio files for classification or just saving extracted feature for other uses.  

### Classifier options   

You can choose between `svm`, `svm_rbf`, `randomforest`, `logisticregression`, `knn`, `gradientboosting` and `extratrees`.    
Hyperparameter tuning is included in the code for each using grid search.  


## Training and Testing Data structuring  (Optional)

The library works with data structured as per this section or alternatively with taking an input dictionary object specifying location paths of the audio files.

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

## Classifying with Pre-trained Models

There are three models that have been pre-trained and provided in this project under the /models directory. They are as follows.

`music genre`: Contains SVM classifier to classify audio into 10 music genres - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock. This classifier was trained using mfcc, gfcc, spectral and chroma features. In order to classify your audio files using this classifier, please follow the audio files structuring guidelines. The following commands in Python can be used to classify your data.

`musicVSspeech`: Contains SVM classifier that classifying audio into two possible classes - music and speech. This classifier was trained using mfcc, spectral and chroma features.

`musicVSspeechVSbirds`: Contains SVM classifier that classifying audio into three possible classes - music, speech and birds. This classifier was trained using mfcc, spectral and chroma features.

There are three ways to specify the data you want to classify.  

1. Classifying a single audio file specified by input `file`.

```
from pyAudioProcessing.run_classification import classify_ms, classify_msb, classify_genre

# musicVSspeech classification
results_music_speech = classify_ms(file="/Users/xyz/Documents/audio.wav")

# musicVSspeechVSbirds classification
results_music_speech_birds = classify_msb(file="/Users/xyz/Documents/audio.wav")

# music genre classification
results_music_genre = classify_genre(file="/Users/xyz/Documents/audio.wav")
```

2. Using `file_names` specifying locations of audios as follows.

```
# {"audios_1" : [<path to audio>, <path to audio>, ...], "audios_2": [<path to audio>, ...],}

# Examples.  

file_names = {
	"music" : ["/Users/abc/Documents/opera.wav", "/Users/abc/Downloads/song.wav"],
	"birds": [ "/Users/abc/Documents/b1.wav", "/Users/abc/Documents/b2.wav", "/Users/abc/Desktop/birdsound.wav"]
}

file_names = {
	"audios" : ["/Users/abc/Documents/opera.wav", "/Users/abc/Downloads/song.wav", "/Users/abc/Documents/b1.wav", "/Users/abc/Documents/b2.wav", "/Users/abc/Desktop/birdsound.wav"]
}
```  

The following commands in Python can be used to classify your data.

```
from pyAudioProcessing.run_classification import classify_ms, classify_msb, classify_genre

# musicVSspeech classification
results_music_speech = classify_ms(file_names=file_names)

# musicVSspeechVSbirds classification
results_music_speech_birds = classify_msb(file_names=file_names)

# music genre classification
results_music_genre = classify_genre(file_names=file_names)
```

3. Using data structured as specified in [structuring guidelines](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring) and passing the parent folder path as `folder_path` input.  

The following commands in Python can be used to classify your data.

```
from pyAudioProcessing.run_classification import classify_ms, classify_msb, classify_genre

# musicVSspeech classification
results_music_speech = classify_ms(folder_path="../data")

# musicVSspeechVSbirds classification
results_music_speech_birds = classify_msb(folder_path="../data")

# music genre classification
results_music_genre = classify_genre(folder_path="../data")
```


Sample results look like  
```
{'../data/music': {'beatles.wav': {'probabilities': [0.8899067858599712, 0.011922234412695229, 0.0981709797273336], 'classes': ['music', 'speech', 'birds']}, ...}
```

## Training and Classifying Audio files  

Audio data can be trained, tested and classified using pyAudioProcessing. Please see [feature options](https://github.com/jsingh811/pyAudioProcessing#feature-options) and [classifier model options](https://github.com/jsingh811/pyAudioProcessing#classifier-options) for more information.   

Sample spoken location name dataset for spoken instances of "london" and "boston" can be found [here](https://drive.google.com/drive/folders/1AayPvvgZh4Jvi6LYDR7YS_ar7l3gEtAy?usp=sharing).

### Examples  

Code example of using `gfcc,spectral,chroma` feature and `svm` classifier. Sample data can be found [here](https://github.com/jsingh811/pyAudioProcessing/tree/master/data_samples). 

There are 2 ways to pass the training data in. 

1. Using locations of files in a dictionary format as the input `file_names`.  

2. Passing in a 	`folder_path` containing sub-folders and audio. Please refer to the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring) to use your own data instead.   

```
from pyAudioProcessing.run_classification import  classify, train

# Training
train(
	file_names={
		"music": [<path to audio>, <path to audio>, ..],
		"speech": [<path to audio>, <path to audio>, ..]
	},
	feature_names=["gfcc", "spectral", "chroma"],
	classifier="svm",
	classifier_name="svm_test_clf"
)

```
Or, to use a directory containing audios organized as in [structuring guidelines](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring), the following can be used
```
train(
	folder_path="../data", # path to dir
	feature_names=["gfcc", "spectral", "chroma"],
	classifier="svm",
	classifier_name="svm_test_clf"
)
```

The above logs files analyzed, hyperparameter tuning results for recall, precision and F1 score, along with the final confusion matrix.

To classify audio samples with the classifier you created above,
```
# Classify a single file 

results = classify(
	file = "<path to audio>",
	feature_names=["gfcc", "spectral", "chroma"],
	classifier="svm",
	classifier_name="svm_test_clf"
)

# Classify multiple files with known labels and locations
results = classify(
	file_names={
		"music": [<path to audio>, <path to audio>, ..],
		"speech": [<path to audio>, <path to audio>, ..]
	},
	feature_names=["mfcc", "gfcc", "spectral", "chroma"],
	classifier="svm",
	classifier_name="svm_test_clf"
)

# or you can specify a folder path as described in the training section.
```  
The above logs the filename where the classification results are saved along with the details about testing files and the classifier used if you pass in logfile=True into the function call.


If you cloned the project via git, the following command line example of training and classification with `gfcc,spectral,chroma` features and `svm` classifier can be used as well. Sample data can be found [here](https://github.com/jsingh811/pyAudioProcessing/tree/master/data_samples). Please refer to the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring) to use your own data instead.   

Training:  
```
python pyAudioProcessing/run_classification.py -f "data_samples/training" -clf "svm" -clfname "svm_clf" -t "train" -feats "gfcc,spectral,chroma"
```  
Classifying:   

```
python pyAudioProcessing/run_classification.py -f "data_samples/testing" -clf "svm" -clfname "svm_clf" -t "classify" -feats "gfcc,spectral,chroma" -logfile "../classifier_results"
```  
Classification results get saved in `../classifier_results_svm_clf.json`.  

## Extracting features from audios  

This feature lets the user extract aggregated data features calculated per audio file. See [feature options](https://github.com/jsingh811/pyAudioProcessing#feature-options) for more information on choices of features available.  

### Examples  

Code example for performing `gfcc` and `mfcc` feature extraction can be found below. To use your own audio data for feature extraction, pass the path to `get_features` in place of `data_samples/testing`. Please refer to the format of directory `data_samples/testing` or the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring).  

```
from pyAudioProcessing.extract_features import get_features

# Feature extraction of a single file

features = get_features(
  file="<path to audio>",
  feature_names=["gfcc", "mfcc"]
)

# Feature extraction of a multiple files

features = get_features(
  file_names={
    "music": [<path to audio>, <path to audio>, ..],
    "speech": [<path to audio>, <path to audio>, ..]
  },
  feature_names=["gfcc", "mfcc"]
)

# or if you have a dir with  sub-folders and audios
# features = get_features(folder_path="data_samples/testing", feature_names=["gfcc", "mfcc"])

# features is a dictionary that will hold data of the following format
"""
{
  music: {file1_path: {"features": <list>, "feature_names": <list>}, ...},
  speech: {file1_path: {"features": <list>, "feature_names": <list>}, ...},
  ...
}
"""
```  
To save features in a json file,
```
from pyAudioProcessing import utils
utils.write_to_json("audio_features.json", features)
```  

If you cloned the project via git, the following command line example of for `gfcc` and `mfcc` feature extractions can be used as well. The features argument should be a comma separated string, example `gfcc,mfcc`.  
To use your own audio files for feature extraction, pass in the directory path containing .wav files as the `-f` argument. Please refer to the format of directory `data_samples/testing` or the section on [Training and Testing Data structuring](https://github.com/jsingh811/pyAudioProcessing#training-and-testing-data-structuring).  

```
python pyAudioProcessing/extract_features.py -f "data_samples/testing"  -feats "gfcc,mfcc"
```  
Features extracted get saved in `audio_features.json`.  

## Audio format conversion

You can convert you audio in `.mp4`, `.mp3`, `.m4a` and `.aac` to `.wav`. This will allow you to use audio feature generation and classification functionalities.

In order to convert your audios, the following code sample can be used.  

```
from pyAudioProcessing.convert_audio import convert_files_to_wav

# dir_path is the path to the directory/folder on your machine containing audio files
dir_path = "data/mp4_files"

# simply change audio_format to "mp3", "m4a" or "acc" depending on the format
# of audio that you are trying to convert to wav
convert_files_to_wav(dir_path, audio_format="mp4")

# the converted wav files will be saved in the same dir_path location.

```


## Audio cleaning

To remove low-activity regions from your audio clip, the following sample usage can be referred to.

```
from pyAudioProcessing import clean

clean.remove_silence(
	      <path to wav file>,
               output_file=<path where you want to store cleaned wav file>
)
```

## Audio visualization

To see time-domain view of the audios, and the spectrogram of the audios, please refer to the following sample usage.

```
from pyAudioProcessing import plot

# spectrogram plot
plot.spectrogram(
     <path to wav file>,
    show=True, # set to False if you do not want the plot to show
    save_to_disk=True, # set to False if you do not want the plot to save
    output_file=<path where you want to store spectrogram as a png>
)

# time-series plot
plot.time(
     <path to wav file>,
    show=True, # set to False if you do not want the plot to show
    save_to_disk=True, # set to False if you do not want the plot to save
    output_file=<path where you want to store the plot as a png>
)
```


## Author  

Jyotika Singh  
https://twitter.com/jyotikasingh_/
https://www.linkedin.com/in/jyotikasingh/  
