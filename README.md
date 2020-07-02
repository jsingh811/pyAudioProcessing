# pyAudioProcessing

![pyaudioprocessing](https://user-images.githubusercontent.com/16875926/63388515-8e66fe00-c35d-11e9-98f5-a7ad0478a353.png)

A Python based library for processing audio data into features and building Machine Learning models.


### Getting Started  

Clone the project and get it setup  

```
git clone git@github.com:jsingh811/pyAudioProcessing.git
pip install -e .
``` 

Get the requirements by running

```
pip install -r requirements/requirements.txt
```


### Choices  

Feature options :  
You can choose between `mfcc`, `gfcc` or `gfcc,mfcc` features to extract from your audio files.  
Classifier options :  
You can choose between `svm`, `svm_rbf`, `randomforest`, `logisticregression`, `knn`, `gradientboosting` and `extratrees`.  
Hyperparameter tuning is included in the code for each using grid search.  
 
 
### Examples  

Command line example of using `gfcc` feature and `svm` classifier.  

Training:  
```
python pyAudioProcessing/run_classification.py -f "data_samples/training" -clf "svm" -clfname "svm_clf" -t "train" -feats "gfcc"
```  
Classifying:   

```
python pyAudioProcessing/run_classification.py -f "data_samples/testing" -clf "svm" -clfname "svm_clf" -t "classify" -feats "gfcc"
```

Code example of using `gfcc` feature and `svm` classifier.  
```
from pyAudioProcessing.run_classification import train_and_classify
# Training
train_and_classify("data_samples/training", "train", ["gfcc"], "svm", "svm_clf")
# Classify data
train_and_classify("data_samples/testing", "classify", ["gfcc"], "svm", "svm_clf")
```

### Author  

Jyotika Singh  
Data Scientist  
https://twitter.com/jyotikasingh_/ 
https://www.linkedin.com/in/jyotikasingh/  



