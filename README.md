# pyAudioProcessing

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
train_and_classify("data_samples/testing", "classify", ["gfcc"], "svm", "svm_clf"]
```

