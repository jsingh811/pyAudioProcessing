# pyAudioProcessing

### Getting Started  

Clone the project and get it setup  

`git clone git@github.com:jsingh811/pyAudioProcessing.git`  

`pip install -e .`  

Get the requirements by running

`pip install -r requirements/requirements.txt`  


### Training  

Choose between `mfcc`, `gfcc` or `gfcc,mfcc` features.  

`python pyAudioProcessing/run_classification.py -f "data_samples/training" -clf "svm" -clfname "svm_clf" -t "train" -feats "gfcc"`  

### Testing  

`python pyAudioProcessing/run_classification.py -f "data_samples/testing" -clf "svm" -clfname "svm_clf" -t "classify" -feats "gfcc"`    
