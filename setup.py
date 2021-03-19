import setuptools

# The text of the README file
with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
   name='pyAudioProcessing',
   version='1.1.0',
   description='Audio processing-feature extraction and building machine learning models from audio data.',
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Jyotika Singh',
   packages=setuptools.find_packages(),#['pyAudioProcessing'],
   url="https://github.com/jsingh811/pyAudioProcessing",
   include_package_data=True,
   python_requires='>=3.6',
   py_modules=["pyAudioProcessing"],
   classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
       "Operating System :: OS Independent",
   ]
)
