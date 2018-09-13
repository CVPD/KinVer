# Kiship Verification: Image-based Family Verification in the wild

Facial image analysis has been an important subject of study in the communities of pattern recognition and computer vision. Facial images contain much information about the person they belong to: identity, age, gender, ethnicity, expression and many more.

Visual kinship recognition is a new research topic in the scope of facial image analysis which is essential for many real-world applications (e.g., kinship verification, automatic photo management, social-media application, and more). However, nowadays there exist only a few practical vision systems capable to handle such tasks.

We propose a flexible pipeline composed by modules that assembled together improve the results obtained individually. Our optimized pipeline is based on deep feature extraction, feature selection, multi-metric learning and classifier blending.

Our kinship verification system improves state-of-the-art results that do not use external data for the two public databases KinFace-I and KinFace-II.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Matlab 2016b or above
Statistics and Machine Learning Toolbox for Matlab
```

### Installing

A step by step series of examples that tell you have to get a development env running

The first step is to clone the repository

```
git clone https://github.com/oserradilla/KinVer
```

Now run the setup file that adds external data and datasets to the project. Also download Matlab's feature-selection-library as zip file and extract all its content into a new folder called fisher inside the path src/external; as specified when executing the setup file:
```
setup.m
```

Make sure that the project works correctly by executing the following file and comparing it with the following results:

```
main.m
```
~~accuracyMNRML =
    0.8220    0.8660    0.8940    0.8760~~    
(updated)
```
accuracyMNRML =

    0.8240    0.8720    0.8900    0.8840
```

(optional) All the features have been extracted from the images and included to the project so they are downloaded when cloning the project. If you want to be able to extract the VGG-Face and VGG-F deep features from the dataset by yourself, run the following file in matlab:

```
setupCNNs.m
```
## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/oserradilla/KinVer/tags). 

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Notes

I do not own the code of the external folder. All the folders contain a readme written by the original authors and their reference.
