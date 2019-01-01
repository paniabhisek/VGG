VGG
===

This repo is an attempt to implement the paper

| **Very Deep Convolutional Networks for Large-Scale Image Recognition**
| K. Simonyan, A. Zisserman
| ICLR 2015 (**oral**)
| `[arXiv (updated 10 Apr 2015)] <http://arxiv.org/abs/1409.1556/>`_  `[ILSVRC 2014 presentation] <http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf>`_  `[Project page & ILSVRC ConvNet models] <http://www.robots.ox.ac.uk/~vgg/research/very_deep/>`_
| 

in tensorflow. The initial ``data.py``, ``utils.py``, ``logs.py`` is taken from `AlexNet <https://github.com/Abhisek-/AlexNet>`_.

Preprocessing
-------------

The following preprocessing steps are performed

1. **Rescaling**: Isotropically rescale the image such that the smallest size is randomly drawn from ``[256, 512]``. In short *isotropically* means the ratio of width to height of the original image should match with that of the new image.
2. **Cropping**: Randomly crop the image from the rescaled image to get a size of ``(224, 224)``.
3. **Augmentation**: Augment the data in two ways
     i. Horizontally flip the image with 50 % probability
     ii. Add PCA as calculated by `AlexNet <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_ to the processed image to give color shifting.
4. **Subtract mean**: Finally subtract the mean activity from the processed image.

**Note**: To calculate eigenvalues and eigenvectors for the imagenet dataset will require significant amount of RAM. So the values are taken from `stackoverflow <https://stackoverflow.com/questions/43328600/does-anyone-have-the-eigenvalue-and-eigenvectors-for-alexnets-pca-noise-from-th>`_ and hardcoded while adding PCA.
