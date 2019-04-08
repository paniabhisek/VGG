VGG
===

This repo is an attempt to implement the paper

| **Very Deep Convolutional Networks for Large-Scale Image Recognition**
| K. Simonyan, A. Zisserman
| ICLR 2015 (**oral**)
| `[arXiv (updated 10 Apr 2015)] <http://arxiv.org/abs/1409.1556/>`_  `[ILSVRC 2014 presentation] <http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf>`_  `[Project page & ILSVRC ConvNet models] <http://www.robots.ox.ac.uk/~vgg/research/very_deep/>`_
| 

in tensorflow. The initial ``data.py``, ``utils.py``, ``logs.py`` is taken from `AlexNet <https://github.com/Abhisek-/AlexNet>`_.

Dataset
-------

Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) **ImageNet Large Scale Visual Recognition Challenge**. arXiv:1409.0575, 2014. `paper <http://arxiv.org/abs/1409.0575>`_ | `bibtex <http://ai.stanford.edu/~olga/bibtex/ILSVRCarxiv14.bib>`_

Dataset info:

- Link: `ILSVRC2010 <http://www.image-net.org/challenges/LSVRC/2010/download-all-nonpub>`_
- Training size: *1261406 images*
- Validation size: *50000 images*
- Test size: *150000 images*
- Dataset size: *124 GB*

To save up time:

I got one corrupted image (``n02487347_1956.JPEG``). The error read: ``Can not identify image file '/path/to/image/n02487347_1956.JPEG n02487347_1956.JPEG``. This happened when I read the image using ``PIL``. Before using this code, please make sure you can open ``n02487347_1956.JPEG`` using ``PIL``. If not delete the image, you won't loose anything if you delete 1 image out of 1 million.

So I trained on ``1261405`` images using *8 GB* GPU.

How to Run
==========

- To train: ``python model.py <path-to-training-data> --train true --test false``
- To test: ``python model.py <path-to-training-data> --train false --test true``

- screenlog-train.0: The log file after running ``python model.py <path-to-training-data> --train true`` in `screen <http://man7.org/linux/man-pages/man1/screen.1.html>`_
- model and logs: `google drive <https://drive.google.com/open?id=1FIXAjopwMHYfXB4_EEDVhxnd0gysoMpI>`_

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

Tensorflow Generated Graphs
---------------------------

**top1 accuracy**:

.. image:: pictures/top1.png

**top5 accuracy**:

.. image:: pictures/top5.png

**loss**:

.. image:: pictures/loss.png

Accuracies
----------

 * Top1 accuracy: **67.1013%**
 * Top5 accuracy: **85.1460%**
