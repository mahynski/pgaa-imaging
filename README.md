pgaa-imaging
---
This repository accompanies the manuscript: ["Imaging PGAA Spectra for Material Classification with Convolutional Neural Networks"](https://doi.org/10.1007/s10967-025-10165-4).

> We trained deep convolutional neural networks (CNN) to classify materials based on their prompt gamma ray activation analysis (PGAA) spectrum.  We focused on two dimensional (2D) models to leverage abundant open-source models pre-trained on other computer vision tasks for transfer learning.  This allows models to be built with a relatively small number of trainable parameters.  Moreover, CNNs can be explained naturally using class activation maps and can be equipped with out-of-distribution tests to identify materials which were not present in its training set. Together, these features suggest such models may be excellent candidates for automated material identification in real-world scenarios.

Citation
---
Please cite the associated manuscript as follows:

~~~code
@article{
    title={Encoding PGAA spectra as images for material classification with convolutional neural networks},
    authors={Mahynski, Nathan A. and Sheen, David A. and Paul, Rick L. and Chen-Mayer, H. Heather and Shen, Vincent K.},
    journal={Journal of Radioanalytical and Nuclear Chemistry},
    year={2025}
}
~~~

See the CITATION.cff file for how to cite this repository.

Installation
---
Set up the conda environment for this project.
~~~code
$ conda env create -f conda-env.yml
~~~

Hugging Face Models
---

Models and data are also available on [Hugging Face](https://huggingface.co/collections/mahynski/pgaa-spectra-classification-66f7fcd65ea4244ba1b9559b).

Data:
* [pgaa-sample-gadf-images](https://huggingface.co/datasets/mahynski/pgaa-sample-gadf-images)

Models:
* [pgaa-osr-inceptionv3-softmax](https://huggingface.co/mahynski/pgaa-osr-inceptionv3-softmax)
* [pgaa-osr-inceptionv3-energy](https://huggingface.co/mahynski/pgaa-osr-inceptionv3-energy)
* [pgaa-osr-inceptionv3-dime](https://huggingface.co/mahynski/pgaa-osr-inceptionv3-dime)
* [pgaa-osr-mobilenetv3small-softmax](https://huggingface.co/mahynski/pgaa-osr-mobilenetv3small-softmax)
* [pgaa-osr-mobilenetv3small-energy](https://huggingface.co/mahynski/pgaa-osr-mobilenetv3small-energy)
* [pgaa-osr-mobilenetv3small-dime](https://huggingface.co/mahynski/pgaa-osr-mobilenetv3small-dime)
* [pgaa-osr-nasnetmobile-softmax](https://huggingface.co/mahynski/pgaa-osr-nasnetmobile-softmax)
* [pgaa-osr-nasnetmobile-energy](https://huggingface.co/mahynski/pgaa-osr-nasnetmobile-energy)
* [pgaa-osr-nasnetmobile-dime](https://huggingface.co/mahynski/pgaa-osr-nasnetmobile-dime)
