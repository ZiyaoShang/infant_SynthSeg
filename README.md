# Infant-SynthSeg

Longitudinal studies of infants' brains are essential for research and clinical detection of Neurodevelopmental Disorders. However, for infant brain MRI scans, effective deep learning-based  segmentation frameworks exist only within small age intervals due the large image intensity and contrast changes that take place in the early postnatal stages of development. However, using different segmentation frameworks or models at different age intervals within the same longitudinal data set would cause segmentation inconsistencies and age-specific biases. Thus, an age-agnostic segmentation model for infants' brains is needed. In this paper, we present "Infant-SynthSeg", an extension of the contrast-agnostic SynthSeg segmentation framework applicable to MRI data of infant at ages within the first year of life. Our work mainly focuses on extending learning strategies related to synthetic data generation and augmentation, with the aim of creating a method that employs training data capturing features unique to infants' brains during this early-stage development. Comparison across different learning strategy settings, as well as a more-traditional contrast-aware deep learning model (NN-Unet) are presented. Our experiments show that our trained Infant-SynthSeg models show consistently high segmentation performance on MRI scans of infant brains throughout the first year of life. Furthermore, as the model is trained on ground truth labels at different ages, even labels that are not present at certain ages (such as cerebellar white matter at 1 month) can be appropriately segmented via Infant-SynthSeg across the whole age range. Finally, while Infant-SynthSeg shows consistent segmentation performance across the first year of life, it is outperformed by age-specific deep learning models trained for a specific narrow age range.

----------------
### Training a model using Infant-SynthSeg 

### Applying our Infant-SynthSeg model on your data

### References

This project is based on the [SynthSeg](https://github.com/BBillot/SynthSeg) framework:

**A Learning Strategy for Contrast-agnostic MRI Segmentation** \
Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca* \
*contributed equally \
MIDL 2020 \
[[link](http://proceedings.mlr.press/v121/billot20a.html) | [arxiv](https://arxiv.org/abs/2003.01995) | [bibtex](bibtex.txt)]

**Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast** \
Benjamin Billot, Eleanor D. Robinson, Adrian V. Dalca, Juan Eugenio Iglesias \
MICCAI 2020 \
[[link](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18) | [arxiv](https://arxiv.org/abs/2004.10221) | [bibtex](bibtex.txt)]
