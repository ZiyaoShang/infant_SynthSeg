# Infant-SynthSeg

Longitudinal studies of infants' brains are essential for research and clinical detection of Neurodevelopmental Disorders. However, for infant brain MRI scans, effective deep learning-based  segmentation frameworks exist only within small age intervals due the large image intensity and contrast changes that take place in the early postnatal stages of development. However, using different segmentation frameworks or models at different age intervals within the same longitudinal data set would cause segmentation inconsistencies and age-specific biases. Thus, an age-agnostic segmentation model for infants' brains is needed. In this paper, we present "Infant-SynthSeg", an extension of the contrast-agnostic SynthSeg segmentation framework applicable to MRI data of infant at ages within the first year of life. Our work mainly focuses on extending learning strategies related to synthetic data generation and augmentation, with the aim of creating a method that employs training data capturing features unique to infants' brains during this early-stage development. Comparison across different learning strategy settings, as well as a more-traditional contrast-aware deep learning model (NN-Unet) are presented. Our experiments show that our trained Infant-SynthSeg models show consistently high segmentation performance on MRI scans of infant brains throughout the first year of life. Furthermore, as the model is trained on ground truth labels at different ages, even labels that are not present at certain ages (such as cerebellar white matter at 1 month) can be appropriately segmented via Infant-SynthSeg across the whole age range. Finally, while Infant-SynthSeg shows consistent segmentation performance across the first year of life, it is outperformed by age-specific deep learning models trained for a specific narrow age range.

----------------
## Training a model using Infant-SynthSeg 
Please put in all the training paremeters/resource directories/configurations in [scripts/niral_scripts/train_infant.py](https://github.com/ZiyaoShang/infant_SynthSeg/blob/master/scripts/niral_scripts/train_infant.py) (additional instructions are provided inside the file) and run 
```
python3 -m scripts.niral_scripts.train_infant 
```
Our training infant label maps are in data/infant_synthseg_training/training_labels.

Our T1 priors are in data/infant_synthseg_training/T1merged.

Our T2 priors are in data/infant_synthseg_training/T2merged.

### If you need to generate intensity priors for your label maps
Please put in the file paths and list of labels to generate in [scripts/niral_scripts/intensity_estimation_infant](https://github.com/ZiyaoShang/infant_SynthSeg/blob/master/scripts/niral_scripts/intensity_estimation_infant.py), then run
```
python3 -m scripts.niral_scripts.intensity_estimation_infant
```

## Applying our Infant-SynthSeg model on your data
Apply a unet model on your data:

```
python3 -m scripts.niral_scripts.predict_infant [grayscale images dir] [dir to save the segmentations] [path to the .h5 model file] --out_posteriors [dir for saving the posterior probabilities]
```
The models we trained are at models/T1.h5 and models/T2.h5
If you are using your own trained model, please update "segmentation_label_list" to what you have trained on at [scripts/niral_scripts/predict_infant](https://github.com/ZiyaoShang/infant_SynthSeg/blob/master/scripts/niral_scripts/predict_infant.py)

## Max-posterior merging

```
python3 -m scripts.niral_scripts.merge_infant [T1 segmentations dir] [T1 posterior dir] [T2 segmentation dir] [T2 posterior dir] [dir to save the merged segmantations] [dir to save the merged posteriors]
```

## Re-merging labels & island removal
Please put in the required resource/saving directories and label list in [scripts/niral_scripts/island_processing](https://github.com/ZiyaoShang/infant_SynthSeg/blob/master/scripts/niral_scripts/island_processing.py) and run:

```
python3 -m scripts.niral_scripts.island_processing 
```

#### Note
The newly added fuse_classes() in [scripts/niral_scripts/train_infant.py](https://github.com/ZiyaoShang/infant_SynthSeg/blob/master/scripts/niral_scripts/train_infant.py) that automates the process generating generation_classes for fused labels might have issues and need more thorough testing. In that case, manually creating the generation_classes may be needed. (an example would be in [scripts/niral_scripts/generate_images_infant](https://github.com/ZiyaoShang/infant_SynthSeg/blob/master/scripts/niral_scripts/generate_images_infant.py)).   

## References

This project is based on the [SynthSeg](https://github.com/BBillot/SynthSeg) framework, where additional tutorials could also be found.

**A Learning Strategy for Contrast-agnostic MRI Segmentation** \
Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca* \
*contributed equally \
MIDL 2020 \
[[link](http://proceedings.mlr.press/v121/billot20a.html) | [arxiv](https://arxiv.org/abs/2003.01995) | [bibtex](bibtex.txt)]

**Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast** \
Benjamin Billot, Eleanor D. Robinson, Adrian V. Dalca, Juan Eugenio Iglesias \
MICCAI 2020 \
[[link](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18) | [arxiv](https://arxiv.org/abs/2004.10221) | [bibtex](bibtex.txt)]
