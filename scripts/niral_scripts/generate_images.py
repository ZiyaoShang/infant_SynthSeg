# Very simple script showing how to generate new images with lab2im

import os
import time
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator
from scripts.niral_scripts.argument_parser import get_argparser

parser = get_argparser()
args = parser.parse_args()
print(args)

# path of the input label map
path_label_map = args.label_dir
# path where to save the generated image
result_dir = args.write_dir

# generate an image from the label map.
# Because the image is spatially deformed, we also output the corresponding deformed label map.
print("Starting ...")
utils.mkdir(result_dir)
print("Created Directory: ", result_dir)
path_label_list = "/home/turja/SynthSeg/generation_labels.txt.npy"
generation_labels = np.load(path_label_list)
print(generation_labels)
brain_generator = BrainGenerator(path_label_map, generation_labels=generation_labels)
for n in range(args.n):
    start = time.time()
    im, lab = brain_generator.generate_brain()
    end = time.time()
    print('generation {0:d} took {1:.01f}s'.format(n, end - start))
    print(im.shape)
    # save output image and label map
    utils.save_volume(np.squeeze(im), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'brain_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(lab), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'labels_%s.nii.gz' % n))
    print("Saved Output.")
print("Done!!!")
