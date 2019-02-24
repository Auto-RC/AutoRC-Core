# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import time
import numpy as np
import argparse
from PIL import Image


# ==================================================================================================
#                                         RGB_to_Image
# ==================================================================================================

def convert(args):

    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        print("Created local directory {}".format(out_dir))

    directory = args.data_dir

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            print("Found data from", os.path.join(directory, filename))
            rdata = np.load(os.path.join(directory, filename))

            t = time.time()
            for counter, i in enumerate(rdata[0]):
                single_rgb = i[0]

                i = numpy2pil(single_rgb)

                i.save(os.path.join(out_dir, "img{}.png".format(counter)), 'PNG')
            print("time taken:", time.time() - t)

def numpy2pil(np_array) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

# ==================================================================================================
#                                            MAIN
# ==================================================================================================

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--data-dir',
        help='Source directory for data.npy files',
    )
    PARSER.add_argument(
        '--out-dir',
        help='Directory to output images to',
    )

    ARGUMENTS, _ = PARSER.parse_known_args()

    convert(ARGUMENTS)





