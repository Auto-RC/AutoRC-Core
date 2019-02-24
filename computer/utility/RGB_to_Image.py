# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import time
import numpy as np
import argparse


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

            data = []

            d = rdata[0]
            for i in d:
                data.append(i[0])

    data = np.array(data).reshape(-1, 128, 96, 3)
    data_len = len(data)
    print("Data size:", data_len)

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





