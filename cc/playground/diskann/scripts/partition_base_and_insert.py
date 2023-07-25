import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
### Script params ####
# set path of fbin containing all points
parser.add_argument("--source_file", type=str, required=True, help="Path of fbin containing all points")
# set path of base fbin containing points that you will build index on
parser.add_argument("--base_file", type=str, required=True, help="Path of base fbin containing points that you will build index on")
# set path of insert fbin containing points that you will insert into the built index
parser.add_argument("--insert_file", type=str, required=True, help="Path of insert fbin containing points that you will insert into the built index")
# get num base points
parser.add_argument("--num_base_pts", type=int, required=True, help="Number of base points")

args = parser.parse_args()
source_file = args.source_file
base_file = args.base_file
insert_file = args.insert_file
num_base_pts = args.num_base_pts

# read all data in as np.uint32
# we don't care about manipulating vector data, only file headers
print(f"Reading from {source_file}")
orig_data = np.fromfile(source_file, dtype=np.uint32)
print(f"Finished reading fbin with {orig_data[0]} pts x {orig_data[1]} dims.")

# check if we have enough num points
num_pts, dim = orig_data[0], orig_data[1]
num_insert_pts = num_pts - num_base_pts
assert num_pts == (num_base_pts + num_insert_pts), f"Found {num_pts} pts in file, but requested partitioning of {num_base_pts + num_insert_pts} pts"

# create base set of points
base_end_idx = 2 + (num_base_pts * dim)
base_pts = np.copy(orig_data[:base_end_idx])
base_pts[0] = num_base_pts
print(f"Writing base file with {num_base_pts} pts to {base_file}")
base_pts.tofile(base_file)
base_pts = []

# create insert set of points
insert_pts = np.hstack((orig_data[:2], orig_data[base_end_idx:]))
insert_pts[0] = num_insert_pts
print(f"Writing base file with {num_insert_pts} pts to {insert_file}")
insert_pts.tofile(insert_file)
insert_pts = []

print(f"Done")
