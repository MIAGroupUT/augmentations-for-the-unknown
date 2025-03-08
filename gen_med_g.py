import os

from pathlib import Path

from medg.data import DatasetGenerator

# get file list (see .input_file_globbing.py for more detailed examples)
data_dir = Path(os.path.join('data', 'prostate'))
image_paths = [str(path) for path in sorted(data_dir.glob('test_data/Clean_0/*/*_frame[0-9][0-9].nii.gz'))]
label_paths = [str(path) for path in sorted(data_dir.glob('test_data/Clean_0/*/*_frame[0-9][0-9]_gt.nii.gz'))]

# image_paths = [str(path) for path in sorted(data_dir.glob('test_data/Clean_0/*/t1.nii.gz'))]
# label_paths = [str(path) for path in sorted(data_dir.glob('test_data/Clean_0/*/truth.nii.gz'))]
input_files = [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]
print(input_files)
# specify output path for benchmarking datsaset
# out_path = os.path.join('data', 'ACDC', 'test_data')
out_path = os.path.join('data', 'prostate', 'nnUNet_raw_data', 'Task158_Prostate158', 'test_data')
# create output path if it does not exist
os.makedirs(out_path, exist_ok=True)

# initialize dataset generator
generator = DatasetGenerator(input_files, out_path)

# run generator and save filename mappings
generator.generate_dataset()
generator.save_filename_mappings(Path(out_path) / 'filename_mappings.csv')
