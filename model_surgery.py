import os
import gc
import shutil
import pickle

from pathlib import Path

import ot
import pandas as pd
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from einops import rearrange
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from ml.model.segmentation.nnunet import process_plans_and_init, prepare_scan, crop_to_original_shape

path_to_data = Path('data')

path_to_acdc_test = path_to_data / 'ACDC' / 'test_data'  # where I kept my data for each transformation and severity in a 
                                                         # seperate .tar.gz


def get_unique_scan_names(scans: list, scans_are_channel_split=False):
    # get the unique scan names
    if scans_are_channel_split:
        _scan_names = set([_scan.name[:-12] for _scan in scans])
    else:
        _scan_names = set([_scan.name[:-7] for _scan in scans])
    return list(sorted(_scan_names))


def add_channel_if_missing(_scan: np.ndarray):
    # if the num of dimensions is 3, add a new axis at the beginning
    if len(_scan.shape) == 3:
        _scan = _scan[None]
    return _scan


def handle_channel_split_files(_parent_dir: Path, _scan_name: str):
    _scan_name_paths = [_parent_dir / f'{_scan_name}_000{i}.nii.gz' for i in range(2)]
    nib_files = [nib.load(_scan_file) for _scan_file in _scan_name_paths]
    _scan = [add_channel_if_missing(nib_file.get_fdata()) for nib_file in nib_files]
    _scan = np.concatenate(_scan, axis=0)
    return _scan


def handle_single_file(_scan_file: Path):
    _nib_file = nib.load(_scan_file)
    _scan = _nib_file.get_fdata()
    _scan = add_channel_if_missing(_scan)
    return _scan


def load_label(_label_file: Path):
    _nib_file = nib.load(_label_file)
    _label = _nib_file.get_fdata()
    return _label


def find_margin(gt, segmentation, latent):
    probabilities = torch.nn.functional.softmax(segmentation, dim=1)
    correct_indices = torch.argmax(gt, 1, keepdim=True)
    correct_probability = torch.gather(probabilities, 1, correct_indices)

    temp = probabilities.clone().detach()
    temp.scatter_(1, correct_indices, -1)
    max_not_correct_probability_indices = torch.argmax(temp, 1, keepdim=True)

    del temp

    max_not_correct_probability = torch.gather(probabilities, 1, max_not_correct_probability_indices)

    [grad_a] = torch.autograd.grad(
        correct_probability,
        [latent],
        grad_outputs=torch.ones_like(correct_probability, requires_grad=False),
        retain_graph=True
    )
    [grad_b] = torch.autograd.grad(
        max_not_correct_probability,
        [latent],
        grad_outputs=torch.ones_like(max_not_correct_probability, requires_grad=False),
        retain_graph=False
    )
    grad = grad_a - grad_b
    point_wise_grad_norm = torch.norm(grad, dim=[1, 2, 3], keepdim=True, p=2)

    _margin = correct_probability - max_not_correct_probability

    return _margin, probabilities, point_wise_grad_norm


def create_latent_space_dataset(
        _model_variant, _parent_dir_scans, _parent_dir_labels, _scans_are_channel_split=False, fold_num=0
):
    # the dataset needs to record the latent space output for each scan at each depth output from the model
    # dataset columns: model_variant, fold, scan_name, depth, path_to_latent_space_matrix, path_to_softmax_prediction_matrix, path_to_gt
    _plans_file = _model_variant / 'plans.pkl'
    with open(_plans_file, 'rb') as f:
        _plans = pickle.load(f)
    _num_classes = _plans['num_classes'] + 1

    for _fold in [
        _fold_path
        for _fold_path in _model_variant.iterdir() if
        _fold_path.is_dir() and _fold_path.name.startswith(f'fold_{fold_num}')
    ]:
        # load the model
        _model_file = _fold / 'model_final_checkpoint.model'
        _model = process_plans_and_init(_plans, _model_file, deep_supervision=True, return_latent=True)
        _model = _model.cuda()
        _model.eval()

        # create inside the fold directory a new directory to store the new data being generated
        _output_dir = _fold / 'esoteric' / _parent_dir_scans.parent.name
        _output_dir.mkdir(exist_ok=True, parents=True)

        _path_to_all_scans = list(_parent_dir_scans.glob('*.nii.gz'))
        # only keep 10 scans for now
        indices_to_keep, _ = train_test_split(
            np.arange(len(_path_to_all_scans)),
            train_size=2,
            random_state=42
        )
        _path_to_all_scans = [_path_to_all_scans[_index] for _index in indices_to_keep]
        _scan_names = get_unique_scan_names(_path_to_all_scans, _scans_are_channel_split)

        _path_to_all_labels = [
            Path(_parent_dir_labels) / f'{_parent_dir_scans.parent.name}_{_scan_name}.nii.gz'
            for _scan_name in _scan_names
        ]

        for (_scan_name, _label_path) in tqdm(
                zip(_scan_names, _path_to_all_labels),
                total=len(_path_to_all_labels),
                desc=_model_variant.name.replace('__nnUNetPlansv2.1', '').replace('nnUNetTrainerV2_', '')
                        .replace('_', ' + ')
                        .replace('nnUNetTrainerV2', 'Base')
                        .replace('nnUNetTrainerNoDAV2', 'No Aug')
        ):
            if _scans_are_channel_split:
                _test_scan = handle_channel_split_files(_parent_dir_scans, _scan_name)
            else:
                _test_scan = handle_single_file(_parent_dir_scans / f'{_scan_name}.nii.gz')
            _test_label = load_label(_label_path)
            _test_scan, _test_label, _original_shape = prepare_scan(_test_scan, _test_label, num_classes=_num_classes)

            with torch.enable_grad():
                _test_scan.requires_grad_(True)
                _model.requires_grad_(True)
                _latent_output, _segmentation_output = _model(_test_scan)

            if not isinstance(_latent_output, tuple):
                _latent_output = [_latent_output]
            if not isinstance(_segmentation_output, tuple):
                _segmentation_output = [_segmentation_output]

            for _depth, (_segmentation, _latent) in enumerate(zip(_latent_output, _segmentation_output)):
                if _depth != 0:
                    continue

                if _depth != 0:
                    _gt = torch.nn.functional.interpolate(_test_label, size=_segmentation.shape[2:], mode='nearest')
                else:
                    _gt = _test_label
                # save the latent space and the softmax prediction
                _latent_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_latent.npy'
                _segmentation_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_seg.npy'
                _gt_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_gt.npy'
                _margin_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_margin.npy'

                _margin, _probabilities, _grad_norm = find_margin(_gt, _segmentation, _latent)
                _latent = _latent.detach().cpu().numpy()
                _margin, _grad_norm = _margin.detach().cpu().numpy(), _grad_norm.detach().cpu().numpy()
                _gn_margin = _margin / (_grad_norm + 1e-6)

                _gt = torch.argmax(_gt, 1, keepdim=True).detach().cpu().numpy()
                np.save(_latent_output_path, _latent)
                np.save(_segmentation_output_path, _probabilities.detach().cpu().numpy())
                np.save(_gt_output_path, _gt)
                np.save(_margin_output_path, _gn_margin)

            del _latent_output, _segmentation_output, _latent, _segmentation, _gt, _margin, _probabilities
            torch.cuda.empty_cache()

        del _model, _output_dir, _path_to_all_labels, _path_to_all_scans
        torch.cuda.empty_cache()

        gc.collect()


def compute_kv_gn_margin_and_pca(severity, depth, model_variant, store_pcas_in, fold_num):
    # there is a folder inside each model_variant / fold_0 that is called esoteric, since we are working with esoteric information :)
    esoteric_folder = Path(model_variant, f'fold_{fold_num}', 'esoteric')
    if esoteric_folder.exists():
        print(esoteric_folder)
    else:
        return

    latents = []
    gts = []
    margins = []
    for esoteric_file in esoteric_folder.rglob('*_latent.npy'):
        if f"_{severity}" not in str(esoteric_file):
            continue

        if esoteric_file.name.endswith(f'_depth_{depth}_latent.npy'):
            # get the corresping gt file
            gt_file = Path(esoteric_file.parent, esoteric_file.name.replace('latent', 'gt'))
            margin_file = Path(esoteric_file.parent, esoteric_file.name.replace('latent', 'margin'))
            latent = np.load(esoteric_file)
            gt = np.load(gt_file)
            margin = np.load(margin_file)

            latent = rearrange(latent, 'd l h w -> (d h w) l')
            gt = rearrange(gt, 'd 1 h w -> (d h w)')
            margin = rearrange(margin, 'd 1 h w -> (d h w)')

            # sample at most 1000 points for each class
            for c in range(4):
                if len(np.where(gt == c)[0]) == 0:
                    continue
                class_indices = np.where(gt == c)[0]
                np.random.shuffle(class_indices)
                latents.append(latent[class_indices]])
                gts.append(gt[class_indices]])
                margins.append(margin[class_indices]])

    latents = np.concatenate(latents, axis=0)
    gts = np.concatenate(gts, axis=0)
    margins = np.concatenate(margins, axis=0)

    gamma = 0.
    indices_for_costs = []
    costs = []

    for c in range(4):  # TODO: change it for your number of classes
        if len(np.where(gts == c)[0]) == 0:
            continue

        class_indices = np.where(gts == c)[0]

        # shuffle it
        np.random.shuffle(class_indices)
        k = min(int(len(class_indices) / 2), 1000)  # computational limit. The authors of k-Variance hoever motivate smaller numbers are also okay.
        features_of_class = latents[class_indices]
        f1, f2 = features_of_class[:k], features_of_class[k:2 * k]
        unif = np.ones((len(f1),)) / len(f1)
        cost, log = ot.emd2(unif, unif, M=pairwise_distances(f1, f2), return_matrix=True)
        cost_matrix = log['G']
        # take the sum of the costs for each point and add the row to costs
        costs.append(np.sum(cost_matrix, axis=1))
        indices_for_costs.append(class_indices[:k])

        gamma += cost * (len(class_indices) / len(gts))

    print(f'Median Margin: {np.median(margins)}')
    print(f'kv-Normalised Margin: {np.median(margins) / gamma}')

    # we need the ground truth. The coulours here magic numbers originating from ACDC and MnMS as I have only looked at most 4
    # number of classes.
    colours = np.zeros_like(gts)
    colours[gts == 0] = 0
    colours[gts == 1] = 1
    colours[gts == 2] = 2
    colours[gts == 3] = 3

    # get 1000 random indices of each class
    rand_indices = []
    for i in range(4):
        of_class = np.where(colours == i)[0]
        rand_indices.extend(np.random.choice(of_class, min(1000, len(of_class)), replace=False))

    rand_indices = np.array(rand_indices)

    # perform PCA on the latent space
    pca = PCA(n_components=2)
    dim_red = pca.fit_transform(latents[rand_indices])

    indices_to_plot = np.where(colours[rand_indices] < 4)[0]
    plt.scatter(
        dim_red[indices_to_plot, 0], dim_red[indices_to_plot, 1],
        c=colours[rand_indices][indices_to_plot], alpha=0.5, cmap='jet'
    )

    # save the pca components in a csv file with columns point, PC1, PC2, label (colour)
    # sample only 500 total points for saving
    indices_to_save = np.random.choice(len(indices_to_plot), 500, replace=False)
    df = pd.DataFrame(dim_red[indices_to_plot][indices_to_save], columns=['PC1', 'PC2'])
    df['label'] = colours[rand_indices][indices_to_plot][indices_to_save]

    save_path = store_pcas_in / str(fold_num) / f'{model_variant.name}_esoteric_{depth=}_{severity=}_pca.csv'
    save_path.parent.mkdir(exist_ok=True, parents=True)

    df.to_csv(save_path, index=False)


def for_prostate(depth=0, severity=5):
    path_to_p158_test = path_to_data / 'prostate' / 'nnUNet_raw_data' / 'Task158_Prostate158' / 'test_data'
    path_to_p158_labels = path_to_data / 'prostate' / 'nnUNet_raw_data' / 'Task158_Prostate158' / 'labelsTs'

    path_to_model_dirs = Path('data', 'prostate', 'models')
    store_pcas_in = path_to_model_dirs.parent / 'pca'

    model_variants = [iter_path for iter_path in path_to_model_dirs.iterdir() if iter_path.is_dir()]
    print([model_variant.name for model_variant in model_variants])

    for model_variant in model_variants:
        confirmation_message = input(f"[P158] Downloaded the {model_variant.name=} folds (0)? (y/n): ")
        if confirmation_message.lower() != 'y':
            continue

        for fold_num in range(1):

            try:
                # check if we already have the pca file for this model
                if (store_pcas_in / str(
                        fold_num) / f'{model_variant.name}_esoteric_{depth=}_{severity=}_pca.csv').exists():
                    print(f"Already have the pca file for {model_variant.name=}, skipping...")
                    continue

                for corruption in [
                    'Clean'
                ]:
                    corruption += f'_{severity}'
                    create_latent_space_dataset(
                        model_variant,
                        path_to_p158_test / corruption / 'imagesTs',
                        path_to_p158_labels,
                        corruption == "Clean_0",
                        fold_num
                    )

                compute_kv_gn_margin_and_pca(severity, depth, model_variant, store_pcas_in, fold_num)
            finally:
                delete_confirmation = 'y'  # input('Delete the esoteric folder and model weights? (y/n): ')
                if delete_confirmation.lower() == 'y':
                    shutil.rmtree(model_variant / f'fold_{fold_num}' / 'esoteric')
                    os.remove(model_variant / f'fold_{fold_num}' / 'model_final_checkpoint.model')
                    os.remove(model_variant / f'fold_{fold_num}' / 'model_final_checkpoint.model.pkl')
                    print('Deleted the esoteric folder and model weights')


def create_latent_space_data_acdc(_model_variant, _parent_dir, _scans_are_channel_split=False, fold_num=0):
    # the dataset needs to record the latent space output for each scan at each depth output from the model
    # dataset columns: model_variant, fold, scan_name, depth, path_to_latent_space_matrix, path_to_softmax_prediction_matrix, path_to_gt
    _plans_file = _model_variant / 'plans.pkl'
    with open(_plans_file, 'rb') as f:
        _plans = pickle.load(f)
    _num_classes = _plans['num_classes'] + 1

    if not _parent_dir.exists():
        # there is a tar file that needs to be extracted
        import tarfile
        _tar_file = _parent_dir.with_suffix('.tar.gz')
        with tarfile.open(_tar_file, 'r') as _tar:
            _tar.extractall(_parent_dir.parent)

    for _fold in [
        _fold_path for _fold_path in _model_variant.iterdir()
        if _fold_path.is_dir() and _fold_path.name.startswith(f'fold_{fold_num}')
    ]:
        # load the model
        _model_file = _fold / 'model_final_checkpoint.model'
        _model = process_plans_and_init(_plans, _model_file, deep_supervision=True, return_latent=True)
        _model = _model.cuda()
        _model.eval()

        # create inside the fold directory a new directory to store the new data being generated
        _output_dir = _fold / 'esoteric' / _parent_dir.name
        _output_dir.mkdir(exist_ok=True, parents=True)

        _path_to_all_labels = _parent_dir.rglob('*_gt.nii.gz')
        _path_to_all_labels = list(sorted(_path_to_all_labels))
        # path to scans just removes the _gt from the end of the file
        _path_to_all_scans = [
            _path_to_label.parent / _path_to_label.name.replace('_gt', '')
            for _path_to_label in _path_to_all_labels
        ]

        # only keep 10 scans for now
        indices_to_keep, _ = train_test_split(
            np.arange(len(_path_to_all_scans)),
            train_size=2,
            random_state=42
        )
        _path_to_all_scans = [_path_to_all_scans[_index] for _index in indices_to_keep]
        _path_to_all_labels = [_path_to_all_labels[_index] for _index in indices_to_keep]

        for (_scan_path, _label_path) in tqdm(
                zip(_path_to_all_scans, _path_to_all_labels),
                total=len(_path_to_all_scans),
                desc=_model_variant.name.replace('__nnUNetPlansv2.1', '')
                        .replace('nnUNetTrainerV2_', '')
                        .replace('_', ' + ')
                        .replace('nnUNetTrainerV2', 'Base')
                        .replace('nnUNetTrainerNoDAV2', 'No Aug')
        ):
            _scan_name = _scan_path.name.split('.')[0]
            if _scans_are_channel_split:
                _test_scan = handle_channel_split_files(_parent_dir, _scan_name)
            else:
                _test_scan = handle_single_file(_scan_path)
            _test_label = load_label(_label_path)
            _test_scan, _test_label, _original_shape = prepare_scan(_test_scan, _test_label, num_classes=_num_classes)
            # print(_test_scan.shape, _test_label.shape, _original_shape)

            with torch.enable_grad():
                _test_scan.requires_grad_(True)
                _model.requires_grad_(True)
                # print(_test_scan.shape)
                _latent_output, _segmentation_output = _model(_test_scan)

            if not isinstance(_latent_output, tuple):
                _latent_output = [_latent_output]
            if not isinstance(_segmentation_output, tuple):
                _segmentation_output = [_segmentation_output]

            for _depth, (_segmentation, _latent) in enumerate(zip(_latent_output, _segmentation_output)):
                if _depth != 0:
                    continue

                if _depth != 0:
                    _gt = torch.nn.functional.interpolate(_test_label, size=_segmentation.shape[2:], mode='nearest')
                else:
                    _gt = _test_label
                # save the latent space and the softmax prediction
                _latent_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_latent.npy'
                # _segmentation_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_seg.npy'
                _gt_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_gt.npy'
                _margin_output_path = _output_dir / f'{_scan_name}_depth_{_depth}_margin.npy'

                _margin, _probabilities, _grad_norm = find_margin(_gt, _segmentation, _latent)
                _latent = _latent.detach().cpu().numpy()
                _margin, _grad_norm = _margin.detach().cpu().numpy(), _grad_norm.detach().cpu().numpy()
                _gn_margin = _margin / (_grad_norm + 1e-6)

                _gt = torch.argmax(_gt, 1, keepdim=True).detach().cpu().numpy()
                np.save(_latent_output_path, _latent)
                # np.save(_segmentation_output_path, _probabilities.detach().cpu().numpy())
                np.save(_gt_output_path, _gt)
                np.save(_margin_output_path, _gn_margin)

            del _latent_output, _segmentation_output, _latent, _segmentation, _gt, _margin, _probabilities
            torch.cuda.empty_cache()

        del _model, _output_dir, _path_to_all_labels, _path_to_all_scans
        torch.cuda.empty_cache()

        gc.collect()

        # delete the extracted files
        shutil.rmtree(_parent_dir)


def for_acdc(depth=0, severity=5):
    path_to_acdc_test = path_to_data / 'ACDC' / 'test_data'

    path_to_model_dirs = Path('data', 'ACDC', 'models')
    store_pcas_in = path_to_model_dirs.parent / 'pca'

    model_variants = [iter_path for iter_path in path_to_model_dirs.iterdir() if iter_path.is_dir()]
    print([model_variant.name for model_variant in model_variants])

    for model_variant in model_variants:
        confirmation_message = input(f"[ACDC] Downloaded the {model_variant.name=} folds (0)? (y/n): ")
        if confirmation_message.lower() != 'y':
            continue

        for fold_num in range(1):

            try:
                if (store_pcas_in / str(
                        fold_num) / f'{model_variant.name}_esoteric_{depth=}_{severity=}_pca.csv').exists():
                    print(f"Already have the pca file for {model_variant.name=}, skipping...")
                    continue

                for corruption in [
                    'Clean'
                ]:
                    corruption += f'_{severity}'
                    create_latent_space_data_acdc(
                        model_variant,
                        path_to_acdc_test / corruption,
                        False,
                        fold_num
                    )

                compute_kv_gn_margin_and_pca(severity, depth, model_variant, store_pcas_in, fold_num)
            finally:
                delete_confirmation = 'y'  # input('Delete the esoteric folder and model weights? (y/n): ')
                if delete_confirmation.lower() == 'y':
                    shutil.rmtree(model_variant / f'fold_{fold_num}' / 'esoteric')
                    os.remove(model_variant / f'fold_{fold_num}' / 'model_final_checkpoint.model')
                    os.remove(model_variant / f'fold_{fold_num}' / 'model_final_checkpoint.model.pkl')
                    print('Deleted the esoteric folder and model weights')


if __name__ == '__main__':
    for_prostate(severity=3)
    for_acdc(severity=3)
