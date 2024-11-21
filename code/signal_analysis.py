from typing import Union, List
from pathlib import Path

from distributed import Client, LocalCluster
import numpy as np
import zarr
import dask.array as da
import argparse
import os
import glob
import json


def get_raw_zarr_paths(image_dir: Union[str, Path]) -> List[Path]:
    """
    Retrieves a list of raw Zarr file paths from the specified image directory.

    Parameters
    ----------
    image_dir : Union[str, Path]
        The directory containing image blocks.

    Returns
    -------
    List[Path]
        A list of paths to raw Zarr files.
    """
    return [
        d / f"{d.name}.zarr"
        for d in Path(image_dir).iterdir()
        if d.is_dir() and "block_" in d.name
    ]


def get_filtered_zarr_paths(image_dir: Union[str, Path]) -> List[Path]:
    """
    Retrieves a list of filtered Zarr file paths from the specified image directory.

    Parameters
    ----------
    image_dir : Union[str, Path]
        The directory containing image blocks.

    Returns
    -------
    List[Path]
        A list of paths to filtered Zarr files.
    """
    return [
        d / f"{d.name}_{i}.zarr"
        for i, d in enumerate(Path(image_dir).iterdir())
        if d.is_dir() and "block_" in d.name
    ]


def get_label_paths(image_dir: Union[str, Path]) -> List[Path]:
    """
    Retrieves a list of label Zarr file paths from the specified image directory.

    Parameters
    ----------
    image_dir : Union[str, Path]
        The directory containing image blocks.

    Returns
    -------
    List[Path]
        A list of paths to label Zarr files.
    """
    return [
        d / "Fill_Label_Mask.zarr"
        for d in Path(image_dir).iterdir()
        if d.is_dir() and "block_" in d.name
    ]


def process_volume_and_label(
    volume_path: Union[str, Path],
    label_image_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Computes intensity statistics within labeled regions of a volume and saves the results as a JSON file.

    Parameters
    ----------
    volume_path : Union[str, Path]
        Path to the volume Zarr file.
    label_image_path : Union[str, Path]
        Path to the label image Zarr file.
    output_path : Union[str, Path]
        Path where the intensity statistics JSON file will be saved.

    Returns
    -------
    None
    """
    volume = da.from_zarr(volume_path, chunks=(256, 256, 256)).squeeze().compute()
    label_image = da.from_zarr(
        label_image_path, component="0", chunks=(1, 1, 256, 256, 256)
    ).squeeze().compute()

    max_label = int(label_image.max())

    labels = np.unique(label_image)
    print(len(labels))

    # Exclude the background label (assumed to be 0)
    labels = labels[labels != 0]

    intensity_stats = {}

    for label in labels:
        # Create a mask for the current label
        label_mask = label_image == label

        # Extract the intensities from the volume where the mask is True
        intensities = volume[label_mask]
        print(intensities)

        mean_intensity = intensities.mean()
        median_intensity = np.percentile(intensities, 50)
        std_intensity = np.std(intensities)

        intensity_stats[int(label)] = {
            'mean': float(mean_intensity),
            'median': float(median_intensity),
            'std': float(std_intensity)
        }
        print(intensity_stats[int(label)])

    with open(output_path, 'w') as f:
        json.dump(intensity_stats, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description='Process a directory of Zarr images.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory containing Zarr images.')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Path to the directory containing label Zarr images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for results.')
    args = parser.parse_args()

    input_dir = args.input_dir
    label_dir = args.label_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    im_paths = get_raw_zarr_paths(input_dir)
    label_paths = get_label_paths(label_dir)

    if not im_paths:
        print(f'No zarr files found in {input_dir}.')
        return

    client = Client(LocalCluster(processes=True))

    for raw_path, label_path in zip(im_paths, label_paths):
        print(f'Processing volume: {raw_path}')
        print(f'Corresponding label image: {label_path}')

        output_file = os.path.join(output_dir, f'{raw_path.stem}_intensity_stats.json')

        process_volume_and_label(raw_path, label_path, output_file)

        print(f'Saved intensity statistics to {output_file}\n')


if __name__ == '__main__':
    main()
