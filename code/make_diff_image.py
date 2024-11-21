from typing import Union, List
from pathlib import Path

from distributed import Client, LocalCluster
import numpy as np
import zarr
import dask.array as da
import argparse
import os

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

def get_label_paths(label_dir: Union[str, Path]) -> List[Path]:
    """
    Retrieves a list of label Zarr file paths from the specified label directory.

    Parameters
    ----------
    label_dir : Union[str, Path]
        The directory containing label blocks.

    Returns
    -------
    List[Path]
        A list of paths to label Zarr files.
    """
    return [
        d / "Fill_Label_Mask.zarr"
        for d in Path(label_dir).iterdir()
        if d.is_dir() and "block_" in d.name
    ]

def process_volumes_and_label(raw_volume_path: Union[str, Path], filtered_volume_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Computes the absolute difference between raw and filtered volumes and saves it as a Zarr array.

    Parameters
    ----------
    raw_volume_path : Union[str, Path]
        Path to the raw volume Zarr file.
    filtered_volume_path : Union[str, Path]
        Path to the filtered volume Zarr file.
    output_path : Union[str, Path]
        Path where the difference volume will be saved as a Zarr file.

    Returns
    -------
    None
    """
    raw_volume = da.from_zarr(raw_volume_path, component="0").squeeze().astype(np.float32)
    filtered_volume = da.from_zarr(filtered_volume_path).squeeze().astype(np.float32)

    diff_volume = da.abs(filtered_volume - raw_volume).astype(np.uint16)

    output_store = zarr.DirectoryStore(output_path)
    da.to_zarr(diff_volume, output_store, overwrite=True)

    print(f'Saved difference volume to {output_path}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate difference volumes in labeled regions.')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Path to the directory containing raw Zarr volumes.')
    parser.add_argument('--filtered_dir', type=str, required=True,
                        help='Path to the directory containing filtered Zarr volumes.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for difference volumes.')
    args = parser.parse_args()

    raw_dir = args.raw_dir
    filtered_dir = args.filtered_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_paths = get_raw_zarr_paths(raw_dir)
    filtered_paths = get_filtered_zarr_paths(filtered_dir)

    if not raw_paths or not filtered_paths:
        print('No Zarr files found in one of the input directories.')
        return

    raw_paths.sort()
    filtered_paths.sort()

    client = Client(LocalCluster(processes=True))

    for raw_path, filtered_path in zip(raw_paths, filtered_paths):
        print(f'Processing raw volume: {raw_path}')
        print(f'Processing filtered volume: {filtered_path}')

        output_file = os.path.join(output_dir, f'{raw_path.stem}_diff_volume.zarr')

        process_volumes_and_label(raw_path, filtered_path, output_file)
        

if __name__ == '__main__':
    main()
