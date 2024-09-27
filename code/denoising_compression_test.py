import sys
import time
from itertools import product
from pathlib import Path

import argparse
import logging
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from dask_image.ndfilters import median_filter
from distributed import Client, LocalCluster
from numcodecs import blosc
from skimage.morphology import cube
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union


def make_serializable(obj: Any) -> Any:
    """
    Recursively converts non-serializable objects within a data structure into serializable forms.

    Parameters
    ----------
    obj : Any
        The object to be serialized.

    Returns
    -------
    Any
        The serialized object.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.shape
    elif callable(obj):
        return obj.__name__
    else:
        return obj


def filter_arr(arr: Any, func: Callable, **kwargs: Any) -> Any:
    """
    Applies a given function to an array with specified keyword arguments.

    Parameters
    ----------
    arr : Any
        The input array to be processed.
    func : Callable
        The function to apply to the array.
    **kwargs : Any
        Additional keyword arguments to pass to the function.

    Returns
    -------
    Any
        The result of applying the function to the array.
    """
    return func(arr, **kwargs)


def get_median_filter_params() -> List[Dict[str, Any]]:
    """
    Generates a list of parameter dictionaries for median filtering.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the function and its arguments for median filtering.
    """
    footprint_sizes = [3, 5]
    all_params = []
    for fp in footprint_sizes:
        all_params.append({"args": {"footprint": cube(fp)}, "func": median_filter})
    return all_params


def get_blosc_params() -> List[Dict[str, Any]]:
    """
    Generates a list of parameter dictionaries for Blosc compression.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing parameters for Blosc compression.
    """
    clevels = [1, 5, 9]
    all_params = []
    for cl in clevels:
        all_params.append(
            {"cname": "zstd", "clevel": cl, "shuffle": blosc.Blosc.SHUFFLE}
        )
    return all_params


def get_zarr_paths(image_dir: Union[str, Path]) -> List[Path]:
    """
    Retrieves a list of Zarr file paths from the specified image directory.

    Parameters
    ----------
    image_dir : Union[str, Path]
        The directory containing image blocks.

    Returns
    -------
    List[Path]
        A list of paths to Zarr files.
    """
    return [
        d / f"{d.name}.zarr"
        for d in Path(image_dir).iterdir()
        if d.is_dir() and "block_" in d.name
    ]


def denoise_compress_blocks(
    parameter_combinations: Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any]]],
    output_dir: Union[str, Path],
) -> None:
    """
    Denoises and compresses blocks of image data and saves the results.

    Parameters
    ----------
    parameter_combinations : Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any]]]
        An iterable of tuples containing Zarr paths, filter parameters, and codec parameters.
    output_dir : Union[str, Path]
        The directory where the output data.csv will be saved.

    Returns
    -------
    None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = []

    for i, (zarr_path, filter_params, codec_params) in enumerate(parameter_combinations):
        logging.info(f"Processing {zarr_path.name}")
        logging.info(f"Filter params: {filter_params}")
        logging.info(f"Codec params: {codec_params}")

        out_zarr_dir = output_dir / zarr_path.stem
        out_zarr_dir.mkdir(parents=True, exist_ok=True)

        arr = da.from_zarr(zarr_path, component="0").squeeze()

        out_zarr = zarr.array(
            data=arr.compute(),
            chunks=arr.chunksize,
            compressor=zarr.Blosc(**codec_params),
        )

        unfiltered_storage_ratio = out_zarr.nbytes / out_zarr.nbytes_stored
        logging.info(f"Unfiltered Storage Ratio: {unfiltered_storage_ratio}")

        out_zarr_denoised = zarr.create(
            shape=arr.shape,
            chunks=arr.chunksize,
            dtype=arr.dtype,
            compressor=zarr.Blosc(**codec_params),
            store=zarr.DirectoryStore(out_zarr_dir / f"{zarr_path.stem}_{i}.zarr"),
            overwrite=True
        )

        # Filter the array
        t0 = time.time()
        out_zarr_denoised[:] = filter_arr(
            arr, filter_params["func"], **filter_params["args"]
        ).compute()
        t1 = time.time()
        process_time = t1 - t0
        logging.info(f"Process Time: {process_time}")

        # Check the compression ratio
        filtered_storage_ratio = (
            out_zarr_denoised.nbytes / out_zarr_denoised.nbytes_stored
        )
        logging.info(f"Filtered Storage Ratio: {filtered_storage_ratio}")

        data = {
            "filter_params": make_serializable(filter_params),
            "codec_params": make_serializable(codec_params),
            "unfiltered_storage_ratio": unfiltered_storage_ratio,
            "filtered_storage_ratio": filtered_storage_ratio,
            "process_time": process_time,
            "block": zarr_path.stem,
            "in_bytes": out_zarr_denoised.nbytes,
            "out_bytes": out_zarr_denoised.nbytes_stored,
        }

        all_data.append(data)

    # Convert the list of dictionaries to a DataFrame and save as CSV
    df = pd.DataFrame(all_data)
    output_csv_path = output_dir / "data.csv"
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise and compress blocks.")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing image blocks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output data.csv.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    client = Client(LocalCluster(processes=False))
    logging.info(f"Dask client started: {client}")

    image_dir: Union[str, Path] = args.image_dir
    zarr_paths: List[Path] = get_zarr_paths(image_dir)

    median_params: List[Dict[str, Any]] = get_median_filter_params()

    blosc_params: List[Dict[str, Any]] = get_blosc_params()

    # Create parameter combinations
    parameter_combinations: Iterable[
        Tuple[Path, Dict[str, Any], Dict[str, Any]]
    ] = product(zarr_paths, median_params, blosc_params)
    # logging.debug(list(parameter_combinations))

    denoise_compress_blocks(parameter_combinations, output_dir=args.output_dir)
