import argparse
import logging
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from dask_image.ndfilters import median_filter
from distributed import Client, LocalCluster
from numcodecs import blosc
from skimage.exposure import rescale_intensity
from skimage.morphology import cube
from skimage.restoration import (denoise_nl_means, denoise_tv_chambolle,
                                 estimate_sigma)
from skimage.util import apply_parallel


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


def filter_arr(arr: Any, func: Callable, kwargs: dict = {}) -> Any:
    """
    Applies a given function to an array with specified keyword arguments.

    Parameters
    ----------
    arr : Any
        The input array to be processed.
    func : Callable
        The function to apply to the array.
    kwargs : dict
        Additional dictionary of keyword arguments to pass to the function.

    Returns
    -------
    Any
        The result of applying the function to the array.
    """
    result = func(arr, **kwargs)

    if isinstance(result, da.Array):
        return result.compute()
    else:
        return result


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


def get_nl_means_params(temp_dir: Path) -> List[Dict[str, Any]]:
    all_params = []
    all_params.append(
        {
            "func": nl_means_wrapper,
            "args": {
                "temp_dir": temp_dir,
                "patch_size": 11,
                "patch_distance": 15,
                "h_factor": 0.8,
                "fast_mode": True,
                "preserve_range": True,
                "channel_axis": None,
            },
        }
    )
    return all_params


def nl_means_wrapper(arr, temp_dir, **kwargs):
    def process_slice(arr, **kwargs):
        a = arr[0].astype(np.float32)
        sigma = estimate_sigma(a, channel_axis=None)
        h_factor = kwargs.pop("h_factor")
        result = denoise_nl_means(a, h=sigma * h_factor, sigma=sigma, **kwargs)
        return result[np.newaxis, ...]

    # Rechunk the array to slices ahead of time
    arr = arr.rechunk((1, arr.shape[1], arr.shape[2]))
    temp_store = da.to_zarr(
        arr,
        temp_dir / "temp_rechunked.zarr",
        compute=True,
        return_stored=True,
        overwrite=True,
    )

    return da.map_blocks(process_slice, temp_store, **kwargs)


def get_tv_chambolle_params():
    weights = [3, 5, 10, 13, 15]
    all_params = []
    for w in weights:
        all_params.append(
            {
                "func": tv_chambolle_wrapper,
                "args": {"weight": w, "max_num_iter": 200, "channel_axis": None},
            }
        )
    return all_params


def tv_chambolle_wrapper(arr, **kwargs):
    minimum, maximum = da.compute(arr.min(), arr.max())
    arr = arr.astype(np.float32).compute()
    arr = denoise_tv_chambolle(arr, **kwargs)
    arr = rescale_intensity(arr, in_range="image", out_range=(minimum, maximum)).astype(
        np.uint16
    )
    return arr


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

    for i, (zarr_path, filter_params, codec_params) in enumerate(
        parameter_combinations
    ):
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
            overwrite=True,
        )

        # Filter the array
        t0 = time.time()
        out_zarr_denoised[:] = filter_arr(
            arr,
            filter_params["func"],
            kwargs=filter_params["args"],
        )
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
        "--filter",
        type=str,
        required=True,
        choices=["median", "nl_means", "tv_chambolle"],
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="/tmp",
        help="Temporary directory for intermediate files.",
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

    dask.config.set(
        {
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": False,
            "distributed.worker.memory.terminate": 0.98,
        }
    )

    client = Client(LocalCluster(processes=False))
    logging.info(f"Dask client started: {client}")

    image_dir: Union[str, Path] = args.image_dir
    zarr_paths: List[Path] = get_zarr_paths(image_dir)

    temp_dir = Path(args.temp_dir)

    if args.filter == "median":
        filter_params: List[Dict[str, Any]] = get_median_filter_params()
    elif args.filter == "nl_means":
        filter_params: List[Dict[str, Any]] = get_nl_means_params(temp_dir)
    elif args.filter == "tv_chambolle":
        filter_params: List[Dict[str, Any]] = get_tv_chambolle_params()
    else:
        raise ValueError(f"Invalid filter: {args.filter}")

    blosc_params: List[Dict[str, Any]] = get_blosc_params()

    # Create parameter combinations
    parameter_combinations: Iterable[
        Tuple[Path, Dict[str, Any], Dict[str, Any]]
    ] = product(zarr_paths, filter_params, blosc_params)
    # logging.debug(list(parameter_combinations))

    denoise_compress_blocks(parameter_combinations, output_dir=args.output_dir)
