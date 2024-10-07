import time
import logging
import numpy as np
import cupy as cp
from skimage.restoration import denoise_tv_chambolle


def denoise_tv_chambolle_gpu(image, weight=0.1, eps=2.0e-4, max_num_iter=200):
    """Perform total-variation denoising on n-dimensional images using GPU.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.
    """

    ndim = image.ndim
    # Transfer image to GPU
    image = cp.asarray(image)
    p = cp.zeros((image.ndim,) + image.shape, dtype=image.dtype)
    g = cp.zeros_like(p)
    d = cp.zeros_like(image)
    i = 0
    while i < max_num_iter:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None)] * ndim
            slices_p = [slice(None)] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax + 1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax + 1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d**2).sum()

        # g stores the gradients of out along each axis
        slices_g = [slice(None)] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax + 1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = cp.diff(out, axis=ax)
            slices_g[ax + 1] = slice(None)

        norm = cp.sqrt((g**2).sum(axis=0))[cp.newaxis, ...]
        E += weight * norm.sum()
        tau = 1.0 / (2.0 * ndim)
        norm *= tau / weight
        norm += 1.0
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if cp.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    # Transfer the result back to CPU
    return cp.asnumpy(out)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)

    image_shape = (256, 256, 256)
    logger.info(f"Generating random image of shape {image_shape}.")
    image = np.random.rand(*image_shape).astype(np.float32)

    # Perform denoising on GPU
    logger.info("Starting GPU denoising.")
    t0 = time.time()
    denoised_image_gpu = denoise_tv_chambolle_gpu(image, weight=0.1, eps=2.0e-4, max_num_iter=200)
    t1 = time.time()
    gpu_time = t1 - t0
    logger.info(f"GPU denoising completed in {gpu_time:.2f} seconds.")

    # Perform denoising on CPU using scikit-image
    logger.info("Starting CPU denoising using scikit-image.")
    t0 = time.time()
    denoised_image_cpu = denoise_tv_chambolle(image, weight=0.1, eps=2.0e-4, max_num_iter=200)
    t1 = time.time()
    cpu_time = t1 - t0
    logger.info(f"CPU denoising completed in {cpu_time:.2f} seconds.")

    # Compare results
    logger.info("Comparing GPU and CPU denoised images.")
    try:
        np.testing.assert_allclose(
            denoised_image_gpu,
            denoised_image_cpu,
            rtol=1e-6,
            atol=1e-6
        )
        logger.info("SUCCESS: GPU and CPU denoised images are within the acceptable error precision.")
    except AssertionError as e:
        logger.error("ERROR: GPU and CPU denoised images differ beyond the acceptable error precision.")
        logger.error(e)
        raise


if __name__ == "__main__":
    main()
