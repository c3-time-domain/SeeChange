# implement the ZOGY (Zackay, Ofek, & Gal-Yam 2016) image subtraction algorithm
# ref: https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract
# also got some useful ideas from the implementation here: https://github.com/pmvreeswijk/ZOGY/blob/main/zogy.py#L15721

import numpy as np
import scipy


def zogy_subtract(image_ref, image_new, psf_ref, psf_new, noise_ref, noise_new, flux_ref, flux_new, dx=None, dy=None):
    """Perform ZOGY image subtraction.

    This algorithm is based on Zackay et al. 2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract).
    It filters the new image with the PSF of the reference, and the reference image with the PSF of the new image
    (which makes it symmetric for switching them) and divides by a normalization factor (in Fourier space)
    that accounts for frequencies where there is less information.
    This division makes sure the result is stable on all frequencies (no ringing as in deconvolution methods).

    In addition to the subtracted image, it also calculates the subtraction PSF, and the "score" image,
    which is the subtraction image match-filtered with its PSF. This image is normalized such that
    any peak found in the score image has a value that is equivalent to the S/N of that detection.

    Besides the classical score image (which is optimal assuming background dominated noise), the function also
    returns the corrected score image, which takes into account the noise from bright sources.

    The function also applies the astrometric noise correction when calculating score_corr, but only if
    dx and dy are given (and if they are non-zero).
    These values are... TODO: finish this

    The last two outputs (alpha, alpha_std) represent a PSF-photometry measurement of the
    flux in the subtracted image. This is equivalent to placing a PSF at each point of the image
    and calculating the best fit normalization that fits the PSF to the pixel values.

    Another way to think about the flux normalizations is as the flux-based zero-point of the image
    (the transmission of atmosphere & optics, quantum efficiency, and exposure time).
    In many cases the reference flux normalization would be 1.0, and used relative to the flux_new.
    The new flux normalization will often be relative to the flux_ref (which is often set to 1.0)
    and will be used to scale the overall brightness normalization between images.
    E.g., if the reference has a total exposure time X and the new image has X/10 then we might set
    flux_ref=1 and flux_new=1/10 (assume the same system and identical sky).
    Measuring the relative flux normalization can be done, e.g., by cross matching stars in each image
    and calculating the average ratio of their fluxes (hence, zero-point).

    NOTES:
     * images must be registered (aligned and distortion corrected to sub-pixel level). They must
       have the same size (use NaNs to pad areas of the image that are not overlapping).
     * They should also be background subtracted and gain corrected, i.e., in units of electrons,
       such that the noise is Poissonian (the variance is equal to the value of each pixel).
     * The background noise is input to this function (including the noise RMS of the sky and read noise).
     * Users may input a scalar background RMS or a map of the background RMS with the same shape as the image.

    Parameters
    ----------
    image_ref : numpy.ndarray
        The reference image, background subtracted, in units of electrons (gain corrected).
    image_new : numpy.ndarray
        The new image, background subtracted, in units of electrons (gain corrected).
    psf_ref : numpy.ndarray
        The PSF of the reference image. Must be normalized to have unit sum. Will be zero-padded to size of images.
    psf_new : numpy.ndarray
        The PSF of the new image. Must be normalized to have unit sum. Will be zero-padded to size of images.
    noise_ref : float or numpy.ndarray
        The noise RMS of the background in the reference image (given as a map or a single average value).
        Does not include source noise!
    noise_new : float or numpy.ndarray
        The noise RMS of the background in the new image (given as a map or a single average value).
        Does not include source noise!
    flux_ref : float
        The flux normalization of the reference image.
    flux_new : float
        The flux normalization of the new image.
    dx : float

    dy : float


    Returns
    -------
    sub_image : numpy.ndarray
        The subtracted image.
    sub_psf: numpy.ndarray
        The PSF of the subtracted image.
    score: numpy.ndarray
        The score image
    score_corr: numpy.ndarray
        The score image corrected for source noise.
    alpha: numpy.ndarray
        The PSF-photometry measurement image
    alpha_std: numpy.ndarray
        The PSF-photometry noise image
    """

    # make copies to avoid modifying the input arrays
    N = np.fft.fftshift(np.copy(image_new))
    R = np.fft.fftshift(np.copy(image_ref))
    Pn = np.fft.fftshift(pad_to_shape(psf_new, N.shape))
    Pr = np.fft.fftshift(pad_to_shape(psf_ref, R.shape))

    # make sure all masked pixels in one image are masked in the other
    nan_mask = np.isnan(R) | np.isnan(N)
    if np.sum(~nan_mask) == 0:
        raise ValueError("All pixels are masked or no overlap between images.")

    # must replace all NaNs with zeros, otherwise the FFT will be all NaNs
    N[nan_mask] = 0
    R[nan_mask] = 0

    if isinstance(noise_ref, np.ndarray) and noise_ref.shape != R.shape:
        raise ValueError("noise_ref must have the same shape as the reference image.")

    if isinstance(noise_new, np.ndarray) and noise_new.shape != N.shape:
        raise ValueError("noise_new must have the same shape as the new image.")

    # get the representative noise values
    sigma_r = np.median(noise_ref[~nan_mask]) if isinstance(noise_ref, np.ndarray) else noise_ref
    sigma_n = np.median(noise_new[~nan_mask]) if isinstance(noise_new, np.ndarray) else noise_new

    # these are just shorthands for the flux normalization of each image
    F_r = flux_ref
    F_n = flux_new

    # Fourier transform the images and the PSFs
    # TODO: do we need to zero pad the PSFs to the same size as the images?
    R_f = np.fft.fft2(R)
    N_f = np.fft.fft2(N)
    P_r_f = np.fft.fft2(Pr)
    P_n_f = np.fft.fft2(Pn)
    P_r_f_abs2 = np.abs(P_r_f) ** 2
    P_n_f_abs2 = np.abs(P_n_f) ** 2

    # now start calculating the main results, equations 12-16 from the paper:
    F_D = F_r * F_n / np.sqrt(sigma_n ** 2 * F_r ** 2 + sigma_r ** 2 * F_n ** 2)  # eq 15
    denominator = sigma_n ** 2 * F_r ** 2 * P_r_f_abs2 + sigma_r ** 2 * F_n ** 2 * P_n_f_abs2  # eq 12's denominator
    # this can happen with certain rounding errors in the PSFs, but the numerator will also be zero, so it is ok:
    denominator[denominator == 0] = 1.0

    # the Fourier transform of the subtracted image and PSF
    D_f = (F_r * P_r_f * N_f - F_n * P_n_f * R_f) / np.sqrt(denominator)  # eq 13
    P_D_f = P_r_f * P_n_f * F_r * F_n / F_D / np.sqrt(denominator)  # eq 14

    # get the score image (match-filtered image)
    S_f = F_D * D_f * np.conj(P_D_f)  # eq 17 (which is equivalent to eq 12)

    # transform the subtracted image (and PSF, score) back to real space
    D = np.real(np.fft.ifft2(D_f))
    P_D = np.real(np.fft.ifft2(P_D_f))
    S = np.real(np.fft.ifft2(S_f))

    # additional corrections from the source noise terms:
    # get the variance maps, assuming the N/R images are background subtracted,
    # so that the noise maps give the background noise and the images contain only the source noise
    V_r = R + sigma_r ** 2
    V_n = N + sigma_n ** 2

    # this kernel is used to estimate the reference source noise
    k_r_f = F_r * F_n ** 2 * np.conj(P_r_f) * P_n_f_abs2 / denominator
    k_r = np.real(np.fft.ifft2(k_r_f))
    k_r2 = k_r ** 2
    k_r2_f = np.fft.fft2(k_r2)

    # this kernel is used to estimate the new source noise
    k_n_f = F_n * F_r ** 2 * np.conj(P_n_f) * P_r_f_abs2 / denominator
    k_n = np.real(np.fft.ifft2(k_n_f))
    k_n2 = k_n ** 2
    k_n2_f = np.fft.fft2(k_n2)

    # Fourier transform the variance (including source noise) images
    V_r_f = np.fft.fft2(V_r)
    V_n_f = np.fft.fft2(V_n)

    # these variance maps are convolved with the kernels
    V_S_r = np.real(np.fft.ifft2(V_r_f * k_r2_f))
    V_S_n = np.real(np.fft.ifft2(V_n_f * k_n2_f))

    if dx is not None and dx != 0 and dy is not None and dy != 0:
        # and calculate astrometric variance
        S_n = np.real(np.fft.ifft2(k_n_f * N_f))
        dS_n_dy = S_n - np.roll(S_n, 1, axis=0)
        dS_n_dx = S_n - np.roll(S_n, 1, axis=1)
        V_S_n_ast = dx ** 2 * dS_n_dx ** 2 + dy ** 2 * dS_n_dy ** 2

        S_r = np.real(np.fft.ifft2(k_r_f * R_f))
        dS_r_dy = S_r - np.roll(S_r, 1, axis=0)
        dS_r_dx = S_r - np.roll(S_r, 1, axis=1)
        V_S_r_ast = dx ** 2 * dS_r_dx ** 2 + dy ** 2 * dS_r_dy ** 2

        V_ast = V_S_r_ast + V_S_n_ast
    else:
        V_ast = 0

    V_S = V_S_r + V_S_n + V_ast

    zero_mask = V_S == 0  # get rid of zeros
    V_S_sqrt = np.sqrt(V_S, where=~zero_mask)
    V_S_sqrt[zero_mask] = 1
    S_corr = S / V_S_sqrt

    # PSF photometry part:
    # Eqs. 41-43 from paper
    F_S = F_n ** 2 * F_r ** 2 * np.sum((P_n_f_abs2 * P_r_f_abs2) / denominator)

    # divide by the number of pixels in the images (related to FFT normalization)
    F_S /= R.size

    alpha = S / F_S
    V_S_sqrt[zero_mask] = 0  # should we replace this with NaNs?
    alpha_std = V_S_sqrt / F_S

    # rename the outputs and fftshift back
    sub_image = np.fft.fftshift(D)
    sub_psf = np.fft.fftshift(P_D)
    score = np.fft.fftshift(S)
    score_corr = np.fft.fftshift(S_corr)
    alpha = np.fft.fftshift(alpha)
    alpha_std = np.fft.fftshift(alpha_std)

    return sub_image, sub_psf, score, score_corr, alpha, alpha_std


def pad_to_shape(arr, shape, value=0):
    """Pad a given array to have the same shape as given, adding equal (up to off-by-one) padding on all sides.

    Parameters
    ----------
    arr: numpy.ndarray
        The array to pad
    shape: tuple
        The desired shape of the array
    value: float
        The value to use for padding. Default is 0.

    Returns
    -------
    new_arr: numpy.ndarray
        The padded array (the array is copied).
    """

    if len(shape) != len(arr.shape):
        raise ValueError(
            f"The shape must have the same number of dimensions as the array. Got {len(shape)} and {len(arr.shape)}."
        )

    if any(s < a for s, a in zip(shape, arr.shape)):
        raise ValueError(
            f"The shape must be larger/equal to the array shape on all dimensions. Got {shape} and {arr.shape}."
        )

    new_arr = np.zeros(shape, dtype=arr.dtype)
    if value != 0:
        new_arr[:] = value

    # calculate the padding on each side
    pad = [(s - a) // 2 for s, a in zip(shape, arr.shape)]
    pad_after = [s - a - p for s, a, p in zip(shape, arr.shape, pad)]

    # pad the array
    new_arr[tuple(slice(p, -pa) for p, pa in zip(pad, pad_after))] = arr

    return new_arr


if __name__ == "__main__":
    arr = np.ones((4, 3))
    shape = (6, 7)
    new_arr = pad_to_shape(arr, shape)
    print(new_arr)
