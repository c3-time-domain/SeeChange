import warnings

import numpy as np
from scipy.signal import convolve
from scipy.ndimage import binary_dilation
from skimage.restoration import inpaint_biharmonic

from pipeline.parameters import Parameters
from improc.tools import sigma_clipping


class ParsInpainter(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.multi_image_method = self.add_par(
            'multi_image_method',
            'median',
            str,
            'Method to interpolate between different images in the cube. '
            'Can be "mean" or "median" (default) or "none". ',
            critical=True
        )

        self.feather_width = self.add_par(
            'feather_width',
            2,
            float,
            'Width of the feathering region (in pixels) around each bad pixel replacement. '
            'This is used only when inpainting between images in the data cube. '
            'Default is 0 (no feathering).',
            critical=True
        )

        self.rescale_method = self.add_par(
            'rescale_method',
            'median',
            str,
            'Method to rescale images before interpolating between them. '
            'Can be "median" (default) or "sigma_clipping" or "none". ',
            critical=True
        )

        self.single_image_method = self.add_par(
            'single_image_method',
            'biharmonic',
            str,
            'Method to interpolate inside a single image. Use "biharmonic" (default). No other methods defined yet. ',
            critical=True
        )

        self.ignore_flags = self.add_par(
            'ignore_flags',
            0,
            int,
            'Pixels with these flags are not interpolated over. '
            'Default is 0 (any flags are considered missing pixels).',
            critical=True
        )

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'inpainting'


class Inpainter:
    def __init__(self, **kwargs):
        self.pars = ParsInpainter(**kwargs)

        self.images = None
        self.flags = None
        self.weights = None

        self.images_cube = None
        self.flags_cube = None
        self.weights_cube = None

        self.images_nan = None
        self.flags_nan = None
        self.weights_nan = None

        self.images_cube_done = None
        self.images_output = None

    def bad_pixel_set_nan(self):
        """Replace any pixels deemed bad with NaNs.

        If the input images are a 2D single image,
        will add a new axis to make it a 3D data cube with one image.

        This will move the inputs self.images, self.flags and self.weights
        into the _cube versions, where they are guaranteed to be 3D data cubes.

        Also, will make self.images_nan, where the bad pixels are replaced with NaNs.
        """
        if len(self.images.shape) == 2:
            self.images_cube = np.expand_dims(self.images, axis=0)
            self.flags_cube = np.expand_dims(self.flags, axis=0)
            self.weights_cube = np.expand_dims(self.weights, axis=0)
        else:
            self.images_cube = self.images
            self.flags_cube = self.flags
            self.weights_cube = self.weights

        # replace bad pixels (those with flags not marked "ignored") with NaNs
        self.images_nan = self.images_cube.copy()
        self.images_nan[self.flags_cube & (~self.pars.ignore_flags) != 0] = np.nan

    def inpaint_cube(self):
        """Interpolate on the images in the data cube by using the average of all images.

        Some bad pixels could remain in case all images had a bad pixel at the same location.
        """
        if self.images is None:
            raise ValueError("No images to interpolate over.")
        if self.flags is None:
            raise ValueError("flags are missing! .")
        if self.weights is None:
            raise ValueError("weights are missing! .")

        self.images_cube_done = self.images_nan.copy()

        if self.pars.multi_image_method == 'none':
            return  # short circuit the rest of this function!

        if self.pars.rescale_method == 'none':
            scaling = np.ones((self.images_nan.shape[0], 1, 1))
        elif self.pars.rescale_method == 'median':
            scaling = np.nanmedian(self.images_nan, axis=(1, 2), keepdims=True)
        elif self.pars.rescale_method == 'sigma_clipping':
            scaling = np.zeros((self.images_nan.shape[0], 1, 1))
            for i in range(self.images_nan.shape[0]):
                mu, sig = sigma_clipping(self.images_nan[i])
                scaling[i] = mu
        else:
            raise ValueError(f"Unknown rescaling method: {self.pars.rescale_method}")
        with warnings.catch_warnings():
            # we definitely expect all images to have NaNs on the same spot for at least some pixels (e.g., saturation)
            warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            if self.pars.multi_image_method == 'mean':
                replacements = np.nanmean(self.images_nan / scaling, axis=0)
            elif self.pars.multi_image_method == 'median':
                replacements = np.nanmedian(self.images_nan / scaling, axis=0)
            else:
                raise ValueError(f"Unknown interpolation method: {self.pars.multi_image_method}")

        for i in range(self.images_cube_done.shape[0]):
            positions = np.isnan(self.images_nan[i])
            positions[np.isnan(replacements)] = False  # spots where all images had bad pixels!
            if self.pars.feather_width == 0:
                self.images_cube_done[i][positions] = replacements[positions] * scaling[i][0][0]
            else:
                w = self.pars.feather_width
                square = np.zeros((2 * w + 3, 2 * w + 3))
                square[1:-1, 1:-1] = 1.0
                k = convolve(square, square, mode='same')  # linear tapering kernel

                struc = np.zeros((3, 3), dtype=bool)
                struc[1, :] = True
                struc[:, 1] = True
                dilat = binary_dilation(np.pad(positions, 2 * w), iterations=w * 2, structure=struc)
                pos_tapered = convolve(dilat, k, mode='same')[2 * w:-2 * w, 2 * w:-2 * w]
                if np.max(pos_tapered) > 0:
                    pos_tapered /= np.max(pos_tapered)
                pos_flipped = 1 - pos_tapered
                self.images_cube_done[i][positions] = 0  # remove NaNs before adding feathered replacement
                self.images_cube_done[i] = (
                    self.images_cube_done[i] * pos_flipped + replacements * scaling[i][0][0] * pos_tapered
                )

    def inpaint_single(self):
        """Go over each image in the datacube and interpolate over any remaining bad pixels.

        Will use each image in images_cube_done, interpolate internally, and put the results
        in the images_output array.
        """
        if self.images_cube_done is None:
            raise ValueError("No images to interpolate over.")

        self.images_output = self.images_cube_done.copy()

        for i in range(self.images_cube_done.shape[0]):
            if self.pars.single_image_method == 'biharmonic':
                self.images_output[i] = inpaint_biharmonic(
                    self.images_cube_done[i], np.isnan(self.images_cube_done[i])
                )
            else:
                raise ValueError(
                    f'Unknown interpolation method: {self.pars.single_image_method}. Use "biharmonic". '
                )

    def run(self, images, flags, weights):
        """
        Run inpainting to fix any missing/bad pixels in the input images.
        The returned image will be the same shape as the input images.
        Bad pixels are replaced using interpolation between images in the
        data cube (if given a 3D data cube) and then any remaining missing/bad
        pixels are interpolated inside each image separately.
        The flags and weights are not modified by this function,
        and only a copy of the images array (with bad pixels removed) is returned.

        Note that the quality of the inpainted images depends on having
        the images scaled to the same zero-point and have them background subtracted.
        If they have very different seeing, the inpainting between images also may be poor.
        In most cases, the inpainted pixels should be flagged so the values they store are
        not used directly to measure anything.
        The inpainted image should be smooth enough to avoid making large artefacts
        in coadded/subtracted images, particularly when using FFT methods (e.g., ZOGY).

        Parameters
        ----------
        images: np.ndarray (2D or 3D)
            The image or image cube to interpolate over.
            If given as a 2D array, will skip right to inpaint_single().
            If given as a 3D data cube, will try to interpolate between images
            in the cube first, and only inpaint pixels that were missing
            in all images.
        flags: np.ndarray (2D or 3D)
            The flags for each pixel in images.
            Has to be the same shape as the images input.
            Any pixels with flags & (~ignore_flags) != 0 will be inpainted.
            Default is 0, such that any kind of non-zero flag value is considered a missing
            pixel and will be inpainted.
        weights: np.ndarray (2D or 3D)
            The weights for each pixel in images.
            Has to be the same shape as the images input.
            Currently, we are not using the weights for any calculations,
            but they could come into play for, e.g., calculating weighted means.

        Returns
        -------
        output: np.ndarray
            The inpainted image or image cube (same size as images).
        """
        if not isinstance(images, np.ndarray):
            raise TypeError(f"images must be an np.ndarray, not {type(images)}")
        if not isinstance(flags, np.ndarray):
            raise TypeError(f"flags must be an np.ndarray, not {type(flags)}")
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"weights must be an np.ndarray, not {type(weights)}")

        if images.shape != flags.shape:
            raise ValueError(f"Images and flags have different shapes: {images.shape} vs {flags.shape}")
        if images.shape != weights.shape:
            raise ValueError(f"Images and weights have different shapes: {images.shape} vs {weights.shape}")

        self.images = images
        self.flags = flags
        self.weights = weights

        # this will also expand_dims for a 2D array into a 3D data cube
        self.bad_pixel_set_nan()

        if len(self.images.shape) == 3:
            self.inpaint_cube()
        elif len(self.images.shape) == 2:  # skip interpolation between images
            self.images_cube_done = self.images_nan.copy()
        else:
            raise ValueError(f"Images have the wrong shape: {self.images.shape}. Use 2D or 3D arrays! ")

        # process the data in image_cube_done etc. going image by image
        self.inpaint_single()

        if np.any(np.isnan(self.images_output)):
            raise RuntimeError('NaNs found in the output image. Something went wrong!')

        return np.reshape(self.images_output.copy(), self.images.shape)

