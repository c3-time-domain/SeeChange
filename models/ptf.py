import numpy as np

from models.instrument import Instrument, InstrumentOrientation, SensorSection


class PTF(Instrument):

    def __init__(self, **kwargs):
        self.name = 'PTF'
        self.telescope = 'P48'
        self.aperture = 1.22
        self.focal_ratio = 2.7
        self.square_degree_fov = 7.26
        self.pixel_scale = 1.01
        self.read_time = 30.0
        self.orientation_fixed = True
        self.orientation = InstrumentOrientation.NleftEup

        self.read_noise = 4.5
        self.dark_current = 0.1
        self.gain = 1.5
        self.saturation_limit = 20000
        self.non_linearity_limit = 20000
        self.allowed_filters = ["R", "V", "I"]

        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        Instrument.__init__(self, **kwargs)

        # we are using preprocessed data as the exposures, so everything is already done
        self.preprocessing_steps_available = []
        self.preprocessing_steps_done = ['overscan', 'linearity', 'flat', 'fringe']

    @classmethod
    def get_section_ids(cls):

        """
        Get a list of SensorSection identifiers for this instrument.
        Includes all 12 CCDs.
        """
        return [str(sid) for sid in range(0, 12)]

    @classmethod
    def check_section_id(cls, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        In this case, it must be an integer in the range [0, 11].
        """
        try:
            section_id = int(section_id)
        except ValueError:
            raise ValueError(
                f"The section_id must be an integer or a string convertible to an integer. Got {section_id}. "
            )

        if not 0 <= section_id <= 11:
            raise ValueError(f"The section_id must be in the range [0, 11]. Got {section_id}. ")

    @classmethod
    def get_filename_regex(cls):
        return [r'PTF.*\.fits']

    @classmethod
    def _get_header_keyword_translations(cls):
        output = super()._get_header_keyword_translations()
        output['ra'] = ['TELRA']
        output['dec'] = ['TELDEC']
        output['project'] = ['OBJECT']
        output['target'] = ['PTFFIELD']

        return output

    @classmethod
    def _get_header_values_converters(cls):
        return {'instrument': lambda x: 'PTF'}

    @classmethod
    def _get_fits_hdu_index_from_section_id(cls, section_id):
        cls.check_section_id(section_id)
        return 0  # TODO: improve this if we ever get multiple HDUs per file

    def _make_new_section(self, section_id):
        """
        Make a single section for the PTF instrument.
        The section_id must be a valid section identifier ([0, 11]).

        Returns
        -------
        section: SensorSection
            A new section for this instrument.
        """
        (dx, dy) = self.get_section_offsets(section_id)  # this also runs "check_section_id" internally
        defective = section_id == 1  # TODO: check which chip is defective!
        return SensorSection(section_id, self.name, size_x=2048, size_y=4096,
                             offset_x=dx, offset_y=dy, defective=defective)

    def _get_default_calibrator(self, mjd, section, calibtype='dark', filter=None, session=None):
        pass

    @classmethod
    def gaia_dr3_to_instrument_mag(cls, filter, catdata):
        """Transform Gaia DR3 magnitudes to instrument magnitudes.

        Uses a polynomial transformation from Gaia MAG_G to instrument magnitude.

        The array trns allows a conversion from Gaia MAG_G to
        the magnitude through the desired filter using:

          MAG_filter = Gaia_MAG_G - sum( trns[i] * ( Gaia_MAG _BP - Gaia_MAG_RP ) ** i )

        (with i running from 0 to len(trns)-1).

        Parameters
        ----------
        filter: str
            The (short) filter name of the magnitudes we want.
        catdata: dict or pandas.DataFrame or numpy.recarray or astropy.Table
            A data structure that holds the relevant data,
            that can be indexed on the following keys:
            MAG_G, MAGERR_G, MAG_BP, MAGERR_BP, MAG_RP, MAGERR_RP
            If a single magnitude is required, can pass a dict.
            If an array of magnitudes is required, can be any
            data structure that when indexed on those keys
            returns a 1D numpy array (e.g., a pandas DataFrame,
            or a named structured numpy array, or even a dict
            with ndarray values).

        Returns
        -------
        trans_mag: float or numpy array
            The catalog magnitude(s) transformed to instrument magnitude(s).
        trans_magerr: float or numpy array
            The catalog magnitude error(s) transformed to instrument magnitude error(s).
        """
        if not isinstance(filter, str):
            raise ValueError(f"The filter must be a string. Got {type(filter)}. ")

        # Emily Ramey came up with these by fitting polynomials to Gaia
        # magnitudes and DECaLS magnitudes
        # transformations = {
        #     'g': np.array([0.07926061, -0.18958323, -0.50588824, 0.11476034]),
        #     'r': np.array([-0.28526417, 0.65444024, -0.25415955, -0.00204337]),
        #     'i': np.array([-0.2491122, 0.51709843, 0.02919352, -0.02097517]),
        #     'z': np.array([-0.38939061, 0.70406435, 0.04190059, -0.01617815])
        # }
        # TODO: must find transformations!
        transformations = {
            'R': np.array([0]),
            'V': np.array([0]),
            'I': np.array([0]),
        }

        if filter not in transformations:
            raise ValueError(f"Unknown short DECam filter name {filter}")

        # instrumental mag is sum(trns[i] * (GaiaBP - GaiaRP) ** i)
        trns = transformations[filter]
        fitorder = len(trns) - 1

        colors = catdata['MAG_BP'] - catdata['MAG_RP']
        colorerrs = np.sqrt(catdata['MAGERR_BP'] ** 2 + catdata['MAGERR_RP'] ** 2)
        colton = colors[:, np.newaxis] ** np.arange(0, fitorder + 1, 1)
        coltonminus1 = np.zeros(colton.shape)
        coltonminus1[:, 1:] = colors[:, np.newaxis] ** np.arange(0, fitorder, 1)
        coltonerr = np.zeros(colton.shape)
        coltonerr[:, 1:] = np.arange(1, fitorder + 1, 1) * coltonminus1[:, 1:] * colorerrs.value[:, np.newaxis]

        trans_mag = catdata['MAG_G'] - (trns[np.newaxis, :] * colton).sum(axis=1)
        trans_magerr = np.sqrt(catdata['MAGERR_G'] ** 2 + (trns[np.newaxis, :] * coltonerr).sum(axis=1) ** 2)

        return trans_mag, trans_magerr
