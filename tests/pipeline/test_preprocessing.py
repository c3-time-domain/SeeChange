import pytest
import pathlib
import uuid

import numpy as np
from astropy.io import fits

from models.base import FileOnDiskMixin, SmartSession
from models.image import Image

from tests.conftest import SKIP_WARNING_TESTS


def test_preprocessing(
        provenance_decam_prep, decam_exposure, test_config, preprocessor, decam_default_calibrators, archive
):
    # The decam_default_calibrators fixture is included so that
    # _get_default_calibrators won't be called as a side effect of calls
    # to Preprocessor.run().  (To avoid committing.)
    preprocessor.pars.test_parameter = uuid.uuid4().hex  # make a new Provenance for this temporary image
    ds = preprocessor.run( decam_exposure, 'S3' )
    assert preprocessor.has_recalculated

    # TODO: this might not work, because for some filters (g) the fringe correction doesn't happen
    # check that running the same processing on the same datastore is a no-op
    ds = preprocessor.run( ds )
    assert not preprocessor.has_recalculated

    # Check some Preprocesor internals
    assert preprocessor.pars.calibset == 'externally_supplied'
    assert preprocessor.pars.flattype == 'externally_supplied'
    assert preprocessor.pars.steps_required == [ 'overscan', 'linearity', 'flat', 'fringe' ]
    ds.exposure.filter[:1] == 'r'
    ds.section_id == 'S3'
    assert set( preprocessor.stepfiles.keys() ) == { 'flat', 'linearity' }

    # Make sure that the BSCALE and BZERO keywords got stripped
    #  from the raw image header.  (If not, when the file gets
    #  written out as floats, they'll be there and will screw
    #  things up.)
    assert 'BSCALE' not in ds.image.header
    assert 'BZERO' not in ds.image.header

    # Flatfielding should have improved the sky noise, though for DECam
    # it looks like this is a really small effect.  I've picked out a
    # section that's all sky.

    # 56 is how much got trimmed from this image
    rawsec = ds.image.raw_data[ 1780:1820, 830+56:870+56 ]
    flatsec = ds.image.data[ 1780:1820, 830:870 ]
    assert flatsec.std() < rawsec.std()

    # Make sure that some bad pixels got masked, but not too many
    assert np.all( ds.image._flags[ 2756:2773, 991:996 ] == 4 )
    assert np.all( ds.image._flags[ 0:4095, 57:60 ] == 1 )
    assert ( ds.image._flags != 0 ).sum() / ds.image.data.size < 0.03

    # Make sure that the weight is reasonable
    assert not np.any( ds.image._weight < 0 )
    assert ( ds.image.data[3959:3980, 653:662].std() ==
             pytest.approx( 1./np.sqrt(ds.image._weight[3959:3980, 653:662]), rel=0.05 ) )

    # Make sure that the expected files get written
    try:
        ds.save_and_commit()
        basepath = pathlib.Path( FileOnDiskMixin.local_path ) / ds.image.filepath
        archpath = archive.test_folder_path / ds.image.filepath

        for suffix, compimage in zip( [ '.image.fits', '.weight.fits', '.flags.fits' ],
                                      [ ds.image.data, ds.image._weight, ds.image._flags ] ):
            path = basepath.parent / f'{basepath.name}{suffix}'
            with fits.open( path, memmap=False ) as hdul:
                assert np.all( hdul[0].data == compimage )
            assert ( archpath.parent / f'{archpath.name}{suffix}' ).is_file()

        with SmartSession() as session:
            q = session.query( Image ).filter( Image.filepath == ds.image.filepath )
            assert q.count() == 1
            imobj = q.first()
            assert imobj.filepath_extensions == [ '.image.fits', '.flags.fits', '.weight.fits' ]
            assert imobj.md5sum is None
            assert len( imobj.md5sum_extensions ) == 3
            for i in range(3):
                assert imobj.md5sum_extensions[i] is not None

    finally:
        ds.delete_everything()

    # TODO : other checks that preprocessing did what it was supposed to do?
    # (Look at image header once we have HISTORY adding in there.)


def test_warnings_and_exceptions(decam_exposure, preprocessor, decam_default_calibrators, archive):
    if not SKIP_WARNING_TESTS:
        preprocessor.pars.inject_warnings = 1

        with pytest.warns(UserWarning) as record:
            preprocessor.run(decam_exposure, 'S3')
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'preprocessing'." in str(w.message)
                   for w in record)

    preprocessor.pars.inject_warnings = 0
    preprocessor.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = preprocessor.run(decam_exposure, 'S3')
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'preprocessing'." in str(excinfo.value)
