import pytest
import uuid
import numpy as np

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements
from pipeline.data_store import DataStore

def test_measurements_attributes(measurer, ptf_datastore, test_config):
    ds = ptf_datastore

    aper_radii = test_config.value('extraction.sources.apertures')
    ds.measurements = None
    ds = measurer.run( ds )
    # check that the measurer actually loaded the measurements from db, and not recalculated
    # TODO -- testing that should be in pipeline/test_measuring.py.  We should just use
    #   here what the fixture gave us
    assert len(ds.measurements) <= len(ds.cutouts.co_dict)  # not all cutouts have saved measurements
    assert len(ds.measurements) == len(ds.measurements)
    assert ds.measurements[0].from_db
    assert not measurer.has_recalculated

    # grab one example measurements object
    m = ds.measurements[0]

    # Make sure the zeropoint we get from the auto-loaded measurements field is the right one
    # (At this point, there are multiple zeropoints in the database, so this is at least sort
    # of a test that the monster query in Measurements.zp is filtering down right.  To really
    # test it, we'd need to get multiple source lists from the new image, each with different
    # provenances.  Make that a TODO?)
    assert m._zp is None
    assert m.zp.id == ds.zp.id

    # check some basic values
    new_im = ds.image
    assert np.allclose(m.aper_radii, m.zp.aper_cor_radii)
    assert np.allclose( m.zp.aper_cor_radii, ds.psf.fwhm_pixels * np.array(aper_radii) )

    # Make sure that flux_psf is the thing we want to muck with below
    assert m.best_aperture == -1

    original_flux_psf = m.flux_psf
    original_flux_psf_err = m.flux_psf_err
    original_flux_ap0 = m.flux_apertures[0]

    # set the flux temporarily to something positive
    m.flux_apertures[0] = 10000
    assert m.mag_apertures[0] == pytest.approx( -2.5 * np.log10(10000) + m.zp.zp + m.zp.aper_cors[0], abs=1e-4 )

    m.flux_psf = 10000
    expected_mag = -2.5 * np.log10(10000) + m.zp.zp
    assert m.mag_psf == pytest.approx( expected_mag, abs=1e-4 )

    # set the flux temporarily to something negative
    m.flux_apertures[0] = -10000
    assert np.isnan(m.mag_apertures[0])

    # check that background is subtracted from the "flux" and "magnitude" properties
    if m.best_aperture == -1:
        assert m.flux == m.flux_psf - m.bkg_mean * m.area_psf
        assert m.magnitude != m.mag_psf  # the magnitude has background subtracted from it
        assert m.magnitude_err > m.mag_psf_err  # the magnitude error is larger because of the error in background
    else:
        assert m.flux == m.flux_apertures[m.best_aperture] - m.bkg_mean * m.area_apertures[m.best_aperture]

    # set the flux and zero point to some randomly chosen values and test the distribution of the magnitude:
    fiducial_zp = m.zp.zp
    original_zp_err = m.zp.dzp
    fiducial_zp_err = 0.03  # more reasonable ZP error value (closer to dflux/flux)
    fiducial_flux = 10000
    fiducial_flux_err = 500
    m.flux_psf_err = fiducial_flux_err
    m.zp.dzp = fiducial_zp_err

    iterations = 1000
    mags = np.zeros(iterations)
    for i in range(iterations):
        m.flux_psf = np.random.normal(fiducial_flux, fiducial_flux_err)
        m.zp.zp = np.random.normal(fiducial_zp, fiducial_zp_err)
        mags[i] = m.magnitude

    m.flux_apertures[m.best_aperture] = fiducial_flux

    # the measured magnitudes should be normally distributed
    assert np.abs(np.std(mags) - m.magnitude_err) < 0.01
    assert np.abs(np.mean(mags) - m.magnitude) < m.magnitude_err * 3   # ...this should fail 0.3% of the time...

    # make sure to return things to their original state
    m.flux_apertures[0] = original_flux_ap0
    m.flux_psf = original_flux_psf
    m.flux_psf_err = original_flux_psf_err
    ds.zp.zp = fiducial_zp
    ds.zp.dzp = original_zp_err

    # TODO: add test for limiting magnitude (issue #143)

    # Test getting cutout image data
    # (Note: I'm not sure what's up with sub_psfflux and sub_psffluxerr.

    m = ds.measurements[1]
    fields = [ 'sub_data', 'ref_data', 'new_data',
               'sub_weight', 'ref_weight', 'new_weight',
               'sub_flags', 'ref_flags', 'new_flags' ]

    def reset_fields():
        for f in fields:
            setattr( m, f'_{f}', None )

    def check_fields_none():
        assert all( getattr( m, f'_{f}' ) is None for f in fields )

    def check_fields_not_none():
        assert all ( getattr( m, f'_{f}' ) is not None for f in fields )

    # Make sure we start clean
    check_fields_none()

    # Make sure we can get stuff explicitly passing cutouts and detections
    m.get_data_from_cutouts( cutouts=ds.cutouts, detections=ds.detections )
    check_fields_not_none()

    reset_fields()
    check_fields_none()

    # Make sure we can get stuff with get_data_from_cutouts pulling cutouts and detections from database.
    m.get_data_from_cutouts()
    check_fields_not_none()

    # Now go through the auto-loaded attributes one by one
    for field in fields:
        reset_fields()
        check_fields_none()
        assert getattr( m, field ) is not None
        check_fields_not_none()


def test_filtering_measurements(ptf_datastore):
    measurements = ptf_datastore.measurements
    m = measurements[0]  # grab the first one as an example

    # test that we can filter on some measurements properties
    with SmartSession() as session:
        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 0)).all()
        assert len(ms) == len(measurements)  # saved measurements will probably have a positive flux

        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 2000)).all()
        assert len(ms) < len(measurements)  # only some measurements have a flux above 2000

        ms = session.scalars(sa.select(Measurements).where(Measurements.bkg_mean > 0)).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have positive background

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.offset_x > 0, Measurements.provenance_id == m.provenance_id
        )).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have positive offsets

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.area_psf >= 0, Measurements.provenance_id == m.provenance_id
        )).all()
        assert len(ms) == len(measurements)  # all measurements have positive psf area

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.width >= 0, Measurements.provenance_id == m.provenance_id
        )).all()
        assert len(ms) == len(measurements)  # all measurements have positive width

        # filter on a specific disqualifier score
        ms = session.scalars(sa.select(Measurements).where(
            Measurements.disqualifier_scores['negatives'].astext.cast(sa.REAL) < 0.1,
            Measurements.provenance_id == m.provenance_id
        )).all()
        assert len(ms) <= len(measurements)


def test_measurements_cannot_be_saved_twice(ptf_datastore):
    m = ptf_datastore.measurements[0]  # grab the first measurement as an example
    # test that we cannot save the same measurements object twice
    m2 = Measurements()
    for key, val in m.__dict__.items():
        if key not in ['_id', '_sa_instance_state']:
            setattr(m2, key, val)  # copy all attributes except the SQLA related ones and the ID

    try:
        with pytest.raises(
                IntegrityError,
                match='duplicate key value violates unique constraint "_measurements_cutouts_provenance_uc"'
        ):
            m2.insert()

        # now change the provenance
        mprov = Provenance.get( m.provenance_id )
        prov = Provenance(
            code_version_id=mprov.code_version_id,
            process=mprov.process,
            parameters=mprov.parameters,
            upstreams=mprov.upstreams,
            is_testing=True,
        )
        prov.parameters['test_parameter'] = uuid.uuid4().hex
        prov.update_id()
        prov.insert_if_needed()
        m2.provenance_id = prov.id
        m2.insert

    finally:
        if 'm2' in locals():
            with SmartSession() as sess:
                sess.execute( sa.delete( Measurements ).where( Measurements._id==m2.id ) )
                sess.commit()


def test_threshold_flagging(ptf_datastore, measurer):

    measurements = ptf_datastore.measurements
    m = measurements[0]  # grab the first one as an example

    measurer.pars.thresholds['negatives'] = 0.3
    measurer.pars.deletion_thresholds['negatives'] = 0.5

    m.disqualifier_scores['negatives'] = 0.1 # set a value that will pass both
    assert measurer.compare_measurement_to_thresholds(m) == "ok"

    m.disqualifier_scores['negatives'] = 0.4 # set a value that will fail one
    assert measurer.compare_measurement_to_thresholds(m) == "bad"

    m.disqualifier_scores['negatives'] = 0.6 # set a value that will fail both
    assert measurer.compare_measurement_to_thresholds(m) == "delete"

    # test what happens if we set deletion_thresholds to unspecified
    #   This should not test at all for deletion
    measurer.pars.deletion_thresholds = {}

    m.disqualifier_scores['negatives'] = 0.1 # set a value that will pass
    assert measurer.compare_measurement_to_thresholds(m) == "ok"

    m.disqualifier_scores['negatives'] = 0.8 # set a value that will fail
    assert measurer.compare_measurement_to_thresholds(m) == "bad"

    # test what happens if we set deletion_thresholds to None
    #   This should set the deletion threshold same as threshold
    measurer.pars.deletion_thresholds = None
    m.disqualifier_scores['negatives'] = 0.1 # set a value that will pass
    assert measurer.compare_measurement_to_thresholds(m) == "ok"

    m.disqualifier_scores['negatives'] = 0.4 # a value that would fail mark
    assert measurer.compare_measurement_to_thresholds(m) == "delete"

    m.disqualifier_scores['negatives'] = 0.9 # a value that would fail both (earlier)
    assert measurer.compare_measurement_to_thresholds(m) == "delete"

# This really ought to be in pipeline/test_measuring.py
def test_deletion_thresh_is_non_critical( ptf_datastore_through_cutouts, measurer ):

    # hard code in the thresholds to ensure no problems arise
    # if the defaults for testing change
    measurer.pars.threshold = {
                'negatives': 0.3,
                'bad pixels': 1,
                'offsets': 5.0,
                'filter bank': 1,
                'bad_flag': 1,
            }

    measurer.pars.deletion_threshold = {
                'negatives': 0.3,
                'bad pixels': 1,
                'offsets': 5.0,
                'filter bank': 1,
                'bad_flag': 1,
            }

    ds1 = DataStore( ptf_datastore_through_cutouts )
    ds2 = DataStore( ptf_datastore_through_cutouts )

    # Gotta remove the 'measuring' provenance from ds1's prov tree
    #  (which I think will also remove it from ds2's, not to mention
    #  ptf_datastore_through_cutout's, as I don't think the copy
    #  construction for DataStore does a deep copy) because we're about
    #  to run measurements with a different set of parameters
    del ds1.prov_tree['measuring']

    ds1 = measurer.run( ds1 )
    ds1provid = ds1.measurements[0].provenance_id


    # Make sure that if we change a deletion threshold, we get
    #   back the same provenance

    # First make sure that the measurements are all cleared out of the database,
    #  so they won't just get reloaded
    with SmartSession() as session:
        session.execute( sa.delete( Measurements ).where( Measurements._id.in_( [ i.id for i in ds1.measurements ] ) ) )
        session.commit()

    measurer.pars.deletion_threshold = None
    # Make sure the data store forgets about its measurements provenance so it will make a new one
    if 'measuring' in ds2.prov_tree:
        del ds2.prov_tree['measuring']
    ds2 = measurer.run( ds2 )

    assert ds2.measurements[0].provenance_id == ds1provid


def test_measurements_forced_photometry(ptf_datastore):
    offset_max = 2.0
    for m in ptf_datastore.measurements:
        if abs(m.offset_x) < offset_max and abs(m.offset_y) < offset_max:
            break
    else:
        raise RuntimeError(f'Cannot find any measurement with offsets less than {offset_max}')

    with pytest.raises( ValueError, match="Must pass PSF if you want to do PSF photometry" ):
        m.get_flux_at_point( m.ra, m.dec, aperture=-1 )

    flux_small_aperture = m.get_flux_at_point(m.ra, m.dec, aperture=1)
    flux_large_aperture = m.get_flux_at_point(m.ra, m.dec, aperture=len(m.aper_radii) - 1)
    flux_psf = m.get_flux_at_point( m.ra, m.dec, aperture=-1, psf=ptf_datastore.psf )
    assert flux_small_aperture[0] == pytest.approx(m.flux_apertures[1], abs=0.01)
    assert flux_large_aperture[0] == pytest.approx(m.flux_apertures[-1], abs=0.01)
    assert flux_psf[0] == pytest.approx(m.flux_psf, abs=0.01)

    # print(f'Flux regular, small: {m.flux_apertures[1]}+-{m.flux_apertures_err[1]} over area: {m.area_apertures[1]}')
    # print(f'Flux regular, big: {m.flux_apertures[-1]}+-{m.flux_apertures_err[-1]} over area: {m.area_apertures[-1]}')
    # print(f'Flux regular, PSF: {m.flux_psf}+-{m.flux_psf_err} over area: {m.area_psf}')
    # print(f'Flux small aperture: {flux_small_aperture[0]}+-{flux_small_aperture[1]} over area: {flux_small_aperture[2]}')
    # print(f'Flux big aperture: {flux_large_aperture[0]}+-{flux_large_aperture[1]} over area: {flux_large_aperture[2]}')
    # print(f'Flux PSF forced: {flux_psf[0]}+-{flux_psf[1]} over area: {flux_psf[2]}')
