import pytest
import uuid
import numpy as np
import os

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.cutouts import Cutouts
from models.measurements import Measurements


def test_measurements_attributes(measurer, ptf_datastore, test_config):

    aper_radii = test_config.value('extraction.sources.apertures')
    ds = measurer.run(ptf_datastore.cutouts)
    # check that the measurer actually loaded the measurements from db, and not recalculated
    assert len(ds.measurements) <= len(ds.cutouts.co_dict)  # not all cutouts have saved measurements
    assert len(ds.measurements) == len(ptf_datastore.measurements)
    assert ds.measurements[0].from_db
    assert not measurer.has_recalculated

    # grab one example measurements object
    m = ds.measurements[0]
    new_im = m.cutouts.sources.image.new_image
    assert np.allclose(m.aper_radii, new_im.zp.aper_cor_radii)
    assert np.allclose(
        new_im.zp.aper_cor_radii,
        new_im.psf.fwhm_pixels * np.array(aper_radii),
    )
    assert m.mjd == new_im.mjd
    assert m.exp_time == new_im.exp_time
    assert m.filter == new_im.filter

    original_flux = m.flux_apertures[m.best_aperture]

    # set the flux temporarily to something positive
    m.flux_apertures[0] = 1000
    assert m.mag_apertures[0] == -2.5 * np.log10(1000) + new_im.zp.zp + new_im.zp.aper_cors[0]

    m.flux_psf = 1000
    expected_mag = -2.5 * np.log10(1000) + new_im.zp.zp
    assert m.mag_psf == expected_mag

    # set the flux temporarily to something negative
    m.flux_apertures[0] = -1000
    assert np.isnan(m.mag_apertures[0])

    # check that background is subtracted from the "flux" and "magnitude" properties
    if m.best_aperture == -1:
        assert m.flux == m.flux_psf - m.bkg_mean * m.area_psf
        assert m.magnitude != m.mag_psf  # the magnitude has background subtracted from it
        assert m.magnitude_err > m.mag_psf_err  # the magnitude error is larger because of the error in background
    else:
        assert m.flux == m.flux_apertures[m.best_aperture] - m.bkg_mean * m.area_apertures[m.best_aperture]

    # set the flux and zero point to some randomly chosen values and test the distribution of the magnitude:
    fiducial_zp = new_im.zp.zp
    original_zp_err = new_im.zp.dzp
    fiducial_zp_err = 0.1  # more reasonable ZP error value
    fiducial_flux = 1000
    fiducial_flux_err = 50
    m.flux_apertures_err[m.best_aperture] = fiducial_flux_err
    new_im.zp.dzp = fiducial_zp_err

    iterations = 1000
    mags = np.zeros(iterations)
    for i in range(iterations):
        m.flux_apertures[m.best_aperture] = np.random.normal(fiducial_flux, fiducial_flux_err)
        new_im.zp.zp = np.random.normal(fiducial_zp, fiducial_zp_err)
        mags[i] = m.magnitude

    m.flux_apertures[m.best_aperture] = fiducial_flux

    # the measured magnitudes should be normally distributed
    assert np.abs(np.std(mags) - m.magnitude_err) < 0.01
    assert np.abs(np.mean(mags) - m.magnitude) < m.magnitude_err * 3

    # make sure to return things to their original state
    m.flux_apertures[m.best_aperture] = original_flux
    new_im.zp.dzp = original_zp_err

    #test limiting magnitude estimation
    srcList = ptf_datastore.sources
    #if interactive testing enabled, make and save a Magnitude vs SNR plot (for debugging)
    makePlot = os.getenv('INTERACTIVE')
    if makePlot:
        limMagResults = srcList.estimate_lim_mag(aperture=1, givePlotParams=makePlot)
        limMagEst = limMagResults[0]
        snrs = limMagResults[1]
        mags = limMagResults[2]
        grad = limMagResults[3]
        intercept = limMagResults[4]
    else:
        limMagEst = srcList.estimate_lim_mag(aperture=1)

    #Skip the lim mag test if using PSF photometry (i.e. if limMagEst==None)
    if limMagEst != None:
        if makePlot: 
            xdata = np.linspace(np.log(3),np.log(20),1000)
            plt.plot(snrs,mags,linewidth=0,marker='o',c='midnightblue')
            plt.plot(xdata, grad * xdata + intercept, c='firebrick')
            plt.xlabel('log SNR')
            plt.ylabel('magnitude')
            plt.title('Limiting magntiude = {:.2f} mag'.format(limMagEst))
            ymin,ymax = plt.gca().get_ylim()
            plt.vlines(x=np.log(5),ymin=ymin,ymax=ymax)
            plt.hlines(y=limMagEst,xmin=np.log(3),xmax=np.log(20))
            plt.xlim(np.log(3),np.log(20))
            plt.ylim(ymin,ymax)
            plt.savefig('plots/snr_mag_plot.png')
            plt.show()

        #check the limiting magnitude is consistent with previous runs
        assert limMagEst == pytest.approx(20.00, abs=0.5)

    from pipeline.data_store import DataStore
    ds = DataStore(image_id=1126)
    im = ds.get_image()
    sl = ds.get_sources()
    # import pdb; pdb.set_trace()
    limMagResults = sl.estimate_lim_mag(aperture=1, givePlotParams=True)
    limMagEst = limMagResults[0]
    snrs = limMagResults[1]
    mags = limMagResults[2]
    grad = limMagResults[3]
    intercept = limMagResults[4]

    xdata = np.linspace(np.log(3),np.log(20),1000)
    plt.plot(snrs,mags,linewidth=0,marker='o',c='midnightblue')
    plt.plot(xdata, grad * xdata + intercept, c='firebrick')
    plt.xlabel('log SNR')
    plt.ylabel('magnitude')
    plt.title('Limiting magntiude = {:.2f} mag'.format(limMagEst))
    ymin,ymax = plt.gca().get_ylim()
    plt.vlines(x=np.log(5),ymin=ymin,ymax=ymax)
    plt.hlines(y=limMagEst,xmin=np.log(3),xmax=np.log(20))
    plt.xlim(np.log(3),np.log(20))
    plt.ylim(ymin,ymax)
    plt.savefig('plots/snr_mag_plot.png')
    plt.show()
    # import pdb; pdb.set_trace()





def test_filtering_measurements(ptf_datastore):
    measurements = ptf_datastore.measurements
    m = measurements[0]  # grab the first one as an example

    # test that we can filter on some measurements properties
    with SmartSession() as session:
        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 0)).all()
        assert len(ms) == len(measurements)  # saved measurements will probably have a positive flux

        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 100)).all()
        assert len(ms) < len(measurements)  # only some measurements have a flux above 100

        ms = session.scalars(
            sa.select(Measurements).join(Cutouts).join(SourceList).join(Image).where(
                Image.mjd == m.mjd, Measurements.provenance_id == m.provenance.id
            )).all()
        assert len(ms) == len(measurements)  # all measurements have the same MJD

        ms = session.scalars(
            sa.select(Measurements).join(Cutouts).join(SourceList).join(Image).where(
                Image.exp_time == m.exp_time, Measurements.provenance_id == m.provenance.id
            )).all()
        assert len(ms) == len(measurements)  # all measurements have the same exposure time

        ms = session.scalars(
            sa.select(Measurements).join(Cutouts).join(SourceList).join(Image).where(
                Image.filter == m.filter, Measurements.provenance_id == m.provenance.id
            )).all()
        assert len(ms) == len(measurements)  # all measurements have the same filter

        ms = session.scalars(sa.select(Measurements).where(Measurements.bkg_mean > 0)).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have positive background

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.offset_x > 0, Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have positive offsets

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.area_psf >= 0, Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) == len(measurements)  # all measurements have positive psf area

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.width >= 0, Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) == len(measurements)  # all measurements have positive width

        # filter on a specific disqualifier score
        ms = session.scalars(sa.select(Measurements).where(
            Measurements.disqualifier_scores['negatives'].astext.cast(sa.REAL) < 0.1,
            Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) <= len(measurements)


def test_measurements_cannot_be_saved_twice(ptf_datastore):
    m = ptf_datastore.measurements[0]  # grab the first measurement as an example
    # test that we cannot save the same measurements object twice
    m2 = Measurements()
    for key, val in m.__dict__.items():
        if key not in ['id', '_sa_instance_state']:
            setattr(m2, key, val)  # copy all attributes except the SQLA related ones

    with SmartSession() as session:
        try:
            with pytest.raises(
                    IntegrityError,
                    match='duplicate key value violates unique constraint "_measurements_cutouts_provenance_uc"'
            ):
                session.add(m2)
                session.commit()

            session.rollback()

            # now change the provenance
            prov = Provenance(
                code_version=m.provenance.code_version,
                process=m.provenance.process,
                parameters=m.provenance.parameters,
                upstreams=m.provenance.upstreams,
                is_testing=True,
            )
            prov.parameters['test_parameter'] = uuid.uuid4().hex
            prov.update_id()
            m2.provenance = prov
            session.add(m2)
            session.commit()

        finally:
            if 'm' in locals() and sa.inspect(m).persistent:
                session.delete(m)
                session.commit()
            if 'm2' in locals() and sa.inspect(m2).persistent:
                session.delete(m2)
                session.commit()


def test_threshold_flagging(ptf_datastore, measurer):

    measurements = ptf_datastore.measurements
    m = measurements[0]  # grab the first one as an example

    m.provenance.parameters['thresholds']['negatives'] = 0.3
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


def test_deletion_thresh_is_non_critical(ptf_datastore, measurer):

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

    ds1 = measurer.run(ptf_datastore.cutouts)

    # This run should behave identical to the above
    measurer.pars.deletion_threshold = None
    ds2 = measurer.run(ptf_datastore.cutouts)

    m1 = ds1.measurements[0]
    m2 = ds2.measurements[0]

    assert m1.provenance.id == m2.provenance.id


def test_measurements_forced_photometry(ptf_datastore):
    offset_max = 2.0
    for m in ptf_datastore.measurements:
        if abs(m.offset_x) < offset_max and abs(m.offset_y) < offset_max:
            break
    else:
        raise RuntimeError(f'Cannot find any measurement with offsets less than {offset_max}')

    flux_small_aperture = m.get_flux_at_point(m.ra, m.dec, aperture=1)
    flux_large_aperture = m.get_flux_at_point(m.ra, m.dec, aperture=len(m.aper_radii) - 1)
    flux_psf = m.get_flux_at_point(m.ra, m.dec, aperture=-1)
    assert flux_small_aperture[0] == pytest.approx(m.flux_apertures[1], abs=0.01)
    assert flux_large_aperture[0] == pytest.approx(m.flux_apertures[-1], abs=0.01)
    assert flux_psf[0] == pytest.approx(m.flux_psf, abs=0.01)

    # print(f'Flux regular, small: {m.flux_apertures[1]}+-{m.flux_apertures_err[1]} over area: {m.area_apertures[1]}')
    # print(f'Flux regular, big: {m.flux_apertures[-1]}+-{m.flux_apertures_err[-1]} over area: {m.area_apertures[-1]}')
    # print(f'Flux regular, PSF: {m.flux_psf}+-{m.flux_psf_err} over area: {m.area_psf}')
    # print(f'Flux small aperture: {flux_small_aperture[0]}+-{flux_small_aperture[1]} over area: {flux_small_aperture[2]}')
    # print(f'Flux big aperture: {flux_large_aperture[0]}+-{flux_large_aperture[1]} over area: {flux_large_aperture[2]}')
    # print(f'Flux PSF forced: {flux_psf[0]}+-{flux_psf[1]} over area: {flux_psf[2]}')
