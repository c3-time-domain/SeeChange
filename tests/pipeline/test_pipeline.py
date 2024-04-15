import os
import pytest
import shutil
import sqlalchemy as sa

from models.base import SmartSession, FileOnDiskMixin, _logger
from models.provenance import Provenance
from models.image import Image, image_upstreams_association_table
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements

from pipeline.top_level import Pipeline


def check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds):
    """
    Check that all the required objects are saved on the database
    and in the datastore, after running the entire pipeline.

    Parameters
    ----------
    exp_id: int
        The exposure ID.
    sec_id: str or int
        The section ID.
    ref_id: int
        The reference image ID.
    session: sqlalchemy.orm.session.Session
        The database session
    ds: datastore.DataStore
        The datastore object
    """
    # find the image
    im = session.scalars(
        sa.select(Image).where(
            Image.exposure_id == exp_id,
            Image.section_id == str(sec_id),
            Image.provenance_id == ds.image.provenance_id,
        )
    ).first()
    assert im is not None
    assert ds.image.id == im.id

    # find the extracted sources
    sources = session.scalars(
        sa.select(SourceList).where(
            SourceList.image_id == im.id,
            SourceList.is_sub.is_(False),
            SourceList.provenance_id == ds.sources.provenance_id,
        )
    ).first()
    assert sources is not None
    assert ds.sources.id == sources.id

    # find the PSF
    psf = session.scalars(
        sa.select(PSF).where(PSF.image_id == im.id, PSF.provenance_id == ds.psf.provenance_id)
    ).first()
    assert psf is not None
    assert ds.psf.id == psf.id

    # find the WorldCoordinates object
    wcs = session.scalars(
        sa.select(WorldCoordinates).where(
            WorldCoordinates.sources_id == sources.id,
            WorldCoordinates.provenance_id == ds.wcs.provenance_id,
        )
    ).first()
    assert wcs is not None
    assert ds.wcs.id == wcs.id

    # find the ZeroPoint object
    zp = session.scalars(
        sa.select(ZeroPoint).where(ZeroPoint.sources_id == sources.id, ZeroPoint.provenance_id == ds.zp.provenance_id)
    ).first()
    assert zp is not None
    assert ds.zp.id == zp.id

    # find the subtraction image
    aliased_table = sa.orm.aliased(image_upstreams_association_table)
    sub = session.scalars(
        sa.select(Image).join(
            image_upstreams_association_table,
            sa.and_(
                image_upstreams_association_table.c.upstream_id == ref_id,
                image_upstreams_association_table.c.downstream_id == Image.id,
            )
        ).join(
            aliased_table,
            sa.and_(
                aliased_table.c.upstream_id == im.id,
                aliased_table.c.downstream_id == Image.id,
            )
        )
    ).first()

    assert sub is not None
    assert ds.sub_image.id == sub.id

    # find the detections SourceList
    det = session.scalars(
        sa.select(SourceList).where(
            SourceList.image_id == sub.id,
            SourceList.is_sub.is_(True),
            SourceList.provenance_id == ds.detections.provenance_id,
        )
    ).first()

    assert det is not None
    assert ds.detections.id == det.id

    # find the Cutouts list
    cutouts = session.scalars(
        sa.select(Cutouts).where(
            Cutouts.sources_id == det.id,
            Cutouts.provenance_id == ds.cutouts[0].provenance_id,
        )
    ).all()
    assert len(cutouts) > 0
    assert len(ds.cutouts) == len(cutouts)
    assert set([c.id for c in ds.cutouts]) == set([c.id for c in cutouts])

    # TODO: add the measurements, but we need to produce them first!


def test_parameters( test_config ):
    """Test that pipeline parameters are being set properly"""

    # Verify that we _enforce_no_new_attrs works
    kwargs = { 'pipeline': { 'keyword_does_not_exist': 'testing' } }
    with pytest.raises( AttributeError, match='object has no attribute' ):
        failed = Pipeline( **kwargs )

    # Verify that we can override from the yaml config file
    pipeline = Pipeline()
    assert not pipeline.preprocessor.pars['use_sky_subtraction']
    assert pipeline.astro_cal.pars['cross_match_catalog'] == 'gaia_dr3'
    assert pipeline.astro_cal.pars['catalog'] == 'gaia_dr3'
    assert pipeline.subtractor.pars['method'] == 'zogy'

    # Verify that manual override works for all parts of pipeline
    overrides = { 'preprocessing': { 'steps': [ 'overscan', 'linearity'] },
                  # 'extraction': # Currently has no parameters defined
                  'astro_cal': { 'cross_match_catalog': 'override' },
                  'photo_cal': { 'cross_match_catalog': 'override' },
                  'subtraction': { 'method': 'override' },
                  'detection': { 'threshold': 3.14 },
                  'cutting': { 'cutout_size': 666 },
                  'measuring': { 'chosen_aperture': 1 }
                 }
    pipelinemodule = { 'preprocessing': 'preprocessor',
                       'subtraction': 'subtractor',
                       'detection': 'detector',
                       'cutting': 'cutter',
                       'measuring': 'measurer'
                      }

    # TODO: this is based on a temporary "example_pipeline_parameter" that will be removed later
    pipeline = Pipeline( pipeline={ 'example_pipeline_parameter': -999 } )
    assert pipeline.pars['example_pipeline_parameter'] == -999

    pipeline = Pipeline( **overrides )
    for module, subst in overrides.items():
        if module in pipelinemodule:
            pipelinemod = getattr( pipeline, pipelinemodule[module] )
        else:
            pipelinemod = getattr( pipeline, module )
        for key, val in subst.items():
            assert pipelinemod.pars[key] == val


def test_data_flow(decam_exposure, decam_reference, decam_default_calibrators, archive):
    """Test that the pipeline runs end-to-end."""
    exposure = decam_exposure

    ref = decam_reference
    sec_id = ref.section_id
    try:  # cleanup the file at the end
        p = Pipeline()
        assert p.extractor.pars.threshold != 3.14
        assert p.detector.pars.threshold != 3.14

        ds = p.run(exposure, sec_id)

        # commit to DB using this session
        with SmartSession() as session:
            ds.save_and_commit(session=session)

        # use a new session to query for the results
        with SmartSession() as session:
            # check that everything is in the database
            provs = session.scalars(sa.select(Provenance)).all()
            assert len(provs) > 0
            prov_processes = [p.process for p in provs]
            expected_processes = ['preprocessing', 'extraction', 'astro_cal', 'photo_cal', 'subtraction', 'detection']
            for process in expected_processes:
                assert process in prov_processes

            check_datastore_and_database_have_everything(exposure.id, sec_id, ref.image.id, session, ds)

        # feed the pipeline the same data, but missing the upstream data. TODO: add cutouts and measurements
        attributes = ['image', 'sources', 'wcs', 'zp', 'sub_image', 'detections']

        for i in range(len(attributes)):
            for j in range(i + 1):
                setattr(ds, attributes[j], None)  # get rid of all data up to the current attribute
            # _logger.debug(f'removing attributes up to {attributes[i]}')
            ds = p.run(ds)  # for each iteration, we should be able to recreate the data

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exposure.id, sec_id, ref.image.id, session, ds)

        # make sure we can remove the data from the end to the beginning and recreate it
        for i in range(len(attributes)):
            for j in range(i):
                obj = getattr(ds, attributes[-j-1])
                if isinstance(obj, FileOnDiskMixin):
                    obj.delete_from_disk_and_database(session=session, commit=True)

                setattr(ds, attributes[-j-1], None)

            ds = p.run(ds)  # for each iteration, we should be able to recreate the data

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exposure.id, sec_id, ref.image.id, session, ds)

    finally:
        if 'ds' in locals():
            ds.delete_everything()
        # added this cleanup to make sure the temp data folder is cleaned up
        # this should be removed after we add datastore failure modes (issue #150)
        shutil.rmtree(os.path.join(os.path.dirname(exposure.get_fullpath()), '115'), ignore_errors=True)
        shutil.rmtree(os.path.join(archive.test_folder_path, '115'), ignore_errors=True)


def test_datastore_delete_everything(decam_datastore):
    im = decam_datastore.image
    sources = decam_datastore.sources
    psf = decam_datastore.psf
    sub = decam_datastore.sub_image
    det = decam_datastore.detections
    cutouts_list = decam_datastore.cutouts
    measurements_list = decam_datastore.measurements

    # make sure we can delete everything
    decam_datastore.delete_everything()

    # make sure everything is deleted
    for path in im.get_fullpath(as_list=True):
        assert not os.path.exists(path)

    assert not os.path.exists(sources.get_fullpath())

    for path in psf.get_fullpath(as_list=True):
        assert not os.path.exists(path)

    for path in sub.get_fullpath(as_list=True):
        assert not os.path.exists(path)

    assert not os.path.exists(det.get_fullpath())

    assert not os.path.exists(cutouts_list[0].get_fullpath())

    # check these don't exist on the DB:
    with SmartSession() as session:
        assert session.scalars(sa.select(Image).where(Image.id == im.id)).first() is None
        assert session.scalars(sa.select(SourceList).where(SourceList.id == sources.id)).first() is None
        assert session.scalars(sa.select(PSF).where(PSF.id == psf.id)).first() is None
        assert session.scalars(sa.select(Image).where(Image.id == sub.id)).first() is None
        assert session.scalars(sa.select(SourceList).where(SourceList.id == det.id)).first() is None
        assert session.scalars(sa.select(Cutouts).where(Cutouts.id == cutouts_list[0].id)).first() is None
        assert session.scalars(
            sa.select(Measurements).where(Measurements.id == measurements_list[0].id)
        ).first() is None
