import os

import sqlalchemy as sa

from models.base import SmartSession, FileOnDiskMixin
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements

from pipeline.top_level import Pipeline


def match_exposure_to_reference_entry(exposure, reference_entry):
    """Make sure the exposure has the same target, project, filter, and section_id as the reference image."""
    exposure.target = reference_entry.target
    exposure.project = reference_entry.image.project
    exposure.filter = reference_entry.filter


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
    im = session.scalars(
        sa.select(Image).where(Image.exposure_id == exp_id, Image.section_id == str(sec_id))
    ).first()
    assert im is not None
    assert ds.image.id == im.id

    sl = session.scalars(
        sa.select(SourceList).where(SourceList.image_id == im.id, SourceList.is_sub.is_(False))
    ).first()
    assert sl is not None
    assert ds.sources.id == sl.id

    wcs = session.scalars(
        sa.select(WorldCoordinates).where(WorldCoordinates.source_list_id == sl.id)
    ).first()
    assert wcs is not None
    assert ds.wcs.id == wcs.id

    zp = session.scalars(
        sa.select(ZeroPoint).where(ZeroPoint.source_list_id == sl.id)
    ).first()
    assert zp is not None
    assert ds.zp.id == zp.id

    sub = session.scalars(
        sa.select(Image).where(Image.new_image_id == im.id, Image.ref_image_id == ref_id)
    ).first()

    assert sub is not None
    assert ds.sub_image.id == sub.id

    sl = session.scalars(
        sa.select(SourceList).where(SourceList.image_id == sub.id, SourceList.is_sub.is_(True))
    ).first()

    assert sl is not None
    assert ds.detections.id == sl.id

    # TODO: add the cutouts and measurements, but we need to produce them first!


def test_data_flow(exposure, reference_entry):
    """Test that the pipeline runs end-to-end."""
    sec_id = reference_entry.section_id

    ds = None
    try:  # cleanup the file at the end
        # add the exposure to DB and use that ID to run the pipeline
        with SmartSession() as session:
            reference_entry = session.merge(reference_entry)
            match_exposure_to_reference_entry(exposure, reference_entry)

            session.add(exposure)
            session.commit()
            exp_id = exposure.id

            filename = exposure.get_fullpath()
            open(filename, 'a').close()
            ref_id = reference_entry.image.id

        p = Pipeline()
        ds = p.run(exp_id, sec_id)

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

            check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds)

        # feed the pipeline the same data, but missing the upstream data
        attributes = ['exposure', 'image', 'sources', 'wcs', 'zp', 'sub_image', 'detections']

        for i in range(len(attributes)):
            for j in range(i):
                setattr(ds, attributes[j], None)  # get rid of all data up to the current attribute

            ds = p.run(ds)

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds)

        print(ds.image.filepath)
        print(ds.sub_image.filepath)
        # make sure we can remove the data from the end to the beginning and recreate it
        for i in range(len(attributes)):
            for j in range(i):
                # print(f'i= {i}, j= {j}. Removing attribute: {attributes[-j-1]}')

                obj = getattr(ds, attributes[-j-1])
                with SmartSession() as session:
                    obj = obj.recursive_merge(session=session)
                    if isinstance(obj, FileOnDiskMixin):
                        obj.remove_data_from_disk()
                    session.delete(obj)
                    session.commit()

                setattr(ds, attributes[-j-1], None)

            ds = p.run(ds)

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds)

    finally:
        if ds is not None:
            ds.delete_everything()


