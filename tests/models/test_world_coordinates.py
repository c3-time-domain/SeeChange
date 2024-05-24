import pytest
import hashlib
import os
import pathlib

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from astropy.io import fits
from astropy.wcs import WCS

from models.base import SmartSession
from models.provenance import Provenance
from models.world_coordinates import WorldCoordinates


def test_world_coordinates( ztf_datastore_uncommitted, provenance_base, provenance_extra ):
    image = ztf_datastore_uncommitted.image
    image.instrument = 'DECam' # hack - otherwise invent_filepath will not work as 'ZTF' is not an Instrument
    hdr = image.header

    origwcs = WCS( hdr )
    origscs = origwcs.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )

    # Make sure we can construct a WorldCoordinates object from a WCS object

    wcobj = WorldCoordinates()
    wcobj.wcs = origwcs
    header_excerpt = wcobj.wcs.to_header().tostring( sep='\n', padding=False)
    md5 = hashlib.md5( header_excerpt.encode('ascii') )
    assert md5.hexdigest() == 'a13d6bdd520c5a0314dc751025a62619'

    # Make sure that we can construct a WCS from a WorldCoordinates

    old_wcs = wcobj.wcs
    wcobj = WorldCoordinates()
    wcobj.wcs = old_wcs
    scs = wcobj.wcs.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    for sc, origsc in zip( scs, origscs ):
        assert sc.ra.value == pytest.approx( origsc.ra.value, abs=0.01/3600. )
        assert sc.dec.value == pytest.approx( origsc.dec.value, abs=0.01/3600. )

    # save the WCS to file and DB
    with SmartSession() as session:
        try:
            provenance_base = session.merge(provenance_base)
            provenance_extra = session.merge(provenance_extra)
            image.sources = ztf_datastore_uncommitted.sources
            image.sources.provenance = provenance_extra
            image.sources.save()
            image.psf.provenance = provenance_extra
            image.psf.save()
            image.provenance = provenance_base
            image.save()
            image = image.merge_all(session)

            wcobj.sources = image.sources
            wcobj.provenance = Provenance(
                process='test_world_coordinates',
                code_version=provenance_base.code_version,
                parameters={'test_parameter': 'test_value'},
                upstreams=[provenance_extra],
                is_testing=True,
            )
            wcobj.save()

            session.add(wcobj)
            session.commit()

            # add a second WCS object and make sure we cannot accidentally commit it, too
            wcobj2 = WorldCoordinates()
            wcobj2.wcs = old_wcs
            wcobj2.sources = image.sources
            wcobj2.provenance = wcobj.provenance
            wcobj2.save() # overwrite the save of wcobj

            with pytest.raises(
                    IntegrityError,
                    match='duplicate key value violates unique constraint "_wcs_sources_provenance_uc"'
            ):
                session.add(wcobj2)
                session.commit()
            session.rollback()

            # ensure you cannot overwrite when explicitly setting overwrite=False
            with pytest.raises( OSError, match=".txt already exists" ):
                wcobj2.save(overwrite=False)

            # if we change any of the provenance parameters we should be able to save it
            wcobj2.provenance = Provenance(
                process='test_world_coordinates',
                code_version=provenance_base.code_version,
                parameters={'test_parameter': 'new_test_value'},  # notice we've put another value here
                upstreams=[provenance_extra],
                is_testing=True,
            )
            wcobj2.save(overwrite=False)

            session.add(wcobj2)
            session.commit()

        finally:

            if 'wcobj' in locals():
                # wcobj.delete_from_disk_and_database(session=session)
                if sa.inspect(wcobj).persistent:
                    session.delete(wcobj)
                    image.wcs = None
                    image.sources.wcs = None
            if 'wcobj2' in locals():
                # wcobj2.delete_from_disk_and_database(session=session)
                if sa.inspect(wcobj2).persistent:
                    session.delete(wcobj2)
                    image.wcs = None
                    image.sources.wcs = None
            session.commit()

            if 'image' in locals():
                image.delete_from_disk_and_database(session=session, commit=True)


def test_save_and_load_wcs(ztf_datastore_uncommitted, provenance_base, provenance_extra):
    image = ztf_datastore_uncommitted.image
    image.instrument = 'DECam' # otherwise invent_filepath will not work as 'ZTF' is not an Instrument
    hdr = image.header

    origwcs = WCS( hdr )
    wcobj = WorldCoordinates()
    wcobj.wcs = origwcs
    wcobj.sources = image.sources
    wcobj.provenance = Provenance(
                process='test_world_coordinates',
                code_version=provenance_base.code_version,
                parameters={'test_parameter': 'test_value'},
                upstreams=[provenance_extra],
                is_testing=True,
            )

    with SmartSession() as session:
        try:
            wcobj.save()

            txtpath = pathlib.Path( wcobj.local_path ) / f'{wcobj.filepath}'

            # check for an error if the file is not found when loading
            os.remove(txtpath)
            with pytest.raises( OSError, match="file is missing" ):
                wcobj.load()
            
            # ensure you can create an identical wcs from a saved one
            wcobj.save()
            wcobj2 = WorldCoordinates()
            wcobj2.load( txtpath=txtpath )

            assert wcobj2.wcs.to_header() == wcobj.wcs.to_header()

            session.commit()

        finally:
            if "wcobj" in locals():
                wcobj.delete_from_disk_and_database(session=session)
            if "wcobj2" in locals():
                wcobj2.delete_from_disk_and_database(session=session)
