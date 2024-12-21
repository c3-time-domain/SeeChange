import pytest
import hashlib
import os
import pathlib

import psycopg2.errors

from astropy.wcs import WCS

from models.world_coordinates import WorldCoordinates


def test_world_coordinates( ztf_datastore_uncommitted, provenance_base, provenance_extra ):
    ds = ztf_datastore_uncommitted
    ds.image.instrument = 'DECam'    # hack - otherwise invent_filepath will not work as 'ZTF' is not an Instrument

    origwcs = WCS( ds.image.header )
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
    try:
        ds.image.provenance_id = provenance_base.id
        ds.image.save()
        ds.image.insert()
        ds.sources.provenance_id = provenance_extra.id
        ds.sources.save()
        ds.sources.insert()
        ds.psf.save( image=ds.image, sources=ds.sources )

        wcobj.sources_id = ds.sources.id
        wcobj.save( image=ds.image, sources=ds.sources )
        wcobj.insert()

        # add a second WCS object and make sure we cannot accidentally commit it, too
        wcobj2 = WorldCoordinates()
        wcobj2.wcs = old_wcs
        wcobj2.sources_id = ds.sources.id
        wcobj2.save( image=ds.image, sources=ds.sources ) # overwrite the save of wcobj

        with pytest.raises( psycopg2.errors.UniqueViolation,
                            match='duplicate key value violates unique constraint "ix_world_coordinates_filepath"' ):
            wcobj2.insert()

        # ensure you cannot overwrite when explicitly setting overwrite=False
        with pytest.raises( OSError, match=".txt already exists" ):
            wcobj2.save(overwrite=False)

    finally:

        if 'wcobj' in locals():
            wcobj.delete_from_disk_and_database()

        if 'wcobj2' in locals():
            wcobj2.delete_from_disk_and_database()

        ds.sources.delete_from_disk_and_database()
        ds.image.delete_from_disk_and_database()


def test_save_and_load_wcs(ztf_datastore_uncommitted, provenance_base, provenance_extra):
    ds = ztf_datastore_uncommitted
    ds.image.instrument = 'DECam' # otherwise invent_filepath will not work as 'ZTF' is not an Instrument
    ds.image.provenance_id = provenance_base.id
    ds.sources.provenance_id = provenance_extra.id

    origwcs = WCS( ds.image.header )
    wcobj = WorldCoordinates()
    wcobj.wcs = origwcs
    wcobj.sources_id = ds.sources.id

    try:
        wcobj.save( image=ds.image, sources=ds.sources )
        txtpath = pathlib.Path( wcobj.local_path ) / f'{wcobj.filepath}'

        # check for an error if the file is not found when loading
        os.remove(txtpath)
        with pytest.raises( OSError, match="File .* not found!" ):
            wcobj.load( download=False )

        # ensure you can create an identical wcs from a saved one
        wcobj.save( image=ds.image, sources=ds.sources )
        wcobj2 = WorldCoordinates()
        wcobj2.load( txtpath=txtpath )

        assert wcobj2.wcs.to_header() == wcobj.wcs.to_header()

    finally:
        if "wcobj" in locals():
            wcobj.delete_from_disk_and_database()
        if "wcobj2" in locals():
            wcobj2.delete_from_disk_and_database()
