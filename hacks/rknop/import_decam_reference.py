import argparse
import uuid

import numpy

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from util.logger import SCLogger

import models.instrument
import models.decam
from models.base import Psycopg2Connection
from models.image import Image
from models.reference import Reference
from models.provenance import Provenance
from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse

# Needed to avoid errors about missing classes later
import models.object  # noqa: F401

from pipeline.data_store import DataStore
from pipeline.top_level import Pipeline

# TODO : figure out what the right way is to get code versions when we aren't
#   using test fixtures...
codeversion_id = '0.0.1'


def import_decam_reference( image, weight, mask, target, hdu, section_id, refset ):
    image_prov = Provenance( process='import_image',
                             parameters={},
                             upstreams=[] )
    image_prov.insert_if_needed()

    SCLogger.info( "Reading image" )

    with fits.open( image ) as img, fits.open( weight ) as wgt, fits.open( mask) as msk:
        img_hdr = img[ hdu ].header
        img_data = img[ hdu ].data
        # wgt_hdr = wgt[ hdu ].header
        wgt_data = wgt[ hdu ].data
        # msk_hdr = msk[ hdu ].header
        msk_data = msk[ hdu ].data
        wcs = WCS( img_hdr )

    # Trust the WCS that's in there to start
    #  for purposes of ra/dec fields

    radec = wcs.pixel_to_world( img_data.shape[1] / 2., img_data.shape[0] / 2. )
    ra = radec.ra.to(u.deg).value
    dec = radec.dec.to(u.deg).value
    lat = radec.galactic.l.to(u.deg).value
    b = radec.galactic.b.to(u.deg).value
    ecl = radec.transform_to( 'geocentricmeanecliptic' )
    ecl_lat = ecl.lat.to(u.deg).value
    ecl_lon = ecl.lon.to(u.deg).value

    xcorner = [ 0., img_data.shape[1]-1., img_data.shape[1]-1., 0. ]
    ycorner = [ 0., 0., img_data.shape[0]-1., img_data.shape[0]-1. ]
    radec = wcs.pixel_to_world( xcorner, ycorner )
    # Don't make assumptions about orientation
    for cra, cdec in zip( radec.ra.to(u.deg).value, radec.dec.to(u.deg).value ):
        if ( cra < ra ) and ( cdec < dec ):
            ra_corner_00 = cra
            dec_corner_00 = cdec
        elif ( cra > ra ) and ( cdec < dec ):
            ra_corner_10 = cra
            dec_corner_10 = cdec
        elif ( cra < ra ) and ( cdec > dec ):
            ra_corner_01 = cra
            dec_corner_01 = cdec
        elif ( cra > ra ) and ( cdec > dec ):
            ra_corner_11 = cra
            dec_corner_11 = cdec
        else:
            raise RuntimeError( "This should never happen" )
    minra = min( ra_corner_00, ra_corner_01 )
    maxra = max( ra_corner_10, ra_corner_11 )
    mindec = min( dec_corner_00, dec_corner_10 )
    maxdec = max( dec_corner_01, dec_corner_11 )

    image = Image( provenance_id=image_prov.id,
                   format='fitsfz',
                   type='ComSci',       # Not really right, but we don't currently have a definition
                   mjd=img_hdr['MJD-OBS'],      # Won't really be right (sum across days), but oh well
                   end_mjd=img_hdr['MJD-END'],  # (same comment)
                   exp_time=img_hdr['EXPTIME'], # (same comment)
                   instrument='DECam',
                   telescope='CTIO-4m',
                   filter=img_hdr['FILTER'],
                   section_id=section_id,
                   project='DECAT-DDF',
                   target=target,
                   ra=ra,
                   dec=dec,
                   gallat=b,
                   gallon=lat,
                   ecllat=ecl_lat,
                   ecllon=ecl_lon,
                   ra_corner_00=ra_corner_00,
                   ra_corner_01=ra_corner_01,
                   ra_corner_10=ra_corner_10,
                   ra_corner_11=ra_corner_11,
                   dec_corner_00=dec_corner_00,
                   dec_corner_01=dec_corner_01,
                   dec_corner_10=dec_corner_10,
                   dec_corner_11=dec_corner_11,
                   minra=minra,
                   maxra=maxra,
                   mindec=mindec,
                   maxdec=maxdec )

    image.header = img_hdr
    image.data = img_data
    image.weight = wgt_data

    # for flags, I know that anything that's non-0 in the weight
    #   image is "bad", somehow, so just set that for our flags
    #   image rather than trying to figure out details.
    # (lensgrinder ended up not worrying about details)

    image.flags = numpy.zeros_like( msk_data, dtype=numpy.int16 )
    image.flags[ msk_data != 0 ] = string_to_bitflag( 'bad pixel', flag_image_bits_inverse )

    SCLogger.info( "Running pipeline for sources / background / wcs / zp" )

    ds = DataStore( image )
    pipeline = Pipeline( pipeline={ 'provenance_tag': 'DECam_manual_refs',
                                    'through_step': 'zp',
                                    'generate_report': False } )
    ds.prov_tree = pipeline.make_provenance_tree( image, ok_no_ref_prov=True )

    # Have to manually run the pipeline steps because pipeline.run()
    #   as it currently exists depends on starting from an Exposure

    SCLogger.info( "Extracting" )
    pipeline.extractor.run( ds )

    SCLogger.info( "Backgrounding" )
    pipeline.backgrounder.run( ds )

    SCLogger.info( "Astrometric" )
    pipeline.astrometor.run( ds )

    SCLogger.info( "Photometric" )
    pipeline.photometor.run( ds )

    SCLogger.info( "Saving data products" )
    ds.save_and_commit()

    # Make the reference

    SCLogger.info( "Creating reference entry" )

    reference_prov = Provenance( process='manual_reference',
                                 parameters={},
                                 upstreams=[image_prov, ds.prov_tree['extraction']] )
    reference_prov.insert_if_needed()

    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( "LOCK TABLE refsets" )
        cursor.execute( "SELECT _id,provenance_id FROM refsets WHERE name=%(name)s", { 'name': refset } )
        row = cursor.fetchone()
        if row is None:
            refsetid = uuid.uuid4()
            cursor.execute( "INSERT INTO refsets(_id,name,description,provenance_id) "
                            "VALUES (%(id)s,%(name)s,'Manually imported DECam Reference', %(provid)s)",
                            { 'id': refsetid, 'name': refset, 'provid': reference_prov.id } )
            conn.commit()
        else:
            refsetid = row[0]
            if reference_prov.id != row[1]:
                raise ValueError( f"Refset {refset} provenance {row[1]} doesn't match "
                                  f"reference_prov.id {reference_prov.id}" )

    ref = Reference( image_id=ds.image.id,
                     sources_id=ds.sources.id,
                     provenance_id=reference_prov.id )
    ref.insert()

    SCLogger.info( "Done" )


def main():
    parser = argparse.ArgumentParser( 'python import_decam_reference.py',
                                      description='Import a DECam image as a reference',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "image", help="FITS file with the reference" )
    parser.add_argument( "weight", help="FITS file with the weight" )
    parser.add_argument( "mask", help="FITS file with the mask" )
    parser.add_argument( "-t", "--target", default="Unknown",
                         help="The target field in the database (field name)" )
    parser.add_argument( "--hdu", type=int, default=0,
                         help="Which HDU has the image (default 0, make 1 for a .fits.fz file)" )
    parser.add_argument( "-s", "--section-id", required=True,
                         help="The section_id (chip, using N1, S1, etc. notation)" )
    parser.add_argument( "-r", "--refset", default="decam_manual",
                         help="Create and add reference provenance to this refset if necessary" )
    args = parser.parse_args()

    import_decam_reference( args.image, args.weight, args.mask,
                            args.target, args.hdu, args.section_id, args.refset )

# ======================================================================


if __name__ == "__main__":
    main()
