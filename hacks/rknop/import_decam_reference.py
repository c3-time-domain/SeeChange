import sys
import logging
import argparse

import numpy

import astropy
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from util.config import Config
from util.logger import SCLogger

from models.base import SmartSession, safe_merge
import models.instrument
import models.decam
from models.image import Image
from models.reference import Reference
from models.provenance import Provenance, CodeVersion
from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse

from pipeline.data_store import DataStore
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator

# TODO : figure out what the right way is to get code versions when we aren't
#   using test fixtures...
codeversion_id = 'hack_0.1'
prov_process = 'import_image'
prov_params = {}
prov_upstreams = []

def import_decam_reference( image, weight, mask, target, hdu, section_id ):
    config = Config.get()

    # This is bad.  It opens sessions and holds them open.  If too many
    #   are running at once, it will overload the database server.
    # Hopefully since this is a hack one-off, we can just cope.
    with SmartSession() as sess:

        # Get the provenance we'll use for the imported references
        # TODO : when I run a bunch of processes at once I'm getting
        #   errors about the code version already existing.
        # Need to really understand how to cope with this sort of thing.

        cvs = sess.query( CodeVersion ).filter( CodeVersion.id == 'hack_0.1' ).all()
        if len( cvs ) == 0:
            try:
                code_ver = CodeVersion( id='hack_0.1' )
                code_ver.update()
                sess.merge( code_ver )
                sess.commit()
            except Exception as ex:
                SCLogger.warning( "Got error trying to create code version, "
                                  "going to assume it's a race condition and all is well." )
                sess.rollback()
        code_ver = sess.query( CodeVersion ).filter( CodeVersion.id == 'hack_0.1' ).first()

        # We should make a get_or_create method for Provenance
        prov = None
        provs = ( sess.query( Provenance )
                  .filter( Provenance.process == prov_process )
                  .filter( Provenance.code_version == code_ver )
                  .filter( Provenance.parameters == prov_params ) ).all()
        for iprov in provs:
            if len( iprov.upstreams ) == 0:
                prov = iprov
                break
        if prov is None:
            prov = Provenance( process = prov_process, code_version = code_ver,
                               parameters = prov_params, upstreams = prov_upstreams )

            prov.update_id()
            prov = sess.merge( prov )
            sess.commit()
            provs = ( sess.query( Provenance )
                      .filter( Provenance.process == prov_process )
                      .filter( Provenance.code_version == code_ver )
                      .filter( Provenance.parameters == prov_params ) ).all()
            for iprov in provs:
                if len( iprov.upstreams ) == 0:
                    prov = iprov
                    break

        SCLogger.info( "Reading image" )

        with fits.open( image ) as img, fits.open( weight ) as wgt, fits.open( mask) as msk:
            img_hdr = img[ hdu ].header
            img_data = img[ hdu ].data
            wgt_hdr = wgt[ hdu ].header
            wgt_data = wgt[ hdu ].data
            msk_hdr = msk[ hdu ].header
            msk_data = msk[ hdu ].data
            wcs = WCS( img_hdr )

        # Trust the WCS that's in there to start
        #  for purposes of ra/dec fields

        radec = wcs.pixel_to_world( img_data.shape[1] / 2., img_data.shape[0] / 2. )
        ra = radec.ra.to(u.deg).value
        dec = radec.dec.to(u.deg).value
        l = radec.galactic.l.to(u.deg).value
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

        image = Image( provenance=prov,
                       format='fits',
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
                       gallon=l,
                       ecllat=ecl_lat,
                       ecllon=ecl_lon,
                       ra_corner_00=ra_corner_00,
                       ra_corner_01=ra_corner_01,
                       ra_corner_10=ra_corner_10,
                       ra_corner_11=ra_corner_11,
                       dec_corner_00=dec_corner_00,
                       dec_corner_01=dec_corner_01,
                       dec_corner_10=dec_corner_10,
                       dec_corner_11=dec_corner_11 )

        image.header = img_hdr
        image.data = img_data
        image.weight = wgt_data

        # for flags, I know that anything that's non-0 in the weight
        #   image is "bad", somehow, so just set that for our flags
        #   image rather than trying to figure out details.
        # (lensgrinder ended up not worrying about details)

        image.flags = numpy.zeros_like( msk_data, dtype=numpy.int16 )
        image.flags[ msk_data != 0 ] = string_to_bitflag( 'bad pixel', flag_image_bits_inverse )

        ds = DataStore( image, session=sess )

        # Extract sources

        SCLogger.info( "Extracting sources" )

        extraction_config = config.value( 'extraction', {} )
        extractor = Detector( **extraction_config )
        ds = extractor.run( ds )

        # WCS

        SCLogger.info( "Astrometric calibration" )

        astro_cal_config = config.value( 'astro_cal', {} )
        astrometor = AstroCalibrator( **astro_cal_config )
        ds = astrometor.run( ds )

        # ZP

        SCLogger.info( "Photometric calibration" )

        photo_cal_config = config.value( 'photo_cal', {} )
        photomotor = PhotCalibrator( **photo_cal_config )
        ds = photomotor.run( ds )

        SCLogger.info( "Saving data products" )

        ds.save_and_commit()

        # Make the reference

        SCLogger.info( "Creating reference entry" )

        # Because SQLAlchmey is annoying and confusing and generally
        # makes me want to break everything around me whenever I have to
        # interact with it, we have to make sure to use the object that
        # SQLAlchemy has properly blessed when referring to the iamge.

        image = ds.image

        ref = Reference( image=image, target=image.target, filter=image.filter, section_id=image.section_id,
                         validity_start='2010-01-01', validity_end='2099-12-31' )
        ref.make_provenance()
        ref = sess.merge( ref )

        sess.commit()


def main():
    parser = argparse.ArgumentParser( 'Import a DECam image as a reference' )
    parser.add_argument( "image", help="FITS file with the reference" )
    parser.add_argument( "weight", help="FITS file with the weight" )
    parser.add_argument( "mask", help="FITS file with the mask" )
    parser.add_argument( "-t", "--target", default="Unknown",
                         help="The target field in the database (field name)" )
    parser.add_argument( "--hdu", type=int, default=0,
                         help="Which HDU has the image (default 0, make 1 for a .fits.fz file)" )
    parser.add_argument( "-s", "--section-id", required=True,
                         help="The section_id (chip, using N1, S1, etc. notation)" )
    args = parser.parse_args()

    import_decam_reference( args.image, args.weight, args.mask, args.target, args.hdu, args.section_id )

# ======================================================================
if __name__ == "__main__":
    main()
