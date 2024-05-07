import sys
import logging
import argparse

import numpy

import astropy
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from models.base import SmartSession
import models.instrument
import models.decam
from models.image import Image
from models.provenance import Provenance, CodeVersion
from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse

# TODO : figure out what the right way is to get code versions when we aren't
#   using test fixtures...
codeversion_id = 'hack_0.1'
prov_process = 'import_image'
prov_params = {}
prov_upstreams = []

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

    with SmartSession() as sess:

        # Get the provenance we'll use for the imported references

        cvs = sess.query( CodeVersion ).filter( CodeVersion.id == 'hack_0.1' ).all()
        if len( cvs ) == 0:
            code_ver = CodeVersion( id='hack_0.1' )
            code_ver.update()
            sess.add( code_ver )
            sess.commit()
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
            sess.add( prov )
            sess.commit()
            provs = ( sess.query( Provenance )
                      .filter( Provenance.process == prov_process )
                      .filter( Provenance.code_version == code_ver )
                      .filter( Provenance.parameters == prov_params ) ).all()
            for iprov in provs:
                if len( iprov.upstreams ) == 0:
                    prov = iprov
                    break

        with fits.open( args.image ) as img, fits.open( args.weight ) as wgt, fits.open( args.mask) as msk:
            img_hdr = img[ args.hdu ].header
            img_data = img[ args.hdu ].data
            wgt_hdr = wgt[ args.hdu ].header
            wgt_data = wgt[ args.hdu ].data
            msk_hdr = msk[ args.hdu ].header
            msk_data = msk[ args.hdu ].data
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
                       section_id=args.section_id,
                       project='DECAT-DDF',
                       target=args.target,
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

        image.flags = numpy.zeros_like( msk_data, dtype=numpy.uint16 )
        image.flags[ msk_data != 0 ] = string_to_bitflag( 'bad pixel', flag_image_bits_inverse )

        image.save()
        sess.add( image )
        sess.commit()

# ======================================================================
if __name__ == "__main__":
    main()
