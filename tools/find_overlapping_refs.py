import io

from models.base import SmartSession
from models.reference import Reference
from models.image import Image

from util.logger import SCLogger

overlapfrac = 0.7
refset = 'decam_manual'


def main():
    global overlapfrac, refset

    with SmartSession() as sess:
        allrefs = sess.query( Reference ).all()

    for i, ref in enumerate( allrefs ):
        if i % 100 == 0:
            SCLogger.info( f"Doing ref {i} of {len(allrefs)}..." )

        img = ref.image
        overlaps, overlapims = Reference.get_references( image=img,
                                                         filter=img.filter,
                                                         overlapfrac=overlapfrac,
                                                         refset=refset,
                                                         instrument=img.instrument )
        if len(overlaps) > 1:
            strio = io.StringIO()
            strio.write( f"Ref {ref._id} has overlaps!\n" )
            strio.write( f" ...image {img.filepath}\n" )
            strio.write( f" ...zp {img.zero_point_estimate:.2f}  lim { img.lim_mag_estimate:.2f}  "
                         f"seeing {img.fwhm_estimate:.2f}\n" )
            strio.write( f" ...filter {img.filter}, instrument {img.instrument}\n" )
            strio.write( f" ...section {img.section_id}, target {img.target}\n" )
            strio.write( " ...overlaps:\n" )
            for ov, ovim in zip( overlaps, overlapims ):
                ovfrac = Image.get_overlap_frac( img, ovim )
                if ov._id != ref._id:
                    strio.write( f"     {ovfrac:.2f} : ref {ov._id}, image {ovim.filepath}\n" )
                    strio.write( f"               ...section {ovim.section_id}, target {ovim.target}\n" )
                    strio.write( f"               ...zp {ovim.zero_point_estimate:.2f}  "
                                 f"lim {ovim.lim_mag_estimate:.2f}  seeing {ovim.fwhm_estimate:.2f}\n" )
            SCLogger.error( strio.getvalue() )


# ======================================================================-
if __name__ == "__main__":
    main()
