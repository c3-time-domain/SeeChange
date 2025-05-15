import datetime

import numpy as np
import psycopg2.extras

from pipeline.parameters import Parameters
from pipeline.coaddition import CoaddPipeline
from pipeline.data_store import DataStore, ProvenanceTree

from models.base import SmartSession, Psycopg2Connection
from models.provenance import Provenance
from models.reference import Reference
from models.image import Image
from models.refset import RefSet

from util.config import Config
from util.logger import SCLogger
from util.util import parse_dateobs


class ParsRefMaker(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = self.add_par(
            'name',
            'default',
            str,
            'Name of the reference set. ',
            critical=False,  # the name of the refset is not in the Reference provenance!
            # this means multiple refsets can refer to the same Reference provenance
        )

        self.description = self.add_par(
            'description',
            '',
            str,
            'Description of the reference set. ',
            critical=False,
        )

        self.start_time = self.add_par(
            'start_time',
            None,
            (None, str, float, datetime.datetime),
            'Only use images taken after this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the start time. ',
            critical=True,
        )

        self.end_time = self.add_par(
            'end_time',
            None,
            (None, str, float, datetime.datetime),
            'Only use images taken before this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the end time. ',
            critical=True,
        )

        self.corner_distance = self.add_par(
            'corner_distance',
            0.8,
            (None, float),
            ( 'When finding references, make sure that we have at least min_number references overlapping '
              'nine positions on the rectangle we care about, specified by minra/maxra/mindec/maxdec passed '
              'to run().  One is the center.  The other eight are in a rectangle around the center; '
              'corner_distance is the fraction of the distance from the center to the edge along the '
              'relevant direction.  If this is None, then only consider the center; in that case, pass '
              'only ra and dec to run().' ),
            critical=True,
        )

        self.overlap_fraction = self.add_par(
            'overlap_fraction',
            0.9,
            (None, float),
            ( "When looking for pre-existing references, only return ones whose are overlaps this "
              "fraction of the desired rectangle's area.  Must be None if corner distance is None." ),
            critical=True,
        )

        self.coadd_overlap_fraction = self.add_par(
            'coadd_overlap_fraction',
            0.1,
            (None, float),
            ( "When looking for images to coadd into a new reference, only consider images whose "
              "min/max ra/dec overlap the sky rectangle of the target by at least this much.  "
              "Ignored when corner_distance is None." ),
            critical=True,
        )

        self.instrument = self.add_par(
            'instrument',
            'DECam',
            str,
            'The instrument for which we are building a reference.',
            critical=True,
        )

        self.projects = self.add_par(
            'projects',
            None,
            (None, list),
            'Only use images from these projects. If None, will not limit the projects. '
            'If given as a list, will use any of the projects in the list. ',
            critical=True,
        )

        self.zp_prov_id = self.add_par(
            'zp_prov_id',
            'placeholder',
            str,
            'The provenance of the ZeroPoint for images to be coadded into the reference.',
            critical=True
        )

        self.__image_query_pars__ = ['airmass', 'background', 'seeing', 'lim_mag', 'exp_time']

        for name in self.__image_query_pars__:
            for min_max in ['min', 'max']:
                self.add_limit_parameter(name, min_max)

        self.__docstrings__['min_lim_mag'] = ('Only use images with lim_mag larger (fainter) than this. '
                                             'If None, will not limit the minimal lim_mag. ')
        self.__docstrings__['max_lim_mag'] = ('Only use images with lim_mag smaller (brighter) than this. '
                                                'If None, will not limit the maximal lim_mag. ')

        self.min_number = self.add_par(
            'min_number',
            1,
            int,
            ( 'Construct a reference only if there are at least this many images that pass all other criteria '
              'If corner_distance is not None, then this applies to all test positions on the image unless '
              'min_only_center is True.' ),
            critical=True,
        )

        self.min_only_center = self.add_par(
            'min_only_center',
            False,
            bool,
            ( 'If True, then min_number only applies to the center position of the target area.  Otherwise, '
              'every test position on the image must have at least min_number references for the construction '
              'not to fail.  Ignored if corner_distance is None.' ),
            critical=True,
        )

        self.max_number = self.add_par(
            'max_number',
            None,
            (None, int),
            'If there are more than this many images, pick the ones with the highest "quality". '
            'WARNING : currently not implemented',
            critical=True,
        )

        self.seeing_quality_factor = self.add_par(
            'seeing_quality_factor',
            3.0,
            float,
            'linear combination coefficient for adding limiting magnitude and seeing FWHM '
            'when calculating the "image quality" used to rank images. ',
            critical=True,
        )

        self.save_new_refs = self.add_par(
            'save_new_refs',
            True,
            bool,
            'If True, will save the coadd image and commit it and the newly created reference to the database. '
            'If False, will only return it. ',
            critical=False,
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)

    def add_limit_parameter(self, name, min_max='min'):
        """Add a parameter in a systematic way. """
        if min_max not in ['min', 'max']:
            raise ValueError('min_max must be either "min" or "max"')
        compare = 'larger' if min_max == 'min' else 'smaller'
        setattr(
            self,
            f'{min_max}_{name}',
            self.add_par(
                f'{min_max}_{name}',
                None,
                (None, float),
                f'Only use images with {name} {compare} than this value. '
                f'If None, will not limit the {min_max}imal {name}.',
                critical=True,
            )
        )

    def get_process_name(self):
        return 'referencing'


class RefMaker:
    def __init__(self, **kwargs):
        """Initialize a reference maker object.

        The possible keywords that can be given are: maker, pipeline,
        coaddition. Each should be a dictionary.  maker keys are defined
        by ParsRefMaker above.  pipeline keys are defined as described
        pipeline/top_level.py::Pipeline, and may have subdictionaries
        pipeline, preprocessing, extraction, sources, bg, wcs, and zp.
        coaddition keys are defined by
        pipeline/coaddition.py::ParsCoadd.

        Parameters are set by first looking at the referencing.pipeline,
        referencing.coaddition, and referncing.maker trees from the
        config file.  They are then overridden by anything passed to the
        constructor.

        The maker contains a Pipeline object, that doesn't do any work,
        but is instantiated so it can build up the provenances of the
        images and their products, that go into the coaddition.  Those
        images need to already exist in the database before calling
        run().  Pass kwargs into the pipeline object using
        kwargs['pipeline'].

        The maker also contains a coadd_pipeline object, that has two
        roles: one is to build the provenances of the coadd image and
        the products of that image (extraction on the coadd) and the
        second is to actually do the work of coadding the chosen images.
        Pass kwargs into this object using kwargs['coaddition'].

        The choice of which images are loaded into the reference coadd
        is determined by the parameters object of the maker itself (and
        the provenances of the images and their products).  To set these
        parameters, use the "referencing.maker" dictionary in the
        config, or pass them in kwargs['maker'].

        """
        # first break off some pieces of the kwargs dict
        maker_overrides = kwargs.pop('maker', {})  # to set the parameters of the reference maker itself
        coadd_overrides = kwargs.pop('coaddition', {})  # to allow overriding the coaddition pipeline

        if len(kwargs) > 0:
            raise ValueError(f'Unknown parameters given to RefMaker: {kwargs.keys()}')

        # now read the config file
        config = Config.get()

        coadd_dict = config.value('referencing.coaddition', {})  # allow overrides from config's referencing.coaddition
        coadd_dict.update(coadd_overrides)  # allow overrides from kwargs['coaddition']
        self.coadd_pipeline = CoaddPipeline(**coadd_dict)  # coaddition parameters, overrides with coadd_dict

        maker_dict = config.value('referencing.maker')
        maker_dict.update(maker_overrides)  # user can provide override arguments in kwargs
        self.pars = ParsRefMaker(**maker_dict)  # initialize without the pipeline/coaddition parameters

        if ( self.pars.corner_distance is None ) != ( self.pars.overlap_fraction is None ):
            raise ValueError( "Configuration error; for RefMaker, must have a float for both of "
                              "corner_distance and overlap_fraction, or both must be None." )

        self.coadd_im_prov = None
        self.coadd_ex_prov = None
        self.coadd_wcs_prov = None
        self.coadd_zp_prov = None
        self.ref_prov = None
        self.refset = None

        self.reset()

    # ======================================================================

    def reset( self ):
        # these attributes tell us the place in the sky (in degrees)
        # where we want to look for objects (given to run()), # and the
        # filter we want to be in.  Optionally, it can also specify a
        # target and section_id to limit images to.

        self.minra = None
        self.maxra = None
        self.mindec = None
        self.maxdec = None
        self.target = None
        self.ra = None  # in degrees
        self.dec = None  # in degrees
        self.target = None  # the name of the target / field ID / Object ID
        self.section_id = None  # a string with the section ID

    # ======================================================================

    def setup_provenances(self, session=None):
        """Make the provenances for the coadd image and all its products.

        The created provenances are loaded into the database.

        """

        zpprov = Provenance.get( self.pars.zp_prov_id )
        if zpprov is None:
            raise RuntimeError( f"Failed to find ZeroPoint provenance {self.pars.zp_prov_id}" )
        upstreams = [ Provenance.get( self.pars.zp_prov_id ) ]
        self.coadd_pipeline.datastore = DataStore()
        coadd_provs = self.coadd_pipeline.make_provenance_tree( None, upstream_provs=upstreams )
        self.coadd_im_prov = coadd_provs['starting_point']
        self.coadd_ex_prov = coadd_provs['extraction']
        self.coadd_wcs_prov = coadd_provs['astrocal']
        self.coadd_zp_prov = coadd_provs['photocal']

        pars = self.pars.get_critical_pars()
        self.ref_prov = Provenance(
            process=self.pars.get_process_name(),
            parameters=pars,
            upstreams=[ self.coadd_zp_prov ],
            is_testing='test_parameter' in pars,
        )
        self.ref_prov.insert_if_needed()


    # ======================================================================

    def make_refset(self, session=None):
        """Create or load an existing RefSet with the required name.

        Sets self.refset.  Will also make all the required provenances
        (using the config) and load them into the database.

        """
        with SmartSession( session ) as dbsession:
            # make sure all the sundry component provenances are in the database
            self.setup_provenances( session=dbsession )

            # make sure the ref_prov is in the database
            self.ref_prov.insert_if_needed( session=dbsession )

        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            # Lock the refset table so we don't have a race condition
            cursor.execute( "LOCK TABLE refsets" )
            # Check to see if the refset already exists
            cursor.execute( "SELECT * FROM refsets WHERE name=%(name)s", { 'name': self.pars.name } )
            rows = cursor.fetchall()
            if len(rows) > 0:
                # refset already exists, make sure the provenance is right
                if rows[0]['provenance_id'] != self.ref_prov.id:
                    raise ValueError( f"Refset {self.pars.name} already exists with provenance "
                                      f"{rows[0]['provenance_id']}, which does not match the "
                                      f"ref provenance we're using: {self.ref_prov.id}" )
                self.refset = RefSet()
                self.refset.set_attributes_from_dict( rows[0] )
            else:
                # refset doesn't exist, make it
                self.refset = RefSet( name=self.pars.name, description=self.pars.description,
                                      provenance_id=self.ref_prov.id )
                cursor.execute( "INSERT INTO refsets(_id,name,description,provenance_id) "
                                "VALUES (%(id)s,%(name)s,%(desc)s,%(prov)s)",
                                { 'id': self.refset.id, 'name': self.refset.name,
                                  'desc': self.refset.description, 'prov': self.refset.provenance_id } )
                conn.commit()


    # ======================================================================

    def parse_arguments( self, image=None, ra=None, dec=None,
                             minra=None, maxra=None, mindec=None, maxdec=None,
                             target=None, section_id=None,
                             filter=None ):
        """Parse arguments for the RefMaker.

        There are three modes in which RefMaker can operate:

        * If the corner_distance parameter is None, then we're making a
          reference that covers a single point (useful for forced
          photometry, for instance).  In this case, either specify an
          image (in which case its central ra and dec are used), or
          specify ra/dec.

        * If the corner_distance parameter is not None, we're making a
          reference that covers a rectangle on the sky (covering at
          least the overlap_fraction parameter of the rectangle).  In
          this case, either specify an image that defines the rectangle
          on the sky, or specify minra/maxra/mindec/maxdec.  The rectangle
          is aligned to NS/EW.

        Parameters
        ----------
          image: Image or None
            Used to get the ra/dec (if pars.corner_distance is None) or min/max ra/dec.

          ra, dec: float or None
            Position to search.   Only makes sense if pars.corner_distance is None

          minra, maxra, mindec, maxdec: float or None
            Area to search.  Only makes sense if pars.corner_disdtance is not None

          target, section_id: string or None
            Optionally, specify a target and section_id that images must
            have to be considered for inclusion in a reference.  Only
            use this if you're using a survey that's very careful about
            setting its target names, and if you always go back to
            exactly the same fields so you know that the same chip is
            always going to be in the same place.

          filter: string or None
            If given, only find images whose filter match this filter

        """
        if ( image is not None ) and any( i is not None for i in [ ra, dec, minra, maxra, mindec, maxdec ] ):
            raise ValueError( "If you pass image to RefMaker.run, you can't pass any coordinates." )

        if self.pars.corner_distance is None:
            if any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                raise ValueError( "For RefMaker corner_distance None, can't specify minra/maxra/mindec/maxdec" )
            if image is not None:
                if ( ra is not None ) or ( dec is not None ):
                    raise ValueError( "For RefMaker corner_distance None, must specify image or ra/dec, not both" )
                ra = image.ra
                dec = image.dec
            else:
                if ( ra is None ) or ( dec is None ):
                    raise ValueError( "For RefMaker corner_distance None, must provide either image or both ra & dec" )
        else:
            if ( ra is not None ) or ( dec is not None ):
                raise ValueError( "For RefMaker corner_distance not None, can't specify ra/dec" )
            if image is not None:
                if any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                    raise ValueError( "For RefMaker corner_distance not None, must specify image or "
                                      "minra/maxra/mindex/maxdec, not both" )
                minra = image.minra
                maxra = image.maxra
                mindec = image.mindec
                maxdec = image.maxdec
            else:
                if any ( i is None for i in [ minra, maxra, mindec, maxdec ] ):
                    raise ValueError( "For RefMaker corner_distance not None, must specify image or "
                                      "all of minra/maxra/mindec/maxdec" )

        self.minra = minra
        self.maxra = maxra
        self.mindec = mindec
        self.maxdec = maxdec
        self.ra = ra
        self.dec = dec
        self.target = target
        self.section_id = section_id
        self.filter = filter

    # ======================================================================

    def identify_reference_images_to_coadd( self, *args, _do_not_parse_arguments=False, **kwargs ):
        """Identify images in the database that could be used to build our reference.

        See parse_arguments for a description of the arguments.

        (Parameter _do_not_parse_arguments is used internally, ignore it
        if calling this from the outside.)

        Returns
        -------
           images, match_pos, match_count

           images: list of Image
             List of images that can be included in the sum.

           match_pos: 2d numpy array
             Each row is [ra,dec] of a position on the summed image.  If
             operating in (ra,dec) mode (rather than min/max ra/dec
             mode), this will be [[ra,dec]].

          match_count: list of int
             Number of images that overlap the corresponding match_pos.

        """
        if not _do_not_parse_arguments:
            self.parse_arguments( *args, **kwargs )

        if self.pars.corner_distance is None:
            match_pos = [ [ self.ra, self.dec ] ]
            match_count = [ 0 ]
            kwargs = { 'ra': self.ra, 'dec': self.dec }
        else:
            if ( self.maxra < self.minra ):
                dra = ( self.maxra + 360. - self.minra ) * self.pars.corner_distance/2.
                ctrra = ( self.maxra+360. + self.minra ) / 2.
                ctrra = ctrra if ctrra >= 0. else ctrra + 360.
            else:
                dra = ( self.maxra - self.minra ) * self.pars.corner_distance/2.
                ctrra = ( self.maxra + self.minra ) / 2.
            ddec = ( self.maxdec - self.mindec ) * self.pars.corner_distance/2.
            ctrdec = ( self.maxdec + self.mindec ) / 2.
            match_pos = np.array( [ [ ctrra + 0.,  ctrdec + 0. ],
                                    [ ctrra - dra, ctrdec - ddec ],
                                    [ ctrra + 0.,  ctrdec - ddec ],
                                    [ ctrra + dra, ctrdec - ddec ],
                                    [ ctrra - dra, ctrdec + 0. ],
                                    [ ctrra + dra, ctrdec + 0. ],
                                    [ ctrra - dra, ctrdec + ddec ],
                                    [ ctrra + 0.,  ctrdec + ddec ],
                                    [ ctrra + dra, ctrdec + ddec ] ] )
            match_count = [ 0 ] * 9
            kwargs = { 'minra': self.minra, 'maxra': self.maxra, 'mindec': self.mindec, 'maxdec': self.maxdec,
                       'overlapfrac': self.pars.coadd_overlap_fraction }

        kwargs['provenance_ids'] = [ self.pars.zp_prov_id ]
        kwargs['provenance_ids_are_zp'] = True
        kwargs['instrument' ] = self.pars.instrument
        kwargs['project'] = self.pars.projects
        kwargs['filter'] = self.filter
        kwargs['min_mjd'] = ( None if self.pars.start_time is None
                              else parse_dateobs( self.pars.start_time, output='mjd' ) )
        kwargs['max_mjd'] = None if self.pars.end_time is None else parse_dateobs( self.pars.end_time, output='mjd' )
        kwargs['max_seeing'] = self.pars.max_seeing
        kwargs['min_lim_mag'] = self.pars.min_lim_mag
        kwargs['min_exp_time'] = self.pars.min_exp_time
        # TODO : airmass, background

        possible = Image.find_images( **kwargs )

        existing = []
        for image in possible:
            keep = False
            for i, pos in enumerate(match_pos):
                if image.contains( pos[0], pos[1] ):
                    match_count[i] += 1
                    keep = True
            if keep:
                existing.append( image )

        return existing, match_pos, match_count


    # ======================================================================

    def run(self, *args, do_not_build=False, **kwargs ):
        """Look to see if there is an existing reference that matches the specs; if not, optionally build one.

        See parse_arguments for function call parameters.  The remaining
        policy for which images to pick, and what provenance to use to
        find references, is defined by the parameters object of self and
        self.pipeline.

        If do_not_build is true, this becomes a thin front-end for Reference.get_references().

        Will check if a RefSet exists with the same provenance and name, and if it doesn't, will create a new
        RefSet with these properties, to keep track of the reference provenances.

        Will return a Reference, or None in case it doesn't exist and cannot be created
        (e.g., because there are not enough images that pass the criteria).

        """

        self.parse_arguments( *args, **kwargs )

        self.make_refset()

        # look for the reference at the given location in the sky (via ra/dec or target/section_id)
        refsandimgs = Reference.get_references(
            ra=self.ra,
            dec=self.dec,
            target=self.target,
            section_id=self.section_id,
            filter=self.filter,
            provenance_ids=self.ref_prov.id,
        )

        refs, _ = refsandimgs

        # if found a reference, can skip the next part of the code!
        if len(refs) == 1:
            return refs[0]
        elif len(refs) > 1:
            raise RuntimeError( f'Found multiple references with the same provenance '
                                f'{self.ref_prov.id} and location!' )

        if do_not_build:
            return None

        ############### no reference found, need to build one! ################

        images, _, match_count = self.identify_reference_images_to_coadd( _do_not_parse_arguments=True )

        # Make sure we got enough

        if len(images) < self.pars.min_number:
            SCLogger.info( f"RefMaker only found {len(images)} images overlapping the desired field, "
                           f"which is less than the minimum of {self.pars.min_number}" )
            return None
        # match_count[0] is always for the center position
        if ( self.pars.min_only_center ) and ( match_count[0] < self.pars.min_number ):
            SCLogger.info( f"RekMaker only found {len(match_count[0])} images overlapping the center of the "
                           f"desired field, which is less than the minimum of {self.pars.min_number}" )
            return None
        elif ( not self.pars.min_only_center ) and any( c < self.pars.min_number for c in match_count ):
            SCLogger.info( f"RefMaker didn't find enough references at at least one point on the image; "
                           f"match_count={match_count}, min_number={self.pars.min_number}" )
            return None


        # Sort the images and create data stores for all of them
        # Have to pull out all the zeropoint upstream provenances
        #   so the DataStore can find its stuff.

        images = sorted(images, key=lambda x: x.mjd)  # sort the images in chronological order for coaddition
        dses = []
        improv = Provenance.get( images[0].provenance_id )
        zpprov = Provenance.get( self.pars.zp_prov_id )
        if ( len(zpprov.upstreams) != 1 ) or ( zpprov.upstreams[0].process != 'astrocal' ):
            raise RuntimeError( "I don't know how to cope" )
        wcsprov = zpprov.upstreams[0]
        if ( len(wcsprov.upstreams) != 1 ) or ( wcsprov.upstreams[0].process != 'extraction' ):
            raise RuntimeError( "I don't know how to cope" )
        srcprov = wcsprov.upstreams[0]
        if ( len(srcprov.upstreams) != 1 ) or ( srcprov.upstreams[0].id != improv.id ):
            raise RuntimeError( "I don't know how to cope" )
        provtree = ProvenanceTree( { p.process: p for p in [ improv, srcprov, wcsprov, zpprov ] },
                                   upstream_steps={ improv.process: [ 'starting_point' ],
                                                    srcprov.process: [ improv.process ],
                                                    wcsprov.process: [ srcprov.process ],
                                                    zpprov.process: [ wcsprov.process ] } )
        for im in images:
            inst = im.instrument
            if inst != self.pars.instrument:
                raise RuntimeError( f"RefMaker for instrument {self.pars.instrument} got an "
                                    f"image from {inst}" )
            if im.provenance_id != improv.id:
                raise RuntimeError( "This should never happen." )
            ds = DataStore( im )
            ds.edit_prov_tree( provtree )
            ds.sources = ds.get_sources()
            ds.bg = ds.get_background()
            ds.psf = ds.get_psf()
            ds.wcs = ds.get_wcs()
            ds.zp = ds.get_zp()
            prods = {p: getattr(ds, p) for p in ['sources', 'psf', 'bg', 'wcs', 'zp']}
            if any( [p is None for p in prods.values()] ):
                raise RuntimeError(
                    f'DataStore for image {im} is missing some of products {prods} for coaddition! '
                    f'Make sure to produce products using the provenances in ex_provs: '
                    f'{self.ex_provs}'
                )
            dses.append( ds )

        self.coadd_pipeline.make_provenance_tree( dses )
        coadd_ds = self.coadd_pipeline.run( dses )

        ref = Reference(
            zp_id = coadd_ds.zp.id,
            provenance_id = self.ref_prov.id
        )

        if self.pars.save_new_refs:
            coadd_ds.save_and_commit()
            ref.insert()

        return ref
