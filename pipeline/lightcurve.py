import sys
import argparse
import datetime
import numbers
import uuid
import textwrap

import numpy as np
from psycopg import sql

import improc.photometry
import models.object
from models.base import PGDB
from models.provenance import Provenance
from models.object import ObjectPosition
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.refset import RefSet
from models.diaforcedphot import DiaForcedPhot
from util.config import Config, NoValue
from util.logger import SCLogger
from util.util import listify
from pipeline.parameters import Parameters
from pipeline.data_store import ProvenanceTree, DataStore
from pipeline.subtraction import Subtractor
import util.util


class ParsLightcurve(Parameters):
    def __init__( self, **kwargs ):
        super().__init__()

        self.zp_prov = self.add_par(
            name = 'zp_prov',
            default = None,
            par_types = ( str, None ),
            docstring = ( "Provenance of the zeropoint to use when searching for images to build into the "
                          "lightcurve.  Pass either this or zp_prov_tag; if you pass both, zp_prov_tag is "
                          "ignored." ),
            # Strictly speaking, this doesn't need to be critical, because the zeropoint provenance
            #   will be in this process' provenance's upstreams.  But!  We need it here, because we don't
            #   only want Provenance.parameters to be enough to uniquely specify the provenance (together
            #   with process, code_version, and upsterams), but also we want to be able to use
            #   Provenance.parmaeters to instantiate a pipeline worker object.
            critical = True
        )

        self.zp_prov_tag = self.add_par(
            name = 'zp_prov_tag',
            default = None,
            par_types = ( str, None ),
            docstring = ( "Provenance tag for the zeropoint to use when searching for images to build into the "
                          "lightcurve.  Ignored if zp_prov is given.  You must include one of the two." ),
            critical = False
        )

        self.zp_prov_tag_process = self.add_par(
            name = 'zp_prov_tag_process',
            default = 'photocal',
            par_types = str,
            docstring = "The process to use when searching provenance tags for the zeropoint provenance.",
            critical = False
        )

        self.object_position_prov = self.add_par(
            name = 'object_position_prov',
            default = None,
            par_types = ( str, None ),
            docstring = ( "Provenance if the object position to use for finding the object's position.  If neither "
                          "this nor object_position_prov_tag is given, will use the raw position from the object." ),
            critical = True
        )

        self.object_position_prov_tag = self.add_par(
            name = 'object_position_prov_tag',
            default = None,
            par_types = ( str, None ),
            docstring = ( "Provenance tag for object positions to use for finding the object's position.  "
                          "Ignored if object_positon_prov is given." ),
            critical=False
        )

        self.object_position_prov_tag_process = self.add_par(
            name = 'object_position_prov_tag_process',
            default = 'positioning',
            par_types = ( str, None ),
            docstring = "The process to use when searching provenance tags for object position provenance.",
            critical = False
        )

        self.only_existing_subtractions = self.add_par(
            name = "only_existing_subtractions",
            default = False,
            par_types = bool,
            docstring = ( "Don't do any new subtractions, only look at existing subtractions to do forced "
                          "photometry on." ),
            critical = False
        )

        # On to actual critical parameters

        self.subtraction_config = self.add_par(
            name = "subtraction_config",
            default = {
                'method': 'zogy',
                'refset': None,
                'alignment_index': 'new',
                'alignment': { 'method': 'swarp' },
                'reference': { 'search_by': 'ra/dec',
                               'match_instrument': True,
                               'match_filter': True,
                               'min_overlap': None,
                               'max_dist': 30. / 3600.,
                               'skip_bad': True,
                               'multiple_ok': False,
                               'choice_criteria': [ 'distance', 'unconstrained' ],
                              }
            },
            par_types = dict,
            docstring = ( "A dictionary with subtraction config.  Will override what's in config files "
                          "and defaults for subtraction." ),
            # Not critical because a subtraction provenance will be in the upstreams of the dia forced
            #   phot provenance
            critical = False
        )

        self.crop_image = self.add_par(
            name = "crop_image",
            default = [ 150, 150 ],
            par_types = ( list, None ),
            docstring = ( "If given, 2-element list (width, height).  Science images will be trimmed to at most "
                          "this size before being fed to subtractions.  If None, use full-size images "
                          "(which is *usually* not what you want)." ),
            critical=True
        )

        # Finally, non-critical parameters that say what to actually do

        self.mjd0 = self.add_par(
            name = "mjd0",
            default = None,
            par_types = ( float, None ),
            docstring = "The earliest mjd to do forced photometry for",
            critical = False
        )

        self.mjd1 = self.add_par(
            name = "mjd1",
            default = None,
            par_types = ( float, None ),
            docstring = "The latest mjd to do forced photometry for",
            critical = False
        )

        self.instrument = self.add_par(
            name = "instrument",
            default = None,
            par_types = ( str, None ),
            docstring = "The instrument that we're building a lightcurve for.  Only do one instrument at a time.",
            # Not critical because the dia forced phot upstream provs will have an image prov that (effectively)
            #   specifies instrument
            critical = False
        )

        self.filters = self.add_par(
            name = "filters",
            default = None,
            par_types = ( list, None ),
            docstring = "Only do forced photometry for these filters (all filters found if not given).",
            critical = False
        )

        self.object_id = self.add_par(
            name = "object_id",
            default = None,
            par_types = ( uuid.UUID, str, None ),
            docstring = "The id of the object to build a lightcurve for.  Specify either this or object_name.",
            critical = False
        )

        self.object_name = self.add_par(
            name = "object_name",
            default = None,
            par_types = ( str, None ),
            docstring ="The name of the object to build a lightcurve for.  Ignored if object_id is given.",
            critical = False
        )

        self.save_to_db = self.add_par(
            name = "save_to_db",
            default = False,
            par_types = bool,
            docstring = ( "Set True to save newly created photometry to the database.  In *any* event, "
                          "subtractions and trimmed images and so forth are going to get saved.  I think." ),
            critical = False
        )

        self.numprocs = self.add_par(
            name = "numprocs",
            default = 1,
            par_types = int,
            docstring = ( "Number of processes to run.  All file identification (news, refs) and ref reading is "
                          "done in the parent process.  It then forks nprocs subprocesses to do the subtractions "
                          "and photometry.  If nprocs=1, it's all done serially." ),
            critical=False
        )

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name( self ):
        return 'lightcurve'


class Lightcurve:
    def __init__( self, **kwargs ):
        """Do forced photometry on difference images."""

        cfg = Config.get()

        self.pars = ParsLightcurve()

        # We want to start with the subtraction config
        # Let this class' defaults override that
        # Let the lightcurve config override that
        # Let kwargs override that.
        self.pars.merge_configs( before=[ {'subtraction_config': cfg.value('subtraction')} ],
                                 after=[ cfg.value('lightcurve'), kwargs ] )

        self.subtractor = Subtractor( **(self.pars.subtraction_config) )

        self.object = None
        self.object_position_prov = None
        self.object_position = None
        self.ra = None
        self.dec = None
        self.zp_prov = None
        self.refset = None
        self.refs = {}
        self.provtree = None
        self.imgs = None
        self.wcsen = None
        self.zps = None
        self.dia_forced_phots = None

        self._has_been_setup = False


    def setup( self, object_id=NoValue(), object_name=NoValue(), mjd0=NoValue(),
               mjd1=NoValue(), filters=NoValue(), pgdb=None ):
        if self._has_been_setup:
            # Doing this because we set some parameters below.  If we wanted to make
            #   this reusable, we'd have to be able to restore them.
            #   (Honestly, that may be true of lots of other objects as well -- they may
            #   set parameters when they're run, not only when they're instantiated, which
            #   means there's a hysteresis.  Should check.)
            raise RuntimeError( "Lightcurve is a single-use object." )
        self._has_been_setup = True

        pgdb_in = pgdb

        self.pars.object_id = object_id if not isinstance( object_id, NoValue ) else self.pars.object_id
        self.pars.object_name = object_name if not isinstance( object_name, NoValue ) else self.pars.object_name
        self.pars.mjd0 = mjd0 if not isinstance( mjd0, NoValue ) else self.pars.mjd0
        self.pars.mjd1 = mjd1 if not isinstance( mjd1, NoValue ) else self.pars.mjd1
        self.pars.filters = listify(filters) if not isinstance( filters, NoValue ) else self.pars.filters

        self.object = None
        if self.pars.object_id is not None:
            objcol = "_id"
            objval = self.pars.object_id
            if self.pars.object_name is not None:
                SCLogger.warning( "Gave both object_id and object_name, ignoring object_name" )
        elif self.pars.object_name is not None:
            objcol = "name"
            objval = self.pars.object_name
        else:
            raise ValueError( "Must give either object_id or object_name" )

        if self.pars.crop_image is not None:
            if ( ( len(self.pars.crop_image) != 2 ) or
                 ( not all ( isinstance(x, numbers.Integral) for x in self.pars.crop_image ) ) ):
                raise ValueError( f"Must give two integer values for crop_image, got {self.pars.crop_image}" )

        with PGDB( pgdb_in, dictcursor=True ) as pgdb:
            if ( self.zp_prov is None ) or ( self.zp_prov.id != self.pars.zp_prov ):
                self.zp_prov = None
                if self.pars.zp_prov is not None:
                    self.zp_prov = Provenance.get( self.pars.zp_prov, pgdb=pgdb )
                elif self.pars.zp_prov_tag is not None:
                    self.zp_prov = Provenance.get_for_tag( self.pars.zp_prov_tag, self.pars.zp_prov_tag_process,
                                                           pgdb=pgdb )
                    self.pars.zp_prov = self.zp_prov.id

            if self.zp_prov is None:
                raise RuntimeError( f"Could not find a zeropoint provenance to use to find images. "
                                    f"zp_prov={self.pars.zp_prov}, zp_prov_tag={self.pars.zp_prov_tag}, "
                                    f"zp_prov_tag_process={self.pars.zp_prov_tag_process}" )

            if ( self.pars.object_position_prov is None ) and ( self.pars.object_position_prov_tag is None ):
                self.object_position_prov = None
            else:
                if self.pars.object_position_prov is not None:
                    if self.pars.object_position_prov_tag is None:
                        SCLogger.warning( "Both object_position_prov and object_position_prov_tag given; "
                                          "ignoring the latter." )
                    if ( ( self.object_position_prov is None ) or
                         ( self.object_position_prov.id != self.pars.object_position_prov )
                        ):
                        self.object_position_prov = Provenance.get( self.pars.object_position_prov, pgdb=pgdb )
                        if self.object_position_prov is None:
                            raise ValueError( f"Could not find object position provenance "
                                              f"{self.pars.object_position_prov}" )
                else:
                    self.object_position_prov = Provenance.get_by_tag( self.pars.object_position_prov_tag,
                                                                       self.pars.object_position_prov_tag_process,
                                                                       pgdb=pgdb_in )
                    if self.object_position_prov is None:
                        raise ValueError( f"Could not find object position provenance for "
                                          f"tag {self.pars.object_position_prov_tag} and "
                                          f"process { self.pars.object_position_prov_tag_process}" )
                    self.pars.object_position_prov = self.object_position_prov.id

            if self.subtractor.pars.refset is None:
                raise ValueError( "Subtractor has no refset defined!" )
            self.refset = RefSet.get_by_name( self.subtractor.pars.refset, pgdb=pgdb )
            if self.refset is None:
                raise ValueError( f"Can't find refset {self.subtractor.pars.refset}" )

            q = sql.SQL( "SELECT * FROM objects WHERE {col}={val}" ).format( col=sql.Identifier(objcol), val=objval )
            rows = pgdb.execute( q )
            if len(rows) > 1:
                raise RuntimeError( "This should never happen" )
            elif len(rows) == 0:
                raise ValueError( f"Could not find object with {objcol}={objval}" )
            else:
                self.object = models.object.Object( **(rows[0]) )

            self.object_position = None
            if self.object_position_prov is not None:
                rows = pgdb.execute( sql.SQL( "SELECT * FROM object_positions "
                                              "WHERE object_id={objid} AND provenance_id={provid} "
                                             ).format( objid=self.object.id,
                                                       provid=self.object_position_prov.id ) )
                if len(rows) > 1:
                    raise RuntimeError( "This should never happen, I don't think, but I'm not really sure." )
                elif len(rows) == 0:
                    raise ValueError( f"Could not find object position for object {self.object.id} "
                                      f"and object position provenacne {self.object_position_prov.id}" )
                else:
                    self.object_position = ObjectPosition( **(rows[0]) )

                self.ra = self.object_position.ra
                self.dec = self.object_position.dec
            else:
                self.ra = self.object.ra
                self.dec = self.object.dec


    def _generate_provenances( self, provtree, pgdb=None ):
        subups = [ provtree['referencing'] ]
        subupsteps = [ 'referencing' ]

        # Get trim image provenances
        if self.pars.crop_image is not None:
            trim_processes = [ 'Image.trim', 'Image.trim.sources', 'Image.trim.wcs', 'Image.trim.zp' ]
            trimupsteps = { 'Image.trim':         [ 'starting_point', 'astrocal' ],
                            'Image.trim.sources': [ 'Image.trim' ],
                            'Image.trim.wcs':     [ 'Image.trim.sources' ],
                            'Image.trim.zp':      [ 'photocal' ] }
            trimprovs = list( Image.get_trim_provs( self.pars.crop_image[0], self.pars.crop_image[1],
                                                    upstreams=[ provtree['starting_point'] ],
                                                    wcs_prov=provtree['astrocal'], zp_prov=provtree['photocal'],
                                                    save=False ) )
            if 'Image.trim' in provtree:
                if any( provtree[trim_processes[i]].id != trimprovs[i].id for i in range(4) ):
                    raise ValueError( "Pre-existing image trim provenances don't match what what "
                                      "they should have been given config." )
                if any( set( trimupsteps[trim_processes[i]] ) != set( provtree.upstream_steps[trim_processes[i]] )
                        for i in range(4) ):
                    raise ValueError( "Pre-existing trim upstream steps weren't what was expected." )
            else:
                # This next if should be False by construction.  If it's True, it
                #   means that there is a code error either here or in Image.get_trim_provs
                if ( len( trimprovs[0].upstreams ) != 2
                     or ( 'astrocal' not in [ p.process for p in trimprovs[0].upstreams ] )
                     or  any( set( u.process for u in trimprovs[i].upstreams )
                              != set( trimupsteps[trim_processes[i]] )
                              for i in (1, 2, 3) )
                    ):
                    raise RuntimeError( "I am surprised." )
            # Gotta include both Image.trim.wcs and
            # Image.trim.zp because Image.trim.zp has only
            # photocal as an upstream, so we don't tag the image
            # provenance by just tagging the Image.trim.zp
            # provenance.
            subups.extend( [ trimprovs[2], trimprovs[3] ] )
            subupsteps.extend( [ 'Image.trim.wcs', 'Image.trim.zp' ] )

        else:
            trimprovs = None
            subups.append( provtree['photocal'] )
            subupsteps.append( 'photocal' )

        # Get subtraction provenance
        subprov = Provenance( code_version_id=Provenance.get_code_version('subtraction', pgdb=pgdb).id,
                              process='subtraction',
                              parameters=self.subtractor.pars.get_critical_pars(),
                              upstreams=subups )
        if 'subtraction' in provtree:
            if subprov.id != provtree['subtraction'].id:
                raise ValueError( f"Found provenance for subtraction {provtree['subtraction'].id} does not "
                                  f"match what this pipeline will create {subprov.id}" )
            if set( subupsteps ) != set( provtree.upstream_steps['subtraction'] ):
                raise ValueError( "Subtraction upstream steps mismatch." )

        # Get the dia forced photometry provenance
        ups = [ subprov ]
        upsteps = [ 'subtraction' ]
        if self.object_position_prov is not None:
            ups.append( provtree['positioning'] )
            upsteps.append( 'positioning' )
        diaforcedprov = Provenance( code_version_id=Provenance.get_code_version('diaforcedphot', pgdb=pgdb).id,
                                    process='diaforcedphot', parameters=self.pars.get_critical_pars(),
                                    upstreams=ups )
        if 'diaforcedphot' in provtree:
            if diaforcedprov.id != provtree['diaforcedphot'].id:
                raise ValueError( f"Found provenance for dia forced photometry {provtree['diaforcedphot'].id} does not "
                                  f"match what this pipeline will create {diaforcedprov.id}" )
            if set( upsteps ) != set( provtree.upstream_steps['diaforcedphot'] ):
                raise ValueError( "Dia forced photometry upstream steps mismatch." )

        return diaforcedprov, subprov, trimprovs


    def make_prov_tree( self, just_read=False, save=True, save_tag=True, provtag=None, pgdb=None,
                        ok_if_preexisting_prov_tag_without_forcedphot=False ):
        """Make a provenance tree for the Lightcurve.

        Datastore.make_prov_tree is designed specifically for use with
        top_level, and is not easy to use here.  Probably that code
        should be moved to top_level.py.

        Parameters
        ----------
           just_read : bool, default False
              Normally, the provenance tree will be generated looking at
              the parameters attached to the Lightcurve object, and
              attached to the Subtractor object that the Lightcurve
              object makes.  Provenances will be generated for all of
              diaforcedphot, subtraction, Image.trim,
              Image.trim.sources, Image.trm.wcs, and Image.trim.zp.  If
              just_read is true, all of that is thrown out, and instead
              the provenances are read from the database using provtag.
              WARNING: if you do this, then don't generate new dia
              forced photometry, just read what's there!

           save : bool, default True
              Save any generated provenances to the database?  Must be
              False if just_read is True.

           provtag : str, default None
              The provenance tag to use to find existing subtraction and
              diaforcedphot provenances.  If save and save_tags are both
              true, than any newly generated provenacnes will be tagged
              with this provenacne tag.  If just_read is False, then if
              preexisting provenances in the database with this tag are
              inconsistent with the ones generated using the object's
              configured parameters, an exception will be raised.

           save_tag : bool, default True
              Ignored if save is False.  If True, then all provenances
              saved to the database are also tagged with the provenance
              tag given in provtag.

           pgdb : base.PGDB, default None
              A database connection.  If not given, one will be created
              and closed as necesary.

        Returns
        -------
          data_store.ProvenanceTree

        """

        if just_read and save:
            raise ValueError( "Can't use just_read and save together." )

        if just_read and ( provtag is None ):
            raise ValueError( "just_read requires provtag" )

        if save_tag and ( not save ):
            SCLogger.warning( "save_tag is True but save is False, ignoring save_tag." )

        # Build a full provenance tree for DataStore to chew on
        # DataStore.make_prov_tree is designed for use with top_level, and is
        #   not easy to use here, so just make one manually.

        pgdb_in = pgdb

        # Read any existing provenances from the databaes.  First make some sets
        #   of what we must have to do anything, and what is allowed.

        must_have_procs = { 'starting_point', 'extraction', 'astrocal', 'photocal', 'referencing' }
        all_procs = must_have_procs.union( { 'subtraction', 'diaforcedphot' } )
        trim_procs = [ 'Image.trim', 'Image.trim.sources', 'Image.trim.wcs', 'Image.trim.zp']

        provtree = ProvenanceTree( noupstreams=['positioning', 'referencing', 'starting_point'],
                                   processmap={'preprocessing': 'starting_point',
                                               'test_image': 'starting_point'} )
        with PGDB( pgdb_in ) as pgdb:
            # First, see if we can find the dia forced photometry tag
            if provtag is not None:
                found_prov = Provenance.get_for_tag( provtag, 'diaforcedphot', pgdb=pgdb )
            else:
                found_prov = None

            if found_prov is not None:
                # This will build the whole tree, adding all the upstreams
                provtree.append_provenance( found_prov, pgdb=pgdb )
                if not just_read:
                    # If we're not just reading, then we know which optional processes should be there
                    if self.pars.crop_image is not None:
                        all_procs = all_procs.union( trim_procs )
                    if ( ( self.pars.object_position_prov is not None ) or
                         ( self.pars.object_position_prov_tag is not None )
                        ):
                        all_procs.add( 'positioning' )
                    must_have_procs = all_procs
                else:
                    # Generate expected provenances for later validation
                    diaforcedprov, subprov, trimprovs = self._generate_provenances( provtree )

            else:
                if just_read:
                    raise RuntimeError( "just_read is true, but could not find dia forced photometry provenance for "
                                        "provenance tag {provtag}" )

                # Based on config, we know what processes are legal
                if self.pars.crop_image is not None:
                    all_procs = all_procs.union( trim_procs )
                if ( ( self.pars.object_position_prov is not None ) or
                     ( self.pars.object_position_prov_tag is not None )
                    ):
                    all_procs.add( 'positioning' )

                # OK... didn't find an existing diaforcedphot provenance so try to read as much as we can
                # First, there's *gotta* be a reference provenances, or we won't be able to do anything
                # (Use Provenance.get_by_id here rather than self.refset.provenance property, so that
                # we can use pgdb.)
                refprov = Provenance.get( self.refset.provenance_id, pgdb=pgdb )
                if refprov is None:
                    raise RuntimeError( f"Failed to find provenance {self.refset.provenance_id} for "
                                        f"refset {self.refset.name}" )
                provtree.append_provenance( refprov, pgdb=pgdb )

                # Likewise, there must be a zeropoint provenance
                if self.pars.zp_prov is not None:
                    zpprov = Provenance.get( self.pars.zp_prov, pgdb=pgdb )
                elif self.pars.zp_prov_tag is not None:
                    zpprov = Provenance.get_for_tag( self.pars.zp_prov_tag, 'photocal', pgdb=pgdb )
                else:
                    raise ValueError( "Must have one of zp_prov or zp_prov_tag" )
                if zpprov is None:
                    raise RuntimeError( f"Failed to find the zeropoint provenance for "
                                        f"zp_prov={self.pars.zp_prov} and zp_prov_tag={self.pars.zp_prov_tag}" )
                elif zpprov.process != 'photocal':
                    raise RuntimeError( f"zeropoint provenance process is {zpprov.process}, "
                                        f"expected 'photocal'" )

                # This will also add the astrocal, soruces, and preprocessing (starting_point) provenances
                provtree.append_provenance( zpprov, pgdb=pgdb )

                # Get the object position provenance if any
                posprov = None
                notfound = False
                if self.pars.object_position_prov is not None:
                    posprov = Provenance.get( self.pars.object_position_prov, pgdb=pgdb )
                    if posprov is None:
                        notfound = True
                    elif posprov.process != 'positioning':
                        raise ValueError( f"The process of provenance {self.pars.object_positon_prov} is "
                                          f"{posprov.process}, but should be 'positioning'." )
                elif self.pars.object_position_prov_tag is not None:
                    posprov = Provenance.get_for_tag( self.pars.object_position_prov_tag, 'positioning', pgdb=pgdb )
                    notfound = posprov is None
                if notfound:
                    raise RuntimeError( f"Failed to find object positioning provenance given "
                                        f"object_position_prov={self.pars.object_position_prov} and "
                                        f"object_position_prov_tag={self.pars.object_position_prov_tag}" )
                if posprov is not None:
                    provtree.append_provenance( posprov, pgdb=pgdb )

                if provtag is not None:
                    # Get the trim provenances if any
                    found_trimprovs = []
                    for proc in [ 'Image.trim', 'Image.trim.sources', 'Image.trim.wcs', 'Image.trim.zp' ]:
                        found_trimprovs.append( Provenance.get_for_tag( provtag, proc, pgd=pgdb ) )
                    if any( i is not None for i in found_trimprovs ):
                        if not all( i is not None for i in found_trimprovs ):
                            raise RuntimeError( f"Database corruption: found some, but not all, image trim provs "
                                                f"in provtag {provtag}" )
                        for prov in found_trimprovs:
                            provtree.append_provenance( prov, pgdb=pgdb )

                    # Get the subtraction provenance if any
                    found_subprov = Provenance.get_for_tag( provtag, 'subtraction', pgdb=pgdb )
                    if found_subprov is not None:
                        # Add the subtraction provenacne and its upstreams.  Most (all?) of the upstreams
                        #   will have already been added above, but _append_provs (supposedly) handles that.
                        provtree.append_provenance( found_subprov, pgdb=pgdb )

                # ...and we don't need to get the dia forced phot prov here because we wouldn't be
                #   inside this "else" if it could be found.

            # Make sure stuff we read out of the database has processes we expected
            have_procs = set( provtree.keys() )
            missing_procs = must_have_procs - have_procs
            unknown_procs = have_procs - all_procs
            if ( len(missing_procs) > 0 ) or ( len(unknown_procs) > 0 ):
                raise RuntimeError( f"Failure building provtree, unexpected processes.  "
                                    f"missing: {missing_procs} ; unknown: {unknown_procs}" )

            # If just_read is True, then we're done!
            if not just_read:
                # OK!  provtree now has all known provenances, including at *least* referencing and zeropoint
                # Make the provenances for the things this pipeline will create; if they were found in
                #   the database, make sure they match.

                diaforcedprov, subprov, trimprovs = self._generate_provenances( provtree, pgdb=pgdb )

                # Add the generated provenances to the provenance tree.  Do this piece by piece,
                #   so that self-consistency will be checeked.  (It does mean redundant database
                #   queries... I think.)
                if trimprovs[0] is not None:
                    for p in trimprovs:
                        provtree.append_provenance( p, nodb=True )
                    must_have_procs = must_have_procs.union( trim_procs )
                provtree.append_provenance( subprov, nodb=True )
                provtree.append_provenance( diaforcedprov, nodb=True )
                must_have_procs = must_have_procs.union( { 'subtraction', 'diaforcedphot' } )

                # So... the provenance three thinks it's self consistent.  Let's check again
                #    that the expected provenances are there, and they should ALL be there now.
                have_procs = set( provtree.keys() )
                missing_procs = must_have_procs - have_procs
                unknown_procs = have_procs - must_have_procs
                if ( len(missing_procs) > 0 ) or ( len(unknown_procs) > 0 ):
                    raise RuntimeError( f"Failure trawling the database for provenances, unexpected processes.  "
                                        f"missing: {missing_procs} ; unknown: {unknown_procs}" )


                # Save them if necessary.  Do this in the right order so upstreams exist.
                if save:
                    provs = []
                    if self.pars.crop_image is not None:
                        provs.extend( provtree[p] for p in [ 'Image.trim', 'Image.trim.sources',
                                                             'Image.trim.wcs', 'Image.trim.zp' ] )
                    provs.extend( [ provtree['subtraction'], provtree['diaforcedphot'] ] )
                    for prov in provs:
                        prov.insert_if_needed( pgdb=pgdb, nocommit=True )
                    pgdb.commit()
                    if provtag is not None:
                        Provenance.addtag( provtag, provs, pgdb=pgdb )

        return provtree


    def process_one_image( self, imgdex ):
        img = self.imgs[ imgdex ]
        ds = DataStore( img )
        ds.prov_tree = self.provtree
        ds.reference = self.refs[ img.filter ]
        with PGDB( dictcursor=True ) as pgdb:
            ds.sources = SourceList.get_by_id( self.wcsen[img.id].sources_id, pgdb=pgdb )
            rows = pgdb.execute( sql.SQL( "SELECT * FROM backgrounds WHERE sources_id={src}" )
                                 .format( src=ds.sources.id ) )
            ds.bg = Background.create( **(rows[0]) )
            rows = pgdb.execute( sql.SQL( "SELECT * FROM psfs WHERE sources_id={src}" )
                                 .format( src=ds.sources.id ) )
            ds.psf = PSF.create( **(rows[0]) )
        ds.wcs = self.wcsen[ img.id ]
        ds.zp = self.zps[ img.id ]

        # Trim if we have to
        if self.pars.crop_image is not None:
            xctr, yctr = ds.wcs.wcs.world_to_pixel_values( self.ra, self.dec )
            # RAGE.  I hate that these get returned as np.array(<value>), i.e. a 0-dimension array,
            #   rather than a scalar, because that confuses all kinds of code later.  So,
            #   convert them to floats.
            xctr = float( xctr )
            yctr = float( yctr )
            ixctr = int( np.floor( xctr + 0.5 ) )
            iyctr = int( np.floor( yctr + 0.5 ) )
            x0 = ixctr - ( self.pars.crop_image[0] // 2 )
            x1 = x0 + self.pars.crop_image[0]
            y0 = iyctr - ( self.pars.crop_image[1] // 2 )
            y1 = y0 + self.pars.crop_image[1]

            trimmed = img.trim( x0, x1, y0, y1, sources=ds.sources, bg=ds.bg, psf=ds.psf,
                                wcs=ds.wcs, zp=ds.zp, adjust_limits=True )

            cropprovs = trimmed['provenances']
            if ( ( cropprovs['image'].id != ds.prov_tree['Image.trim'].id ) or
                 ( cropprovs['sources'].id != ds.prov_tree['Image.trim.sources'].id ) or
                 ( cropprovs['wcs'].id != ds.prov_tree['Image.trim.wcs'].id ) or
                 ( cropprovs['zp'].id != ds.prov_tree['Image.trim.zp'].id ) ):
                raise ValueError( "Image trim provenances didn't match!  This should never happen." )

            x0, x1, y0, y1 = trimmed['limits']
            # Offset xctr and yctr so they're on the trimmed image
            xctr -= x0
            yctr -= y0

            cropds = DataStore( trimmed['image'] )
            cropds.prov_tree = ds.prov_tree
            cropds.reference = ds.reference
            cropds.sources = trimmed['sources']
            cropds.bg = trimmed['bg']
            cropds.psf = trimmed['psf']
            cropds.wcs = trimmed['wcs']
            cropds.zp = trimmed['zp']
            ds = cropds

        ds = self.subtractor.run( ds, ra=self.ra, dec=self.dec, trust_datastore_reference=True, do_not_load=False )
        sub_image = ds.get_sub_image()

        # Save to database if requested
        if self.pars.save_to_db:
            ds.save_and_commit( save_warped_ref=self.subtractor.pars.save_warped_ref, overwrite=False )

        # See if we can load pre-existing dia forced photometry from the database
        diaforcedphot = None
        with PGDB( dictcursor=True ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT * FROM dia_forced_photometry
                WHERE object_id={obj}
                  AND provenance_id={prov}
                  AND subtraction_id={sub}
                  AND object_position_id{posclause}
                """
            ) ).format( obj=self.object.id,
                        prov=ds.prov_tree['diaforcedphot'].id,
                        sub=sub_image.id,
                        posclause=( sql.SQL( "={posid}".format(posid=self.object_position.id) )
                                    if self.object_position is not None
                                    else sql.SQL( " IS NULL" ) ) )
            rows = pgdb.execute( q )
            if len(rows) > 1:
                raise RuntimeError( "Database corruption, dia forced phot multiply defined." )
            elif len(rows) == 1:
                diaforcedphot = DiaForcedPhot( **(rows[0]) )

        if diaforcedphot is None:
            # Now actually do photometry
            # First, make things the way photutils wants them
            sub_mask = np.full_like( sub_image.flags, False, dtype=bool )
            sub_mask[ sub_image.flags != 0 ] = True
            sub_mask[ sub_image.weight <= 0. ] = True
            sub_noise = 1. /np.sqrt( sub_image.weight )
            sub_noise[ sub_mask ] = np.nan

            new_zp = ds.get_zp()
            # TODO FIGURE THIS OUT (Issue #194)
            new_psf = ds.get_psf()

            measurements = improc.photometry.photometry( sub_image.data, sub_noise, sub_mask, positions=[(xctr, yctr)],
                                                         psfobj=new_psf, apers=new_zp.aper_cor_radii )
            measurements = measurements[0]

            diaforcedphot = DiaForcedPhot(
                object_id=self.object.id,
                object_position_id=None if self.object_position is None else self.object_position_id,
                provenance_id=ds.prov_tree['diaforcedphot'].id,
                subtraction_id=sub_image.id,
                x=xctr,
                y=yctr,
                flux_psf=measurements.flux_psf,
                flux_psf_err=measurements.flux_psf_err,
                flux_apertures=measurements.flux_apertures,
                flux_apertures_err=measurements.flux_apertures_err
            )

            if self.pars.save_to_db:
                diaforcedphot.insert()

        # For convenience for tests, stick the aperture corrections in
        # the diaforcedphot object.  The "right" way to do this is go
        # from subtraction_id to zp_id and get it there.
        diaforcedphot._aper_cors = ds.get_zp().aper_cors

        return diaforcedphot

    def write_csv_file( self, filepath ):
        raise NotImplementedError( "File writing not implemented." )


    def find_images( self ):
        if self.pars.filter is None:
            imgs, wcsen, zps = Image.find_images( ra=self.ra, dec=self.dec, type='Sci',
                                                  provenance_ids=self.provtree['photocal'].id,
                                                  provenance_ids_are_zp=True,
                                                  instrument=self.pars.instrument,
                                                  min_mjd=self.pars.mjd0, max_mjd=self.pars.mjd1,
                                                  order_by='earliest', return_wcs=True, return_zeropoints=True )
            filters = set( i.filter for i in imgs )
        else:
            imgs = []
            wcsen = {}
            zps = {}
            filters = self.pars.filter
            for filt in filters:
                thisimgs, thiswcsen, thiszps = Image.find_images( ra=self.ra, dec=self.dec, type='Sci',
                                                                  provenance_ids=self.provtree['photocal'].id,
                                                                  provenance_ids_are_zp=True,
                                                                  instrument=self.pars.instrument,
                                                                  min_mjd=self.pars.mjd0, max_mjd=self.pars.mjd1,
                                                                  filter=filt,
                                                                  order_by='earliest',
                                                                  return_wcs=True, return_zeropoints=True )
                imgs.extend( thisimgs )
                wcsen.update( thiswcsen )
                zps.update( thiszps )
            imgs.sort( key=lambda x: x.mjd )

        self.imgs = imgs
        self.wcsen = wcsen
        self.zps = zps
        self.filters = filters
        self.dia_forced_phots = [ None ] * len(imgs)

    def find_refs( self, pgdb=None ):
        # Make an empty datastore to do use for finding references.  (Issue #550)
        ds = DataStore()
        ds.prov_tree = self.provtree

        if self.object_position is not None:
            ra = self.object_position.ra
            dec = self.object_position.dec
        else:
            ra = self.object.ra
            dec = self.object.dec

        kwargs = self.subtractor.pars.reference.copy()
        kwargs['instrument'] = self.pars.instrument
        kwargs['provenances'] = self.refset.provenance_id
        kwargs['ra'] = ra
        kwargs['dec'] = dec
        kwargs['mjd0'] = self.imgs[0].mjd
        kwargs['mjd1'] = self.imgs[-1].mjd

        if any( x in kwargs for x in ( 'must_match_section', 'must_match_target' ) ):
            raise ValueError( "Don't use must_match_section or must_match_target in subtraction_conifg['reference']" )

        refs = {}
        with PGDB( pgdb ) as pgdb:
            for filt in self.filters:
                ref = ds.get_reference( filter=filt, pgdb=pgdb, **kwargs )
                if ref is None:
                    raise RuntimeError( f"Cannot find a reference at ({ra:.4f}, {dec:.4f}) for instrument "
                                        f"{self.pars.instrument}, filter {filt}, and parameters {kwargs}" )
                refs[filt] = ref

        oks = [ f for f, r in self.refs.items() if r is not None ]
        missings = [ f for f, r in self.refs.items() if r is None ]
        if len(missings) > 0:
            SCLogger.error( f"Failed to find refs for filters {missings}; did find refs for {oks}" )
            raise RuntimeError( "Some refs missing filters." )
        else:
            SCLogger.info( f"Found refs for all filters: {oks}" )

        self.refs = refs

    def load( self, *args, **kwargs ):
        self.setup( *args, **kwargs )
        self.provtree = self.make_prov_tree( save=False )
        self.find_images()
        self.find_refs()

        self.dia_forced_phots = []
        with PGDB() as pgdb:
            q = sql.SQL( "SELECT f.* FROM dia_forced_photometry f "
                         "INNER JOIN images i ON i._id=f.subtracton_id "
                         "WHERE object_id={obj} "
                         "AND provenance_id={prov} "
                         "AND object_position_id{poscaluse}"
                         "ORDER BY i.mjd"
                        ).format( obj=self.object.id,
                                  provid=self.provtree['diaforcedphot'].id,
                                  posclause=( sql.SQL( "={posid}".format(posid=self.object_position.id) )
                                              if self.object_position is not None
                                              else sql.SQL( " IS NULL" ) ) )
            rows = pgdb.execute( q )

        for row in rows:
            self.dia_forced_phots.append( **row )

        SCLogger.info( "Loaded dia forced photometry for {len(self.dia_forced_phots)} out of {len(self.mgs)}" )


    def export_image_mess( self, namebase="phot_" ):
        for phot in self.dia_forced_phots:
            with PGDB() as pgdb:
                subim = Image.get_by_id( phot.subtraction_id, pgdb=pgdb )
                _q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT i.* FROM image_subtraction_components isc
                    INNER JOIN zero_points z ON isc.new_zp_id=z._id
                    INNER JOIN world_coordinates w ON z.wcs_id=w._id
                    INNER JOIN source_lists s """
                ) ).format( ROB="YOU WERE HERE" )
                raise RuntimeError(subim)


    def run( self, *args, die_on_fail=False, **kwargs ):
        """Do dia forced photometry based on the object configuration.

        Parameters
        ----------
          object_id : str or uuid
          object_name : str
          mjd0 : float
          mjd1 : float
          filters : list of str
             All of these can override their corresponding parameters
             that were set when the Lightcurve object was instantiated.
             This will change what is in those parameters, so if you
             call the run() method more than once on the same Lightcurve
             object (which is in general a scary thing to do), don't
             count on them having reverted to what you constructed
             the Lightcurve object with!

          pgdb: PGDB, default None
             Database connection.  Connections will be opened and closed
             as needed if this is None.

        Returns
        -------
          List of DiaForcedPhot

          That list is also in self.dia_forced_phots

        """


        self.setup( *args, **kwargs )
        self.provtree = self.make_prov_tree( save=True )

        SCLogger.info( "Lightcurve finding images." )
        self.find_images()

        if len(self.imgs) == 0:
            SCLogger.warning( "No images found to build a lightcurve for!" )
            return None
        SCLogger.info( f"lightcurve found {len(self.imgs)} images" )

        SCLogger.info( f"Lightcurve finding refs for {len(self.filters)} filters." )
        self.find_refs()

        SCLogger.info( f"Lightcurve doing dia forced photometry on "
                       f"{len([ i for i in self.imgs if i.filter in self.refs ])} images." )
        for i in range( len(self.imgs) ):
            try:
                self.dia_forced_phots[i] = self.process_one_image(i)
            except Exception as ex:
                if die_on_fail:
                    raise
                else:
                    SCLogger.exception( f"Exception on image {self.imgs[i].filepath}: {ex}; moving on." )

        SCLogger.info( "Lightcurve complete" )
        return self.dia_forced_phots


# ======================================================================

def main():
    sys.stderr.write( f"lightcurve starting at {datetime.datetime.now(tz=datetime.UTC).isoformat()}\n" )

    parser = argparse.ArgumentParser( 'lightcurve', description='DIA forced photomtery',
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog=
"""Build a lightcurve by doing forced photomtery on difference images.

Rob write longer description.

For all config options, if the argument isn't given, it will default to
what's in the config, or, if it's not in the config, the defaults
defined in the ParsLightcurve class definition.
"""
                                     )
    parser.add_argument( 'outfile', default=None, nargs='?',
                          help=( "[Optional] Write out a CSV file with the photometry.  If not given, write no file. "
                                 "If you neither specify this nor --save-to-db, you are just wasting time." ) )
    parser.add_argument( '--zp-prov', default=argparse.SUPPRESS,
                         help=( "Provenance id for ZeroPoint to use to find images.  Need either this or "
                                "--zp-prov-tag.  If both are given, this takes precedence (I think)." ) )
    parser.add_argument( '-z', '--zp-prov-tag', default=argparse.SUPPRESS,
                         help=( "Provenance tag to use to find ZeroPoints, which in turn specify Images to "
                                "build the lightcurve from." ) )
    parser.add_argument( "--zp-prov-tag-process", default=argparse.SUPPRESS )
    parser.add_argument( "--object-position-prov", default=argparse.SUPPRESS,
                         help=( "Provenance id for object positioning.  If this and --object-position-prov-tag "
                                "are both None, will use raw object position" ) )
    parser.add_argument( "-p", "--object-position-prov-tag", default=argparse.SUPPRESS,
                         help="Provenace tag for object positoning." )
    parser.add_argument( "--only-existing-subtractions", action='store_true', default=argparse.SUPPRESS,
                         help=( "Only do forced photometry on existing subtractons, don't make new ones.  "
                                "(NOT IMPLEMENTED.)" ) )
    parser.add_argument( "-c", "--crop-image", type=int, nargs=2, default=argparse.SUPPRESS,
                         help="Crop images to this size around object position before subtracting." )
    parser.add_argument( "-f", "--filters", nargs='+', default=argparse.SUPPRESS,
                         help=( "Filters to build lightcurve for.  If not specified (here or in config), "
                                "builds lightcurves for all fitlers of images found in the database for "
                                "this object and date range" ) )
    parser.add_argument( "--mjd0", type=float, default=argparse.SUPPRESS,
                         help="Start building lightcurve at this mjd" )
    parser.add_argument( "--mjd1", type=float, default=argparse.SUPPRESS,
                         help="Stop building lightcurve at this mjd" )
    parser.add_argument( "-i", "--instrument", required=True,
                         help="Name of instrument whose images we're doing photometry on" )
    parser.add_argument( "--object-id", default=argparse.SUPPRESS,
                         help="Database UUID of the object to build a lightcurve for." )
    parser.add_argument( "-o", "--object-name", default=argparse.SUPPRESS,
                         help=( "Database name of object to build a lightcurve for.  Unless, perversely, "
                                "you've set one in config, you need either this or --ojbect-id." ) )
    parser.add_argument( "-s", "--save-to-db", default=False, action="store_true",
                         help="Save trimmed images, subtractions, and dia forced photometry to database?" )
    parser.add_argument( "-n", "--numprocs", type=int, default=1,
                         help="Run this many subprocesses in parallel.  1=run fully serially.  NOT IMLEMENTED." )
    parser.add_argument( "-v", "--verbose", default=False, action="store_true",
                         help="Log at DEBUG level (default INFO)." )
    parser.add_argument( "-w", "--warnings-only", default=False, action="store_true",
                         help=( "Log at the WARNING level (default INFO).  Ignored if --verbose or --errors-only "
                                "are set." ) )
    parser.add_argument( "-e", "--errors-only", default=False, action="store_true",
                         help=( "Log at the ERROR Level (default INFO), if you are bold and really don't want to "
                                "see the warnings.  Ignored if --verbose is given." ) )
    args = parser.parse_args()

    kwargs = vars(args).copy()

    SCLogger.setLevel( "DEBUG" if kwargs['verbose']
                       else "ERROR" if kwargs['errors_only']
                       else "WARNING" if kwargs['warnings_only']
                       else "INFO" )

    SCLogger.info( "lightcurve run with:\n"
                   f"{util.util.reconstruct_commandline(parser, args, 'python /seechange/pipeline/lightcurve.py')}" )

    for kw in [ 'verbose', 'warnings_only', 'errors_only' ]:
        if kw in kwargs:
            del kwargs[kw]

    if ( kwargs['outfile'] is None ) and ( not kwargs['save_to_db'] ):
        SCLogger.error( "Must either give an outputifle, or use --save-to-db.  Otherwise, this does nothing "
                        "but burn cpu time." )
        sys.exit( 1 )
    outfile = kwargs['outfile']
    del kwargs['outfile']

    if outfile is not None:
        raise NotImplementedError( "File writing not implemented." )

    lightcurve = Lightcurve( **kwargs )
    lightcurve.run()

    nfail = len( [ f for f in lightcurve.dia_forced_phots if f is None ] )
    if nfail > 0:
        SCLogger.warning( f"{nfail} out of {len(lightcurve.dia_forced_phots)} (at least!) failed.  "
                          f"(The others returned values, but that doesn't mean they're good....)" )

    if outfile is not None:
        SCLogger.info( f"Writing csv file {outfile}..." )
        lightcurve.write_csv_file( outfile )
        SCLogger.info( "...done." )


# ======================================================================
if __name__ == "__main__":
    main()
