import time
import re

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as sqlUUID

from models.base import Base, SeeChangeBase, UUIDMixin
from models.enums_and_bitflags import (
    bitflag_to_string,
    string_to_bitflag,
    process_steps_dict,
    process_steps_inverse,
    pipeline_products_dict,
    pipeline_products_inverse,
)

from util.logger import SCLogger


class Report(Base, UUIDMixin):
    """A report on the status of analysis of one section from an Exposure.

    The report's main role is to keep a database record of when we started
    and finished processing this section of the Exposure. It also keeps
    track of any errors or warnings that came up during processing.
    """
    __tablename__ = 'reports'

    exposure_id = sa.Column(
        sa.ForeignKey('exposures._id', ondelete='CASCADE', name='reports_exposure_id_fkey'),
        nullable=True,
        index=True,
        doc=(
            "ID of the exposure for which the report was made. "
        )
    )

    section_id = sa.Column(
        sa.Text,
        nullable=True,
        index=True,
        doc=(
            "ID of the section of the exposure for which the report was made. "
        )
    )

    # Not making this a formal foreign key, because in at least one case
    #   in our tests the image is not yet committed to the database when
    #   we try to save the report to the database.  (Originally, reports
    #   assumed they came off of an exposure, and the pipeline assumes
    #   exposure are committed before starting.  The pipeline is able to
    #   run on an image that's not committed to the database, however.)
    image_id = sa.Column(
        # sa.ForeignKey( 'images._id', ondelete='CASCADE', name='reports_image_id_fkey' ),
        sqlUUID,
        nullable=True,
        index=True,
        doc="ID of the image for which the report was made.  Report has (exposure_id,sectionid) XOR image_id."
    )

    start_time = sa.Column(
        sa.DateTime,
        nullable=False,
        index=True,
        doc=(
            "Time when processing of the section started. "
        )
    )

    finish_time = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=(
            "Time when processing of the section finished. "
            "If an error occurred, this will show the time of the error. "
            "If the processing is still ongoing (or hanging) this will be NULL. "
        )
    )

    success = sa.Column(
        sa.Boolean,
        nullable=False,
        index=False,
        server_default='false',
        doc=(
            "Whether the processing of this section was successful. "
        )
    )

    # These next two are not a composite foreign key because
    #   PipelineWorkers only holds active workers, and we want to be
    #   able to see reports for workers that have exited.
    cluster_id = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "ID of the cluster that ran this section (see PipelineWorker). "
        )
    )

    node_id = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "ID of the node where the worker/process ran this section (see PipelineWorker). "
        )
    )

    error_step = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Name of the processing step where an error occurred. "
        )
    )

    error_type = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Type of error that was raised during processing. "
        )
    )

    error_message = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Error message that was raised during processing. "
        )
    )

    warnings = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Comma-separated string of warnings that were raised during processing. "
            "Each warning begins with the processing step name, followed by the warning type and message. "
        )
    )

    process_provid = sa.Column(
        JSONB,
        nullable=True,
        index=False,
        doc="Dictionary of process→provenance_id for the provenances used in pipeline process."
    )

    process_memory = sa.Column(
        JSONB,
        nullable=False,
        server_default='{}',
        doc='Memory usage of the process during processing. '
            'Each key in the dictionary is for a processing step, '
            'and the value is the memory usage in megabytes. '
    )

    process_runtime = sa.Column(
        JSONB,
        nullable=False,
        server_default='{}',
        doc='Runtime of the process during processing. '
            'Each key in the dictionary is for a processing step, '
            'and the value is the runtime in seconds. '
    )

    progress_steps_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=False,
        doc='Bitflag recording what processing steps have already been applied to this section. '
    )

    @property
    def progress_steps(self):
        """A comma separated string of the processing steps that have already been applied to this section. """
        return bitflag_to_string(self.progress_steps_bitflag, process_steps_dict)

    @progress_steps.setter
    def progress_steps(self, value):
        """Set the progress steps for this report using a comma separated string. """
        self.progress_steps_bitflag = string_to_bitflag(value, process_steps_inverse)

    def append_progress(self, value):
        """Add some keywords (in a comma separated string)
        describing what is processing steps were done on this section.
        The keywords will be added to the list "progress_steps"
        and progress_bitflag for this report will be updated accordingly.
        """
        self.progress_steps_bitflag |= string_to_bitflag(value, process_steps_inverse)

    products_exist_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=False,
        doc='Bitflag recording which pipeline products were not None when the pipeline finished. '
    )

    @property
    def products_exist(self):
        """A comma separated string representing which products
        have already been filled on the datastore when the pipeline finished.
        """
        return bitflag_to_string(self.products_exist_bitflag, pipeline_products_dict)

    @products_exist.setter
    def products_exist(self, value):
        """Set the products_exist for this report using a comma separated string. """
        self.products_exist_bitflag = string_to_bitflag(value, pipeline_products_inverse)

    def append_products_exist(self, value):
        """Add some keywords (in a comma separated string)
        describing which products existed (were not None) on the datastore.
        The keywords will be added to the list "products_exist"
        and products_exist_bitflag for this report will be updated accordingly.
        """
        self.products_exist_bitflag |= string_to_bitflag(value, pipeline_products_inverse)

    products_committed_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=False,
        doc='Bitflag recording which pipeline products were not None when the pipeline finished. '
    )

    @classmethod
    def query_for_reports( cls, prov_tag=None, section_id=None, fields=None ):
        """Return a SQL query to find reports.

        Returns a query that, when passed, will find all of the most
        recent reports for each section_id for a given exposure where
        all of the process provenance ids in that report match the
        desired provenance tag.

        You might wrap this inside a JOIN or a SELECT ... FROM (...) where
        join on, or where on, exposure_id.

        See seechange_webap.py::Exposures for an example of usage.

        Parameters
        ----------
          prov_tag : str, default None
            The provenance tag that all processes in the report must
            have provenances tagged with.  Omitted, then just will get
            the most recent report for the exposure regardless of
            provenances.

          section_id : str, default None
            The section to find reports.  If not given, will get reports
            for all sections of the requested exposure.

          fields : str, default None
            Fields of the report table you want.  If not given you, you
            get all of them except for process_provid.  This can NOT
            include process_provid.

        Returns
        -------
          str, dict

          A query and a subdict.  The query is useful for adding to a
          query string as a subquery, and the dict ius useful for
          updating a substitution dict, to pass to psycopg2's
          cursor.execute.

        """

        # ZOMG
        #
        # The goal here is to find all the reports where *all* of the process_provid in the reports
        # are tagged with provtag.  It's not good enough that some processes are tagged.  E.g.,
        # you could easily have a report where all the steps through "subtraction" are tagged with
        # both provtag1 and provtag2, but the later steps are tagged only with provtag2.
        # If you're looking for a report where all steps are tagged with provtag1, then that report
        # is not the one you're looking for.  (Jedi hand wave.)
        #
        # We do this, going from inside to out, by:

        #      1. (subquery re2) explode the process_provid dictionary (it's a jsonb column) into lots of
        #         rows— each row is expanded into separate rows for each process/provenance pair.  Join to
        #         the provenance tags so we have a table of report*, process, provtag; report* is repeated
        #         lots of times, at least once for each process in the report.  Process *might* be
        #         repeated if the provenance id is tagged with multiple tags (which is common).
        #
        #      2. Set aggregate (array w/ distinct) all of the tags for a given (report*, process)
        #         (subquery re1)
        #
        #      3. Array aggregate over processes, making a new array where each element is true
        #         if the process associated with that element has provtag in its provtag array,
        #         false otherwise
        #         (subquery re)
        #
        #      4. Select out only the reports where every element from the previous step is true.
        #         (subquery r)

        # NOTE : keep this synced with the fields that exist in reports
        # Could perhaps do this with introspection....
        allfields = [ '_id', 'exposure_id', 'section_id', 'image_id', 'start_time', 'finish_time',
                      'success', 'cluster_id', 'node_id', 'error_step', 'error_type', 'error_message',
                      'warnings', 'process_memory', 'process_runtime', 'progress_steps_bitflag',
                      'products_exist_bitflag', 'products_committed_bitflag' ]
        fields = allfields if fields is None else fields
        badfields = set()
        # Make sure we aren't going to be bobble tablesed
        for f in fields:
            if f not in allfields:
                badfields.add( f )
        if len(badfields) != 0:
            raise ValueError( f"Invalid fields: {','.join(badfields)}" )

        if not all( [ isinstance( field, str ) for field in fields ] ):
            raise ValueError( "Each field in fields must be a string" )
        if len(fields) < 1:
            raise ValueError( "If fields is not None, it must be a list with at least one element." )
        alphanum = re.compile( '^[a-z0-9_]+' )
        if not all( [ alphanum.search(field) for field in fields ] ):
            # This should be redundant, but, you know, Bobble Tables is a monster
            raise ValueError( "Each field in fields must be a sequence of [a-z0-9_]" )
        disallowed_fields = [ 'process_provid' ]
        if any( [ f in [ disallowed_fields ] for f in fields ] ):
            raise ValueError( f"Fields selected cannot include any of {', '.join(disallowed_fields)}" )

        subdict = {}
        q = "SELECT DISTINCT ON(re.exposure_id, re.section_id) "
        q += ",".join( [ f"re.{f}" for f in fields ] )

        if prov_tag is None:
            ' FROM reports re '
            if section_id is not None:
                q += '   WHERE section)id=%(secid)s ) re '
                subdict[ 'secid' ] = section_id
        else:
            for f in [ "_id", "section_id", "start_time" ]:
                if f not in fields:
                    fields.insert( 0, f )
            fields1 = ",".join( [ f"re1.{f}" for f in fields ] )
            fields2_list = list( fields )
            for f in [ 'exposure_id', 'section_id', 'success', 'error_message', 'start_time' ]:
                if f not in fields2_list:
                    fields2_list.append( f )
            fields2 = ",".join( [ f"re2.{f}" for f in fields2_list ] )
            fields3 = ",".join( [ f"re3.{f}" for f in fields2_list ] )
            q += ( f' FROM ( SELECT {fields1}, array_agg(%(provtag)s=ANY(re1.tags)) AS gotem '
                   f'        FROM ( SELECT {fields2}, array_agg(re2.tag) as tags '
                   f'               FROM ( SELECT DISTINCT ON( re3._id, x.key, r3pt.tag ) '
                   f'                             {fields3}, x.key AS process, r3pt.tag '
                   f'                      FROM reports re3 '
                   f'                      CROSS JOIN jsonb_each_text( re3.process_provid ) x '
                   f'                      INNER JOIN provenance_tags r3pt ON x.value=r3pt.provenance_id ' )
            if section_id is not None:
                q += '                      WHERE section_id=%(secid)s '
                subdict[ 'secid' ] = section_id
            q += ( f'                      ORDER BY re3._id, r3pt.tag '
                   f'                    ) re2 '
                   f'               GROUP BY ( {fields2} )'
                   f'             ) re1 '
                   f'        GROUP BY ( {fields1} ) '
                   f'      ) re '
                   f'WHERE true=ALL( gotem ) ' )
        q += 'ORDER BY re.exposure_id, re.section_id, re.start_time DESC '
        subdict[ 'provtag' ] = prov_tag

        return q, subdict


    @property
    def products_committed(self):
        """A comma separated string representing which products
        have already been successfully saved using the datastore when the pipeline finished.
        """
        return bitflag_to_string(self.products_committed_bitflag, pipeline_products_dict)

    @products_committed.setter
    def products_committed(self, value):
        """Set the products_committed for this report using a comma separated string. """
        self.products_committed_bitflag = string_to_bitflag(value, pipeline_products_inverse)

    def append_products_committed(self, value):
        """Add some keywords (in a comma separated string)
        describing which products were successfully saved by the datastore.
        The keywords will be added to the list "products_committed"
        and products_committed_bitflag for this report will be updated accordingly.
        """
        self.products_committed_bitflag |= string_to_bitflag(value, pipeline_products_inverse)

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # do not pass kwargs to Base.__init__, as there may be non-column attributes

        # verify these attributes get their default even if the object is not committed to DB
        self.success = False
        self.progress_steps_bitflag = 0
        self.products_exist_bitflag = 0
        self.products_committed_bitflag = 0
        self.process_provid = {}
        self.process_memory = {}
        self.process_runtime = {}

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)

    def scan_datastore( self, ds, process_step=None ):
        """Go over all the data in a datastore and update the report accordingly.
        Will commit the Report object to the database.
        If there are any exceptions pending on the datastore it will re-raise them.

        Parameters
        ----------
        ds : DataStore
            The datastore to scan for information.
        process_step : str, optional
            The name of the process step that was just completed.
            This will be added to the progress bitflag.
            If not given, will skip adding the progress flag,
            but will also skip checking warnings and errors...
            Use without a process_step just to update general
            properties of the datastore like the runtime and memory usage,
            or for updating the products_exist and products_committed bitflags
            (e.g., after saving the datastore).
        session : sqlalchemy.orm.Session, optional
            The session to use for committing the changes to the database.
            If not given, will open a session and close it at the end
            of the function.

        """
        t0 = time.perf_counter()
        if 'reporting' not in self.process_runtime:
            self.process_runtime['reporting'] = 0.0

        # check which objects exist on the datastore, and which have been committed
        for prod in pipeline_products_dict.values():
            if getattr(ds, prod) is not None:
                self.append_products_exist(prod)

        # Make sure image id is set if appropriate
        if ds.image is not None:
            self.image_id = ds.image.id

        # Set the cluster and node IDs if they're known
        if hasattr( ds, 'cluster_id' ):
            self.cluster_id = ds.cluster_id
        if hasattr( ds, 'node_id' ):
            self.node_id = ds.node_id

        # Products that have been committed
        self.products_committed = ds.products_committed

        # store the runtime and memory usage statistics
        self.process_runtime.update(ds.runtimes)
        self.process_memory.update(ds.memory_usages)

        if process_step is not None:
            # append the newest step to the progress bitflag
            if process_step in process_steps_inverse:  # skip steps not in the dict
                self.append_progress(process_step)

            # parse the warnings, if they exist
            if isinstance(ds.warnings_list, list):
                new_string = self.read_warnings(process_step, ds.warnings_list)
                if self.warnings is None or self.warnings == '':
                    self.warnings = new_string
                else:
                    self.warnings += '\n***|***|***\n' + new_string

        if ( ds.exceptions is not None ) and ( len(ds.exceptions) > 0 ):
            # The first exception is going to be the reported error type
            if self.error_step is None:
                self.error_step = process_step if process_step is not None else "(unknown)"
                self.error_type = ds.exceptions[0].__class__.__name__
            if self.error_message is None:
                self.error_message = ''
            else:
                self.error_message += '\n***|***|***\n'
            for e in ds.exceptions:
                self.error_message += ( f"Exception {f'in step {process_step}' if process_step is not None else ''}: "
                                        f"{str(e)}\n" )

            # Now that we've reported them, clear out the exceptions
            ds.exceptions = []

        self.upsert()

        self.process_runtime['reporting'] += time.perf_counter() - t0


    @staticmethod
    def read_warnings(process_step, warnings_list):
        """Convert a list of warnings into a comma separated string. """
        formatted_warnings = []
        for w in warnings_list:
            text = f'{process_step}: {w.category} {w.message} ({w.filename}:{w.lineno})'
            formatted_warnings.append(text)
            SCLogger.warning(text)  # make sure warnings also get printed to the log/on screen.

        warnings_list.clear()  # remove all the warnings but keep the list object

        return ', '.join(formatted_warnings)

    def get_downstreams( self, session=None ):
        """Reports have no downstreams."""
        return []
