import time

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB

from models.base import Base, SeeChangeBase, UUIDMixin, SmartSession
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
        nullable=False,
        index=True,
        doc=(
            "ID of the exposure for which the report was made. "
        )
    )

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc=(
            "ID of the section of the exposure for which the report was made. "
        )
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
        index=True,
        server_default='false',
        doc=(
            "Whether the processing of this section was successful. "
        )
    )

    num_prev_reports = sa.Column(
        sa.Integer,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        doc=(
            "Number of previous reports for this exposure, section, and provenance. "
        )
    )

    worker_id = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "ID of the worker/process that ran this section. "
        )
    )

    node_id = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "ID of the node where the worker/process ran this section. "
        )
    )

    cluster_id = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "ID of the cluster where the worker/process ran this section. "
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
        index=True,
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
        index=True,
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
        index=True,
        doc='Bitflag recording which pipeline products were not None when the pipeline finished. '
    )

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

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='images_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this report. "
            "The provenance has upstreams that point to the "
            "measurements and R/B score objects that themselves "
            "point back to all the other provenances that were "
            "used to produce this report. "
        )
    )

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # do not pass kwargs to Base.__init__, as there may be non-column attributes

        # verify these attributes get their default even if the object is not committed to DB
        self.success = False
        self.num_prev_reports = 0
        self.progress_steps_bitflag = 0
        self.products_exist_bitflag = 0
        self.products_committed_bitflag = 0
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

        # parse the error, if it exists, so we can get to other data products without raising
        exception = ds.read_exception()

        # check which objects exist on the datastore, and which have been committed
        for prod in pipeline_products_dict.values():
            if getattr(ds, prod) is not None:
                self.append_products_exist(prod)

        self.products_committed = ds.products_committed

        # store the runtime and memory usage statistics
        self.process_runtime.update(ds.runtimes)  # update with new dictionary
        self.process_memory.update(ds.memory_usages)  # update with new dictionary

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

            if exception is not None:
                self.error_type = exception.__class__.__name__
                self.error_message = str(exception)
                self.error_step = process_step

        self.upsert()

        self.process_runtime['reporting'] += time.perf_counter() - t0

        if exception is not None:
            raise exception


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

    def get_downstreams( self, session=None, siblings=False ):
        """Reports have no downstreams."""
        return []

    # ======================================================================
    # The fields below are things that we've deprecated; these definitions
    #   are here to catch cases in the code where they're still used

    @property
    def exposure( self ):
        raise RuntimeError( f"Don't use Report.exposure, use exposure_id" )

    @exposure.setter
    def exposure( self, val ):
        raise RuntimeError( f"Don't use Report.exposure, use exposure_id" )
