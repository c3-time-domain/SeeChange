from models.image import Image
from models.datafile import DataFile
from enums_and_bitflags import CalibratorTypeConverter

class CalibratorFile(Base, AutoIDMixin):
    __tablename__ = 'calibrator_files'

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        default=CalibratorTypeConverter.convert( 'unknown' ),
        doc="Type of calibrator (Dark, Flat, Linearity, etc.)"
    )

    @hybrid_property
    def type( self ):
        return CalibratorTypeConverter.convert( self._type )

    @type.expression
    def type( cls ):
        return sa.case( CalibratorTypeConverter.dict, value=cls._type )

    @type.setter
    def type( self, value ):
        self._type = CalibratorTypeConverter.convert( value )

    calibrator_set = sa.Column(
        sa.Text,
        nullable=False,
        doc="A string identifying the set of calibrators, which will go into provenance"
    )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Instrument this calibrator image is for"
    )

    sensor_section = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Sensor Section of the Instrument this calibrator image is for"
    )

    image_id = sa.Column(
        sa.ForeignKey( 'images.id', ondelete='CASCADE', name='calibrator_files_image_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the image (if any) that is this calibrator'
    ),

    image = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',  # ROB REVIEW THIS
        doc='Image for this CalibratorImage (if any)'
    ),

    datafile_id = sa.Column(
        sa.ForeignKey( 'data_files.id', ondelete='CASCADE', name='calibrator_files_data_file_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the miscellaneous data file (if any) that is this calibrator'
    ),

    datafile = orm.relationship(
        'DataFile',
        cascade='save-update, merge, refresh-expire, expunge', # ROB REVIEW THIS
        doc='DataFile for this CalibratorFile (if any)'
    ),

    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file becomes valid.  If None, this '
              ' calibrator is valid from the beginning of time.' )
    )

    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file is no longer.  If None, this '
              ' calibrator is valid from the beginning of time.' )
    )
