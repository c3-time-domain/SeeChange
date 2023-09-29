import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property

from models.base import Base, AutoIDMixin
from models.image import Image
from models.datafile import DataFile
from models.enums_and_bitflags import CalibratorTypeConverter, CalibratorSetConverter, FlatTypeConverter

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

    _calibrator_set = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        default=CalibratorTypeConverter.convert('unknown'),
        doc="Calibrator set for instrument (unknown, externally_supplied, general, nightly)"
    )

    @hybrid_property
    def calibrator_set( self ):
        return CalibratorSetConverter.convert( self._type )

    @calibrator_set.expression
    def calibrator_set( cls ):
        return sa.case( CalibratorSetConverter.dict, value=cls._calibrator_set )

    @calibrator_set.setter
    def calibrator_set( self, value ):
        self._calibrator_set = CalibratorSetConverter.convert( value )

    _flat_type = sa.Column(
        sa.SMALLINT,
        nullable=True,
        index=True,
        doc="Type of flat (unknown, observatory_supplied, sky, twilight, dome), or None if not a flat"
    )

    @hybrid_property
    def flat_type( self ):
        return FlatTypeConverter.convert( self._type )

    @flat_type.inplace.expression
    @classmethod
    def flat_type( cls ):
        return sa.case( FlatTypeConverter.dict, value=cls._flat_type )

    @flat_type.setter
    def flat_type( self, value ):
        self._flat_type = FlatTypeConverter.convert( value )

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
    )

    image = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',  # ROB REVIEW THIS
        doc='Image for this CalibratorImage (if any)'
    )

    datafile_id = sa.Column(
        sa.ForeignKey( 'data_files.id', ondelete='CASCADE', name='calibrator_files_data_file_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the miscellaneous data file (if any) that is this calibrator'
    )

    datafile = orm.relationship(
        'DataFile',
        cascade='save-update, merge, refresh-expire, expunge', # ROB REVIEW THIS
        doc='DataFile for this CalibratorFile (if any)'
    )

    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file becomes valid.  If None, this '
              ' calibrator is valid from the beginning of time.' )
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file is no longer.  If None, this '
              ' calibrator is valid to the end of time.' )
    )
