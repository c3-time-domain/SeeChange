import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, AutoIDMixin, SmartSession
from models.image import Image
from models.provenance import Provenance
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint


class Reference(Base, AutoIDMixin):
    """
    A table that refers to each reference Image object,
    based on the validity time range, and the object/field it is targeting.
    The provenance of this table (tagged with the "reference" process)
    will have as its upstream IDs the provenance IDs of the image,
    the source list, the PSF, the WCS, and the zero point.

    This means that the reference should always come loaded
    with the image and all its associated products,
    based on the provenance given when it was created.
    """

    __tablename__ = 'refs'   # 'references' is a reserved postgres word

    image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete='CASCADE', name='references_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the reference image that this object is referring to. "
    )

    image = orm.relationship(
        'Image',
        lazy='selectin',
        cascade='save-update, merge, refresh-expire, expunge',
        foreign_keys=[image_id],
        doc="The reference image that this entry is referring to. "
    )

    # the following can't be association products (as far as I can tell) because they need to be indexed
    target = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc=(
            'Name of the target object or field id. '
            'This string is used to match the reference to new images, '
            'e.g., by matching the field ID on a pre-defined grid of fields. '
        )
    )

    filter = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Filter used to make the images for this reference image. "
    )

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Section ID of the reference image. "
    )

    # this allows choosing a different reference for images taken before/after the validity time range
    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc="The start of the validity time range of this reference image. "
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc="The end of the validity time range of this reference image. "
    )

    # this badness is in addition to the regular bitflag of the underlying products
    # it can be used to manually kill a reference and replace it with another one
    # even if they share the same time validity range
    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Whether this reference image is bad. "
    )

    bad_reason = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "The reason why this reference image is bad. "
            "Should be a single pharse or a comma-separated list of reasons. "
        )
    )

    bad_comment = sa.Column(
        sa.Text,
        nullable=True,
        doc="Any additional comments about why this reference image is bad. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='references_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this reference. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this reference. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this reference. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this reference. "
        )
    )

    def __init__(self, **kwargs):
        self.sources = None
        self.psf = None
        self.wcs = None
        self.zp = None
        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        if key == 'image' and value is not None:
            self.target = value.target
            self.filter = value.filter
            self.section_id = value.section_id
            self.sources = value.sources
            self.psf = value.psf
            self.wcs = value.wcs
            self.zp = value.zp

        super().__setattr__(key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        self.sources = None
        self.psf = None
        self.wcs = None
        self.zp = None
        this_object_session = orm.Session.object_session(self)
        if this_object_session is not None:  # if just loaded, should usually have a session!
            self.load_upstream_products(this_object_session)

    def make_provenance(self):
        """Make a provenance for this reference image. """
        upstreams = [self.image.provenance]
        for att in ['image', 'sources', 'psf', 'wcs', 'zp']:
            if getattr(self, att) is not None:
                upstreams.append(getattr(self, att).provenance)
            else:
                raise ValueError(f'Reference must have a valid {att}.')

        self.provenance = Provenance(
            code_version=self.image.provenance.code_version,
            process='reference',
            parameters={},  # do we need any parameters for a reference's provenance?
            upstreams=upstreams,
        )

    def get_upstream_provenances(self):
        """Collect the provenances for all upstream objects.
        Assumes all the objects are already committed to the DB
        (or that at least they have provenances with IDs).

        Returns
        -------
        list of Provenance objects:
            a list of unique provenances, one for each data type.
        """
        prov = []
        if self.image is None or self.image.provenance is None or self.image.provenance.id is None:
            raise ValueError('Reference must have a valid image with a valid provenance ID.')
        prov.append(self.image.provenance)

        # TODO: it seems like we should require that Reference always has all of these when saved
        if self.sources is not None and self.sources.provenance is not None and self.sources.provenance.id is not None:
            prov.append(self.sources.provenance)
        if self.psf is not None and self.psf.provenance is not None and self.psf.provenance.id is not None:
            prov.append(self.psf.provenance)
        if self.wcs is not None and self.wcs.provenance is not None and self.wcs.provenance.id is not None:
            prov.append(self.wcs.provenance)
        if self.zp is not None and self.zp.provenance is not None and self.zp.provenance.id is not None:
            prov.append(self.zp.provenance)
        return prov

    def load_upstream_products(self, session=None):
        """Make sure the reference image has its related products loaded.

        This only works after the image and products are committed to the database,
        with provenances consistent with what is saved in this Reference's provenance.
        """
        with SmartSession(session) as session:
            prov_ids = self.provenance.upstream_ids

            sources = session.scalars(
                sa.select(SourceList).where(
                    SourceList.image_id == self.image_id,
                    SourceList.provenance_id.in_(prov_ids),
                )
            ).all()
            if len(sources) > 1:
                raise ValueError(
                    f"Image {self.image_id} has more than one SourceList matching upstream provenance."
                )
            elif len(sources) == 1:
                self.image.sources = sources[0]
                self.sources = sources[0]

            psfs = session.scalars(
                sa.select(PSF).where(
                    PSF.image_id == self.image_id,
                    PSF.provenance_id.in_(prov_ids),
                )
            ).all()
            if len(psfs) > 1:
                raise ValueError(
                    f"Image {self.image_id} has more than one PSF matching upstream provenance."
                )
            elif len(psfs) == 1:
                self.image.psf = psfs[0]
                self.psf = psfs[0]

            if self.sources is not None:
                wcses = session.scalars(
                    sa.select(WorldCoordinates).where(
                        WorldCoordinates.sources_id == self.sources.id,
                        WorldCoordinates.provenance_id.in_(prov_ids),
                    )
                ).all()
                if len(wcses) > 1:
                    raise ValueError(
                        f"Image {self.image_id} has more than one WCS matching upstream provenance."
                    )
                elif len(wcses) == 1:
                    self.image.wcs = wcses[0]
                    self.wcs = wcses[0]

                zps = session.scalars(
                    sa.select(ZeroPoint).where(
                        ZeroPoint.sources_id == self.sources.id,
                        ZeroPoint.provenance_id.in_(prov_ids),
                    )
                ).all()
                if len(zps) > 1:
                    raise ValueError(
                        f"Image {self.image_id} has more than one ZeroPoint matching upstream provenance."
                    )
                elif len(zps) == 1:
                    self.image.zp = zps[0]
                    self.zp = zps[0]

    def merge_all(self, session):
        """Merge the reference into the session, along with Image and products. """

        new_ref = session.merge(self)
        new_ref.image = self.image.merge_all(session)

        return new_ref
