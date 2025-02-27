import uuid
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.dialects.postgresql import JSONB
from models.base import Base, SmartSession


class AuthUser(Base):
    __tablename__ = "authuser"

    id = sa.Column( sqlUUID(as_uuid=True), primary_key=True, default=uuid.uuid4 )
    username = sa.Column( sa.Text, nullable=False, unique=True, index=True )
    displayname = sa.Column( sa.Text, nullable=False )
    email = sa.Column( sa.Text, nullable=False, index=True )
    pubkey = sa.Column( sa.Text )
    privkey = sa.Column( JSONB )

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self._groups = None

    @orm.reconstructor
    def init_on_load(self):
        self._groups = None

    def load_groups( self, session=None ):
        self._groups = None
        with SmartSession( session ) as sess:
            groups = ( sess.query( AuthGroup )
                       .join( auth_user_group, auth_user_group.c.groupid==AuthGroup.id )
                       .filter( auth_user_group.c.userid==self.id ) ).all()
            self._groups = [ g.name for g in groups ]

    # A list of strings indicating which groups the user is in
    @property
    def groups( self ):
        if self._groups is None:
            self.load_groups()
        return self._groups

    @classmethod
    def get_by_id( cls, uuid, session=None ):
        with SmartSession( session ) as sess:
            return sess.query( cls ).filter( cls.id==uuid ).first()


class PasswordLink(Base):
    __tablename__ = "passwordlink"

    id = sa.Column( sqlUUID(as_uuid=True), primary_key=True, default=uuid.uuid4 )
    userid = sa.Column( sqlUUID(as_uuid=True), sa.ForeignKey("authuser.id", ondelete="CASCADE"), index=True )
    expires = sa.Column( sa.DateTime(timezone=True) )

    @classmethod
    def get_by_id( cls, uuid, session=None ):
        with SmartSession( session ) as sess:
            return sess.query( cls ).filter( cls.id==uuid ).first()


class AuthGroup(Base):
    __tablename__ = "authgroup"

    id = sa.Column( sqlUUID(as_uuid=True), primary_key=True, default=uuid.uuid4 )
    name = sa.Column( sa.Text, nullable=False, unique=True, index=True )
    description = sa.Column( sa.Text )

    @classmethod
    def get_by_id( cls, uuid, session=None ):
        with SmartSession( session ) as sess:
            return sess.query( cls ).filter( cls.id==uuid ).first()


auth_user_group = sa.Table(
    'auth_user_group',
    Base.metadata,
    sa.Column( 'userid',
               sqlUUID,
               sa.ForeignKey( 'authuser.id', ondelete='CASCADE', name='auth_user_group_user_fkey' ),
               index=True,
               primary_key=True ),
    sa.Column( 'groupid',
               sqlUUID,
               sa.ForeignKey( 'authgroup.id', ondelete='CASCADE', name='auth_user_group_group_fkey' ),
               index=True,
               primary_key=True ),
)
