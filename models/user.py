import uuid
import sqlalchemy as sa
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
    isadmin = sa.Column( sa.Boolean, nullable=False, server_default='false' )

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
