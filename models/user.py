from datetime import datetime, timedelta
import dateutil.parser
import pytz
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.dialects.postgresql import JSONB
from models.base import Base, SmartSession
from util.util import as_UUID, as_datetime


class AuthUser(Base):
    __tablename__ = "authuser"

    id = sa.Column( sqlUUID(as_uuid=True), primary_key=True, default=uuid.uuid4 )
    username = sa.Column( sa.Text, nullable=False, unique=True, index=True )
    displayname = sa.Column( sa.Text, nullable=False )
    email = sa.Column( sa.Text, nullable=False, index=True )
    pubkey = sa.Column( sa.Text )
    privkey = sa.Column( JSONB )

    # This is here rather than just using things already defined in base
    #   because this table is designed to work with the pre-existing auth
    #   library in rkwebutil.
    @classmethod
    def get( cls, id, session=None ):
        id = id if isinstance( id, uuid.UUID) else uuid.UUID( id )
        with SmartSession(session) as sess:
            q = sess.query(cls).filter( cls.id==id )
            if q.count() > 1:
                raise ErrorMsg( f'Error, {cls.__name__} {id} multiply defined!  This shouldn\'t happen.' )
            if q.count() == 0:
                return None
            return q[0]

    @classmethod
    def getbyusername( cls, name, session=None ):
        with SmartSession(session) as sess:
            q = sess.query(cls).filter( cls.username==name )
            return q.all()

    @classmethod
    def getbyemail( cls, email, session=None ):
        with SmartSession(session) as sess:
            q = sess.query(cls).filter( cls.email==email )
            return q.all()


class PasswordLink(Base):
    __tablename__ = "passwordlink"

    id = sa.Column( sqlUUID(as_uuid=True), primary_key=True, default=uuid.uuid4 )
    userid = sa.Column( sqlUUID(as_uuid=True), sa.ForeignKey("authuser.id", ondelete="CASCADE"), index=True )
    expires = sa.Column( sa.DateTime(timezone=True) )
    
    @classmethod
    def new( cls, userid, expires=None, session=None ):
        if expires is None:
            expires = datetime.now(pytz.utc) + timedelta(hours=1)
        else:
            expires = as_datetime( expires )
        with SmartSession(session) as sess:
            link = PasswordLink( userid = as_UUID(userid),
                                 expires = expires )
            sess.add( link )
            sess.commit()
            return link

    # This is here rather than just using things already defined in base
    #   because this table is designed to work with the pre-existing auth
    #   library in rkwebutil.
    @classmethod
    def get( cls, uuid, session=None ):
        with SmartSession(session) as sess:
            q = sess.query( PasswordLink ).filter( PasswordLink.id==uuid )
            if q.count() == 0:
                return None
            else:
                return q.first()
