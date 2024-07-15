import uuid
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.dialects.postgresql import JSONB
from models.base import Base

class AuthUser(Base):
    __tablename__ = "authuser"

    id = sa.Column( sqlUUID(as_uuid=True), primary_key=True, default=uuid.uuid4 )
    username = sa.Column( sa.Text, nullable=False, unique=True, index=True )
    displayname = sa.Column( sa.Text, nullable=False )
    email = sa.Column( sa.Text, nullable=False, index=True )
    pubkey = sa.Column( sa.Text )
    privkey = sa.Column( JSONB )

class PasswordLink(Base):
    __tablename__ = "passwordlink"

    id = sa.Column( sqlUUID(as_uuid=True), primary_key=True, default=uuid.uuid4 )
    userid = sa.Column( sqlUUID(as_uuid=True), sa.ForeignKey("authuser.id", ondelete="CASCADE"), index=True )
    expires = sa.Column( sa.DateTime(timezone=True) )
