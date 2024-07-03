import random

from models.base import Base, AutoIDMixin, FileOnDiskMixin

class DiskFile(Base, AutoIDMixin, FileOnDiskMixin):
    """A temporary database table for testing FileOnDiskMixin

    """
    hexbarf = ''.join( [ random.choice( '0123456789abcdef' ) for i in range(8) ] )
    __tablename__ = f"test_diskfiles_{hexbarf}"
    nofile = True
    