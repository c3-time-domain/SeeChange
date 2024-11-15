import requests
import binascii

from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA

from util.config import Config
from util.rkauth_client import rkAuthClient

class ConductorConnector(rkAuthClient):
    def __init__( self, url=None, username=None, password=None, verify=True ):
        cfg = Config.get()
        url = url if url is not None else cfg.value( 'conductor.conductor_url' )
        username = username if username is not None else cfg.value( 'conductor.username' )
        if password is None:
            password = cfg.value( 'conductor.password' )
            if password is None:
                if cfg.value( 'conductor.password_file' ) is None:
                    raise ValueError( "ConductorConnector must have a password, either passed, or from "
                                      "conductor.password or conductor.password_file configs" )
                with open( cfg.value( "conductor.password_file" ) ) as ifp:
                    password = ifp.readline().strip()
        super().__init__( url, username, password, verify=verify )
