import requests
import binascii

from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA

from util.config import Config


class ExposureLauncher:
    def __init__( self, cluster_id, verify=True ):
        self.sleeptime = 120
        self.cluster_id = cluster_id
        cfg = Config.get()
        self.url = cfg.value( 'conductor.conductor_url' )
        self.username = cfg.value( 'conductor.username' )
        self.password = cfg.value( 'conductor.password' )
        self.verify = verify
        self.req = None
        
            
    def __call__( self ):
        done = False
        req = None
        while not done:
            # Make sure we're logged into the conductor
                    
                
        
        
