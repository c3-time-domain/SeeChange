import requests
import binascii

from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA

from util.config import Config

class ConductorConnector:
    def __init__( self, url=None, username=None, password=None, verify=True ):
        cfg = Config.get()
        self.url = url if url is not None else cfg.value( 'conductor.conductor_url' )
        self.username = username if username is not None else cfg.value( 'conductor.username' )
        if password is None:
            password = cfg.value( 'conductor.password' )
            if password is None:
                if cfg.value( 'conductor.password_file' ) is None:
                    raise ValueError( "ConductorConnector must have a password, either passed, or from "
                                      "conductor.password or conductor.password_file configs" )
                with open( cfg.value( "conductor.password_file" ) ) as ifp:
                    password = ifp.readline().strip()
        self.password = password
        self.verify = verify
        self.req = None

    def verify_logged_in( self ):
        must_log_in = False
        if self.req is None:
            must_log_in = True
        else:
            response = self.req.post( f'{self.url}/auth/isauth', verify=self.verify )
            if response.status_code != 200:
                raise RuntimeError( f"Error talking to conductor: {response.text}" )
            data = response.json()
            if not data['status'] :
                must_log_in = True
            else:
                if data['username'] != self.username:
                    response = self.req.post( f'{self.url}/auth/logout', verify=self.verify )
                    if response.status_code != 200:
                        raise RuntimeError( f"Error logging out of conductor: {response.text}" )
                    data = response.json()
                    if ( 'status' not in data ) or ( data['status'] != 'Logged out' ):
                        raise RuntimeError( f"Unexpected response logging out of conductor: {response.text}" )
                    must_log_in = True

        if must_log_in:
            self.req = requests.Session()
            response = self.req.post( f'{self.url}/auth/getchallenge', json={ 'username': self.username },
                                      verify=self.verify )
            if response.status_code != 200:
                raise RuntimeError( f"Error trying to log into conductor: {response.text}" )
            try:
                data = response.json()
                challenge = binascii.a2b_base64( data['challenge'] )
                enc_privkey = binascii.a2b_base64( data['privkey'] )
                salt = binascii.a2b_base64( data['salt'] )
                iv = binascii.a2b_base64( data['iv'] )
                aeskey = PBKDF2( self.password.encode('utf-8'), salt, 32, count=100000, hmac_hash_module=SHA256 )
                aescipher = AES.new( aeskey, AES.MODE_GCM, nonce=iv )
                privkeybytes = aescipher.decrypt( enc_privkey )
                # SOMETHING I DON'T UNDERSTAND, I get back the bytes I expect
                # (i.e. if I dump doing the equivalent decrypt operation in the
                # javascript that's in rkauth.js) here, but there are an additional
                # 16 bytes at the end; I don't know what they are.
                privkeybytes = privkeybytes[:-16]
                privkey = RSA.import_key( privkeybytes )
                rsacipher = PKCS1_OAEP.new( privkey, hashAlgo=SHA256 )
                decrypted_challenge = rsacipher.decrypt( challenge ).decode( 'utf-8' )
            except Exception as e:
                raise RuntimeError( "Failed to log in, probably incorrect password." )

            response = self.req.post( f'{self.url}/auth/respondchallenge',
                                      json={ 'username': self.username, 'response': decrypted_challenge },
                                      verify=self.verify )
            if response.status_code != 200:
                raise RuntimeError( f"Failed to log into conductor: {response.text}" )
            data = response.json()
            if ( ( data['status'] != 'ok' ) or ( data['username'] != self.username ) ):
                raise RuntimeError( f"Unexpected response logging in to conductor: {response.text}" )

    def send( self, url, postjson={} ):
        self.verify_logged_in()
        slash = '/' if ( ( self.url[-1] != '/' ) and ( url[0] != '/' ) ) else ''
        response = self.req.post( f'{self.url}{slash}{url}', json=postjson, verify=self.verify )
        if response.status_code != 200:
            raise RuntimeError( f"Got response {response.status_code} from conductor: {response.text}" )
        if response.headers.get('Content-Type')[:16]!= 'application/json':
            raise RuntimeError( f"Expected json back from conductor but got "
                                f"{response.headers.get('Content-Type')}" )
        return response.json()
