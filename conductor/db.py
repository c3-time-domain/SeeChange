from models.base import SmartSession
import models.user

AuthUser = models.user.AuthUser
PasswordLink = models.user.PasswordLink

def DBSession( *args, **kwargs ):
    return SmartSession( *args, **kwargs )
