import multiprocessing
from util.runner import Runner


class twoxplusone:
    def __init__( self ):
        self.vals = []
        self.tot = 0
        
    def calc( self, x ):
        return 2 * x + 1

    def summer( self, dex, x ):
        self.vals.append( x )
        self.tot += x

class sometimesfails:
    def __init__( self ):
        self.args = [ True, True, False, True, False, False, True ]
        self.args = [ ( i, self.args[i] ) for i in range(len(self.args)) ]
        self.vals = [ -1 for i in range(len(self.args)) ]

    def do( self, arg ):
        i, notfail = arg
        if not notfail:
            raise RuntimeError( "Fail" )
        return i

    def aggregator( self, dex, x ):
        self.vals[dex] = x


def test_runner():
    numprocs = 4
    
    for persist in ( True, False ):
        # Test when the number of values is < the number of processes

        doer = twoxplusone()
        runner = Runner( numprocs, doer.calc, aggregator=doer.summer, persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( [1, 2, 3] )
        kids = multiprocessing.active_children()
        assert all( [ not k.is_alive() for k in kids ] )
        assert nargs == 3
        assert nsuccess == 3
        assert nfail == 0
        assert set( doer.vals ) == { 3, 5, 7 }
        assert doer.tot == 15

        # Test when the number of vals is > the number of processses

        doer = twoxplusone()
        runner = Runner( numprocs, doer.calc, aggregator=doer.summer, persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( [1, 2, 3, 4, 5, 6, 7] )
        kids = multiprocessing.active_children()
        assert all( [ not k.is_alive() for k in kids ] )
        assert nargs == 7
        assert nsuccess == 7
        assert nfail == 0
        assert set( doer.vals ) == { 3, 5, 7, 9, 11, 13, 15 }
        assert doer.tot == 63

        # Test when the number of vals is = the number of processes

        doer = twoxplusone()
        runner = Runner( numprocs, doer.calc, aggregator=doer.summer, persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( [1, 2, 3, 4] )
        kids = multiprocessing.active_children()
        assert all( [ not k.is_alive() for k in kids ] )
        assert nargs == 4
        assert nsuccess == 4
        assert nfail == 0
        assert set( doer.vals ) == { 3, 5, 7, 9 }
        assert doer.tot == 24

        # Test failures

        doer = sometimesfails()
        runner = Runner( numprocs, doer.do, aggregator=doer.aggregator, persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( doer.args )
        kids = multiprocessing.active_children()
        assert all( [ not k.is_alive() for k in kids ] )
        assert nargs == 7
        assert nsuccess == 4
        assert nfail == 3
        assert doer.vals == [ 0, 1, None, 3, None, None, 6 ]
        

