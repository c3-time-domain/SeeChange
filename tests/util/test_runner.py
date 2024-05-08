from util.runner import Runner


class twoxplusone:
    def __init__( self ):
        self.vals = []
        self.tot = 0
        
    def calc( self, x ):
        return 2 * x + 1

    def summer( self, x ):
        self.vals.append( x )
        self.tot += x

class sometimesfails:
    def __init__( self ):
        self.vals = []

    def do( self, fail ):
        if fail:
            raise RuntimeError( "Fail" )

    def aggregator( self, x ):
        self.vals.append( x )


def test_runner():
    numprocs = 4
    
    for persist in ( True, False ):
        # Test when the number of values is < the number of processes

        doer = twoxplusone()
        runner = Runner( numprocs,
                         lambda x: doer.calc(x),
                         aggregator=lambda x: doer.summer(x),
                         persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( [1, 2, 3] )
        assert nargs == 3
        assert nsucces == 3
        assert nfail == 0
        assert doer.vals == [ 3, 5, 7 ]
        assert doer.tot == 15

        # Test when the number of vals is > the number of processses

        doer = twoxplusone()
        runner = Runner( numprocs,
                         lambda x: doer.calc(x),
                         aggregator=lambda x: doer.summer(x),
                         persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( [1, 2, 3, 4, 5, 6, 7] )
        assert nargs == 7
        assert nsuccess == 7
        assert nfail == 0
        assert doer.vals == [ 3, 5, 7, 9, 11, 13, 15 ]
        assert doer.tot == 63

        # Test when the number of vals is = the number of processes

        doer = twoxplusone()
        runner = Runner( numprocs,
                         lambda x: doer.calc(x),
                         aggregator=lambda x: doer.summer(x),
                         persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( [1, 2, 3, 4] )
        assert nargs == 4
        assert nsuccess == 4
        assert nfail == 0
        assert doer.vals == [ 3, 5, 7, 9 ]
        assert doer.tot == 24

        # Test failures

        doer = sometimesfails()
        runner = Runner( numprocs,
                         lambda x: doer.do(x),
                         aggregator=lambda x: doer.aggregator(x),
                         persistent_processes=persist )
        nargs, nsuccess, nfail = runner.go( [ True, True, False, True, False, False, True ] )
        assert nargs == 7
        assert nsuccess == 4
        assert nfail == 3
        assert doer.vals == [ True, True, None, True, None, None, True ]
        

