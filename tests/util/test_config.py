import os
import pathlib
import pytest

from util import config

_rundir = pathlib.Path(__file__).parent

# A note about pytest: Things aren't completely sandboxed.  When I call
# config.Config.get(), it sets Config._default, and that carries over
# from one test to the next even if the call wasn't in a fixture with
# class scope.  (The tests below are ordered with this in mind.)


class TestConfig:
    @pytest.fixture(scope='class')
    def cfg(self):
        # We don't want to set the default config, because the config
        #   here is just for these tests.  All the other tests need
        #   what's in tests/seechange_config_test.yaml
        return config.Config.get(_rundir / 'test.yaml', setdefault=False)

    def test_default_default( self ):
        # make sure that when we load a config without parameters,
        # it uses the default config file
        default_config_path = os.getenv( 'SEECHANGE_CONFIG' )
        assert default_config_path is not None
        assert config.Config._default_default == str(default_config_path)

    def test_config_path( self, cfg ):
        assert cfg._path == (_rundir / "test.yaml").resolve()

    def test_preload(self, cfg):
        assert cfg.value('preload1dict1.preload1_1val1') == '2_1val1'
        assert cfg.value('preload1dict1.preload1_1val2') == '1_1val2'
        assert cfg.value('preload1dict1.preload2_1val2') == 'main1val2'
        assert cfg.value('preload1dict1.preload2_1val3') == '2_1val3'
        assert cfg.value('preload1dict1.main1val3') == 'main1val3'
        assert cfg.value('preload1dict1.augment1val1') == 'a1_1val1'

        assert cfg.value('preload1dict2') == '2scalar1'

        assert cfg.value('preload1list1') == [ 'main1' ]
        assert cfg.value('preload1list2') == [ '1_2val0', '1_2val1', 'a1_2val0' ]

        assert cfg.value('preload1scalar1') == '2scalar1'
        assert cfg.value('preload1scalar2') == '1scalar2'
        assert cfg.value('preload1scalar3') == 'main3'
        assert cfg.value('preload2scalar2') == '2scalar2'

    def test_main(self, cfg):
        assert cfg.value('maindict.mainval1') == 'val1'
        assert cfg.value('maindict.mainval2') == 'val2'

        assert cfg.value('mainlist1') == ['main1', 'main2', 'main3', 'aug1', 'aug2']
        assert cfg.value('mainlist2') == ['override1', 'override2']

        assert cfg.value('mainscalar1') == 'aug1'
        assert cfg.value('mainscalar2') == 'override2'
        assert cfg.value('mainscalar3') == 'override2'

        assert cfg.value('mainnull') is None
        with pytest.raises( ValueError, match="Field.*doesn't exist" ):
            cfg.value('notdefined')

    def test_override(self, cfg):
        assert cfg.value( 'override1list1.0' ) == '1_1override1'
        assert cfg.value( 'override1list1.1' ) == '1_1override2'
        assert cfg.value( 'override1list2' ) == [ '2_2override1', '2_2override2' ]

    def test_fieldsep( self, cfg ):
        fields, isleaf, curfield, ifield = cfg._fieldsep( 'nest.nest1.0.nest1a' )
        assert isleaf == False
        assert curfield == 'nest'
        assert fields == ['nest', 'nest1', '0', 'nest1a' ]
        assert ifield is None
        fields, isleaf, curfield, ifield = cfg._fieldsep( '0.test' )
        assert isleaf == False
        assert ifield == 0
        fields, isleaf, curfield, ifield = cfg._fieldsep( 'mainlist2' )
        assert isleaf
        fields, isleaf, curfield, ifield = cfg._fieldsep( 'mainscalar1' )
        assert isleaf

    def test_nest(self, cfg):
        assert cfg.value( 'nest' ) ==  { 'nest1': [ { 'nest1a': { 'val': 'foo' } }, 42 ],
                                         'nest2': { 'val': 'bar' } }
        assert cfg.value( 'nest.nest1.0.nest1a.val' ) == 'foo'

    def test_missing_value_with_default(self, cfg):
        with pytest.raises(ValueError, match="Field .* doesn't exist"):
            cfg.value( 'nest_foo' )
        assert cfg.value( 'nest_foo', 'default' ) == 'default'

        with pytest.raises(ValueError, match="Error getting field .*"):
            cfg.value( 'nest.nest15' )
        assert cfg.value( 'nest.nest15', 15) == 15

        with pytest.raises(ValueError, match="Error getting field .*"):
            cfg.value( 'nest.nest1.99' )
        assert cfg.value( 'nest.nest1.99', None) is None

        with pytest.raises(ValueError, match="Error getting field .*"):
            cfg.value( 'nest.nest1.0.nest1a.foo' )
        assert cfg.value( 'nest.nest1.0.nest1a.foo', 'bar') == 'bar'

    def test_set(self, cfg):
        clone = config.Config.get( cfg._path, static=False )
        assert config.Config.get( cfg._path ) is cfg
        assert config.Config.get( cfg._path ) is not clone

        with pytest.raises( RuntimeError, match="Not permitted to modify static Config object." ):
            cfg.set_value( 'mainscalar1', 'this_should_not_work' )
        assert cfg.value( 'mainscalar1' ) != 'this_should_not_work'

        with pytest.raises( TypeError, match="Tried to add a non-integer field to a list." ):
            clone.set_value( 'settest.list.notanumber', 'kitten', appendlists=True )
        with pytest.raises( TypeError, match="Tried to add an integer field to a dict." ):
            clone.set_value( 'settest.0', 'puppy' )
        with pytest.raises( TypeError, match="Tried to add an integer field to a dict." ):
            clone.set_value( 'settest.0.subset', 'bunny' )
        with pytest.raises( TypeError, match="Tried to add an integer field to a dict." ):
            clone.set_value( 'settest.dict.0', 'iguana' )
        with pytest.raises( TypeError, match="Tried to add an integer field to a dict." ):
            clone.set_value( 'settest.dict.2.something', 'tarantula' )

        clone.set_value( 'settest.list.0', 'mouse', appendlists=True )
        assert clone.value('settest.list.2') == 'mouse'
        assert cfg.value('settest.list') == [ 'a', 'b' ]
        clone.set_value( 'settest.list.5', 'mongoose' )
        assert clone.value('settest.list') == [ 'mongoose' ]
        assert cfg.value('settest.list') == [ 'a', 'b' ]

        clone.set_value( 'settest.dict.newkey', 'newval' )
        assert clone.value( 'settest.dict' ) == { 'key1': 'val1',
                                                'key2': 'val2',
                                                'newkey': 'newval' }
        assert 'newkey' not in cfg.value( 'settest.dict' )
        assert clone.value( 'settest.dict.newkey' ) == 'newval'

        clone.set_value( 'settest.dict2', 'scalar' )
        assert clone.value('settest.dict2') == 'scalar'
        assert cfg.value( 'settest.dict2' ) == { 'key1': '2val1', 'key2': '2val2' }

        clone.set_value( 'settest.scalar', 'notathing' )
        assert clone.value('settest.scalar') == 'notathing'
        assert cfg.value( 'settest.scalar' ) == 'thing'

        clone.set_value( 'settest.scalar.thing1', 'thing1' )
        clone.set_value( 'settest.scalar.thing2', 'thing2' )
        assert clone.value('settest.scalar') == { 'thing1': 'thing1', 'thing2': 'thing2' }
        assert cfg.value( 'settest.scalar' ) == 'thing'

        clone.set_value( 'settest.scalar2.0.key', "that wasn't a scalar" )
        assert clone.value('settest.scalar2') == [ { "key": "that wasn't a scalar" } ]
        assert cfg.value( 'settest.scalar2' ) == 'foobar'

        clone.set_value( 'totallynewvalue.one', 'one' )
        clone.set_value( 'totallynewvalue.two', 'two' )
        assert clone.value('totallynewvalue') == { 'one': 'one', 'two': 'two' }
        with pytest.raises( ValueError, match="Field totallynewvalue doesn't exist" ):
            _ = cfg.value( 'totallynewvalue' )
