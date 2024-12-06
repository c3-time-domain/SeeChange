import pytest

import numpy as np
from pipeline.parameters import Parameters, ParsDemoSubclass


def test_parameters_override_augment():
    p = Parameters()
    p.add_par('int', 1, int, 'an integer')
    p.add_par('float', 2.0, [int, float], 'a float')
    p.add_par('str', 'hello', str, 'a string')
    p.add_par('bool', True, bool, 'a boolean')
    p.add_par('list', [1, 2, 3], list, 'a list')
    p.add_par('dict', {'a': 1, 'b': 2}, dict, 'a dict')
    p.add_par('set', {1, 2, 3}, set, 'a set')

    assert p.int == 1
    assert p['int'] == 1
    assert p.float == 2.0
    assert p['float'] == 2.0
    assert p.str == 'hello'
    assert p['str'] == 'hello'
    assert p.bool is True
    assert p['bool'] is True
    assert p.list == [1, 2, 3]
    assert p['list'] == [1, 2, 3]
    assert p.dict == {'a': 1, 'b': 2}
    assert p['dict'] == {'a': 1, 'b': 2}
    assert p.set == {1, 2, 3}

    # test not being able to set wrong types:
    with pytest.raises(TypeError, match='must be of type'):
        p.int = 3.0
    assert p.int == 1

    with pytest.raises(TypeError, match='must be of type'):
        p.float = 'hello'
    assert p.float == 2.0

    with pytest.raises(TypeError, match='must be of type'):
        p.str = 3
    assert p.str == 'hello'

    with pytest.raises(TypeError, match='must be of type'):
        p.bool = 3
    assert p.bool is True

    with pytest.raises(TypeError, match='must be of type'):
        p.list = 3
    assert p.list == [1, 2, 3]

    with pytest.raises(TypeError, match='must be of type'):
        p.dict = 3
    assert p.dict == {'a': 1, 'b': 2}

    with pytest.raises(TypeError, match='must be of type'):
        p.set = 3
    assert p.set == {1, 2, 3}

    # turn off type checks:
    p._enforce_type_checks = False

    p.int = 2.5
    assert p.int == 2.5

    # use override:
    p.override(dict(int=2, float=3.0, str='world', bool=False, list=[4, 5, 6], dict={'c': 3, 'd': 4}, set={4, 5, 6}))
    assert p.int == 2
    assert p.float == 3.0
    assert p.str == 'world'
    assert p.bool is False
    assert p.list == [4, 5, 6]
    assert p.dict == {'c': 3, 'd': 4}
    assert p.set == {4, 5, 6}

    # use augment:
    p.augment(dict(int=3, float=4.0, str='!', bool=True, list=[7, 8, 9], dict={'e': 5, 'f': 6}, set={7, 8, 9}))
    assert p.int == 3
    assert p.float == 4.0
    assert p.str == '!'
    assert p.bool is True
    assert p.list == [7, 8, 9]
    # note that dict and set are the only ones that gets merged:
    assert p.dict == {'c': 3, 'd': 4, 'e': 5, 'f': 6}
    assert p.set == {4, 5, 6, 7, 8, 9}


def test_parameters_read_kwargs():
    # free form parameters, can add them on the fly
    # (but this will not log them using add_par)
    p = Parameters(foo='bar', baz=42)
    assert p.foo == 'bar'
    assert p.baz == 42

    # this parameters subclass has specific parameters defined:
    p = ParsDemoSubclass(int=2, float=3.0, plot=False, _secret=42, null=None)
    assert p.integer_parameter == 2
    assert p.float_parameter == 3.0
    assert p.plotting_value is False
    assert p._secret_parameter == 42
    assert p.nullable_parameter is None

    with pytest.raises(AttributeError, match='has no attribute'):
        ParsDemoSubclass(foo='bar')


def test_parameter_name_matches():
    # cannot initialize parameters subclass (which is locked) with wrong key
    with pytest.raises(AttributeError) as e:
        ParsDemoSubclass(wrong_key="test")
    assert "object has no attribute " in str(e)

    rng = np.random.default_rng()
    int_par = int( rng.integers(2, 10) )
    float_par = float( rng.uniform(2, 10) )

    p = ParsDemoSubclass(Int_Par=int_par, FLOAT_P=float_par, plot=False, null=None)
    p._remove_underscores = True

    assert p.IntPar == int_par
    assert p.integer_parameter == int_par
    assert p.FLOATP == float_par
    assert p.float_parameter == float_par
    assert p.plot is False
    assert p.plotting_value is False
    assert p.null is None
    assert p.nullable_parameter is None

    # try to set parameters using aliases with capitals
    p.INTEGER = int_par * 2
    assert p.integer_parameter == int_par * 2
    p.INT = int_par * 3
    assert p.integer_parameter == int_par * 3
    p.FLOAT = float_par * 2
    assert p.float_parameter == float_par * 2

    # an attribute not related to any of these parameters still fails
    with pytest.raises(AttributeError) as e:
        p.wrong_attribute = 1
    assert "object has no attribute " in str(e)

    # now turn off the partial matches
    p._allow_shorthands = False

    with pytest.raises(AttributeError) as e:
        p.integer = int_par * 2
    assert "object has no attribute " in str(e)

    with pytest.raises(AttributeError) as e:
        p.float = float_par * 2
    assert "object has no attribute " in str(e)

    # still works with capitals and no underscores
    p.IntegerParameter = int_par * 2
    assert p.integer_parameter == int_par * 2

    # still works with aliases
    p.INT_PAR = int_par * 3
    assert p.integer_parameter == int_par * 3

    # turn off case-insensitivity
    p._ignore_case = False

    with pytest.raises(AttributeError) as e:
        p.integer_parameteR = int_par * 2
    assert "object has no attribute" in str(e)

    # aliases still work
    p.int_par = int_par * 3
    assert p.integer_parameter == int_par * 3

    # underscores are still removed
    p.float___parameter = float_par * 2
    assert p.float_parameter == float_par * 2

    # turn partial matches back on:
    p._allow_shorthands = True
    p.float_ = float_par * 2
    assert p.float_parameter == float_par * 2

    # but capitals are still turned off
    with pytest.raises(AttributeError) as e:
        p.Float_Parameter = float_par * 2
    assert "object has no attribute" in str(e)

    # check critical parameters can be filtered out using to_dict
    keys = p.to_dict(critical=True, hidden=False).keys()
    keys == ["integer_parameter", "float_parameter", "nullable_parameter"]

    keys = p.to_dict(critical=True, hidden=True).keys()
    keys == [
        "integer_parameter",
        "float_parameter",
        "_secret_parameter",
        "nullable_parameter",
    ]

    keys = p.to_dict(critical=False, hidden=True).keys()
    keys == [
        "integer_parameter",
        "float_parameter",
        "_secret_parameter",
        "plotting_value",
        "nullable_parameter",
        "_enforce_no_new_attrs",
        "_allow_shorthands",
        "_ignore_case",
    ]
