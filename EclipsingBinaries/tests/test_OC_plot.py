from .OC_plot import calculate_oc


def test_calculate_oc():
    # Test case 1: Test when T0 = 0
    m = 10.0
    err = 0.1
    T0 = 0
    T0_err = 0.1
    p = 1.0
    e, OC, OC_err, T0, T0_err = calculate_oc(m, err, T0, T0_err, p)
    assert e == 10.0
    assert OC == "0.00000"
    assert OC_err == "0.14142"

    # Test case 2: Test when T0 != 0
    m = 15.0
    err = 0.1
    T0 = 5.0
    T0_err = 0.1
    p = 2.0
    e, OC, OC_err, T0, T0_err = calculate_oc(m, err, T0, T0_err, p)
    assert e == 5.0
    assert OC == "0.00000"
    assert OC_err == "0.14142"
