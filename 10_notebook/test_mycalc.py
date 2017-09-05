from mycalc import calc_two_terms

def test_calc_two_terms():
    assert calc_two_terms(3, 5, '+') == 8
    assert calc_two_terms(3, 5, '-') == -2
    assert calc_two_terms(3, 5, '*') == 17
    assert calc_two_terms(3, 5, '/') == 0.6
    assert calc_two_terms(3, 5, '**') == 243
    assert calc_two_terms(3, 5, '$') == None  
