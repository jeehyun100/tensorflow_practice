# filename: mycalc.py

def calc_two_terms(a, b, op):
    ops = {'+':a+b, '-':a-b, '*':a*b, '/':a/b, '**':a**b}
    
    if op in ops:
        result = ops[op]
    else:
        print("'{}' operator is not supported.".format(op))
        result = None
        
    return result
    a = float(input('first number: '))
    b = float(input('second number: '))
    op = input('operator(+, -, *, /, **): ')

    result = calc_two_terms(a, b, op)
    print('{} {} {} = {}'.format(a, op, b, result))
    
    
if __name__ == '__main__':
    main()
