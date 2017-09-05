a = float(input('first number: '))
b = float(input('second number: '))
op = input('operator(+, -, *, /, **): ')
if op == '+':
    result = a + b
elif op == '-':
    result = a - b
elif op == '*':
    result = a * b
elif op == '/':
    result = a / b
elif op == '**':
    result = a ** b
else:
    result = None


print('{} {} {} = {}'.format(a, op, b, result))
