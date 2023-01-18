import math
number = 329073
check = True
for i in range(2, int(math.sqrt(329073))):
    if number % i == 0:
        print( i, 'no')
        check = False
        break

if check:
    print('yes')
