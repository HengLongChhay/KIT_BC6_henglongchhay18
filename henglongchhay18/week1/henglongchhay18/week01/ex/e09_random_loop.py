import random
input1 = input("Enter a number:")
if float(input1) % 1 == 0:
    for i in range(1, int(input1)+1):
        print(random.randint(1, 101))
else:
    print(random.randint(1, 101))