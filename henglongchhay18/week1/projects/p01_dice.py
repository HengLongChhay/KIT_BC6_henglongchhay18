import random
res = 0
input1 = input("Welcome to dice game!\nEnter the number of dice you want to roll:")
if input1.isalpha():
    print("Usage: The number must be between 1 and 8")
elif input1 == "":
    print("Usage: The number must be between 1 and 8")
elif 1 <= int(input1) <= 8:
    if int(input1) == 1:
        print("RESULT: "+str(random.randint(1, 6)))
    else:
        for i in range(1, int(input1) + 1):
            ran = random.randint(1, 6)
            res += int(ran)
            print("Dice "+str(i)+" : "+str(ran))
        print("==========\n"+"TOTAL: "+str(res)+"\n==========")
else:
    print("Usage: The number must be between 1 and 8")
