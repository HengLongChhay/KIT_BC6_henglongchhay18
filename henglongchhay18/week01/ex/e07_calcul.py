sum1 = 0
num1 = 0
num = ""
while num != "exit" and num != "EXIT":
    num = input("Enter a number:\n>>")
    if num == "":
        num1 = 0
    elif num.isalpha():
        num1 = 0
    else:
        num1 = num
    sum1 = int(sum1) + int(num1)
    print("TOTAL: "+str(sum1))

