num = 0
while True:
    num = input("Enter a number:\n>>")
    if str(num) == "exit" or str(num) == "EXIT":
        break
    elif float(num) % 2 == 0:
        print(num+" is EVEN")
    elif float(num) % 2 == 1:
        print(num+" is ODD")
    elif float(num) % 2 == -1:
        print(num+" is ODD")
    else:
        print(num+" is not a valid number")

