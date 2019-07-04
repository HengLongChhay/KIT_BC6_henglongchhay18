yn = ""
while yn != "exit" and yn != "EXIT":
    num = input("Enter a number:\n>>")
    if float(num) % 2 == 0:
        print(num+" is EVEN")
    elif float(num) % 2 == 1:
        print(num+" is ODD")
    elif float(num) % 2 == -1:
        print(num+" is ODD")
    else:
        print(num+" is not a valid number")
    yn = input(">>")
