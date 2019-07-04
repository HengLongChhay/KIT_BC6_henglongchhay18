import random
i = 1
j = 0
name = input("Hello, what is your name?\n>>")
number_in_mind = random.randint(1, 88)
print(number_in_mind)
y_n = ""
guess_number = input("Well " + name + ", try to guess the number I have in mind!\n>>")
while True:
    if i == 1:
        if int(number_in_mind) == int(guess_number):
            print("You won in 1 turn only, that’s amazing!")
            i += 1
            while y_n != "y" and y_n != "Y" and y_n != "N" and y_n != "n":
                if j == 0:
                    y_n = input("Do you want to play again? [Y/N]\n>>")
                if y_n == "n" or y_n == "N":
                    print("Ok, bye "+name+"! See you later!")
                    exit()
                elif y_n.isalpha() or y_n == "" or y_n.isnumeric():
                    print("Sorry, I did not understand. Let me repeat:")
                    y_n = input("Do you want to play again? [Y/N]\n>>")
                    j += 1
                if y_n == "y" or y_n == "Y":
                    j = 0
                    i = 1
                    name = input("Hello, what is your name?\n>>")
                    number_in_mind = random.randint(1, 88)
                    print(number_in_mind)
                    y_n = ""
                    guess_number = input("Well " + name + ", try to guess the number I have in mind!\n>>")
                    break
        elif 1 > int(guess_number) or int(guess_number) > 88:
            guess_number = input("Invalid number, USAGE: 1-88, try again!\n>>")
        elif int(number_in_mind) > int(guess_number) >= 1:
            guess_number = input("Too low, try again!\n>>")
            i += 1
        elif int(guess_number) <= 88 and int(number_in_mind) < int(guess_number):
            guess_number = input("Too high, try again!\n>>")
            i += 1
    else:
        if int(number_in_mind) == int(guess_number):
            print("It took you "+str(i)+" turns to guess my number which was "+str(number_in_mind)+"!")
            i += 1
            while y_n != "y" and y_n != "Y" and y_n != "N" and y_n != "n":
                if j == 0:
                    y_n = input("Do you want to play again? [Y/N]\n>>")
                if y_n == "n" or y_n == "N":
                    print("Ok, bye "+name+"! See you later!")
                    exit()
                elif y_n.isalpha() or y_n == "" or y_n.isnumeric():
                    print("Sorry, I did not understand. Let me repeat:")
                    y_n = input("Do you want to play again? [Y/N]\n>>")
                    j += 1
                if y_n == "y" or y_n == "Y":
                    i = 1
                    j = 0
                    name = input("Hello, what is your name?\n>>")
                    number_in_mind = random.randint(1, 88)
                    print(number_in_mind)
                    y_n = ""
                    guess_number = input("Well " + name + ", try to guess the number I have in mind!\n>>")
                    break
        elif 1 > int(guess_number) or int(guess_number) > 88:
            guess_number = input("Invalid number, USAGE: 1-88, try again!\n>>")
        elif int(number_in_mind) > int(guess_number) >= 1:
            guess_number = input("Too low, try again!\n>>")
            i += 1
        elif int(guess_number) <= 88 and int(number_in_mind) < int(guess_number):
            guess_number = input("Too high, try again!\n>>")
            i += 1

