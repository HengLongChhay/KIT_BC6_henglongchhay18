string = input("Enter a string:\n>>")
if string == "":
    print("The string is empty.")
else:
    print(string[len(string)-1])