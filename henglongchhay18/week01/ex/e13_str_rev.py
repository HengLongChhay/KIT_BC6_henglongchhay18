string = input("Enter a string:\n>>")
if string == "":
    print("The string is empty.")
else:
    string = "".join(reversed(string))
    print(string)