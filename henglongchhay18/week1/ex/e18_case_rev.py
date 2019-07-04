string = ""
input1 = input("Enter a string:\n>>")
if input1 == "":
    print("Empty")
else:
    for i in range(0, len(input1)):
        if "z" >= input1[i] >= "a":
            string += input1[i].upper()
        else:
            string += input1[i].lower()
print(string)
