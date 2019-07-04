output = ""
j = 0
input1 = input("Enter your secret message:\n>>")
if input1 == "":
    print("Nothing to encode.")
else:
    while j < len(input1):
        if input1[j].isspace():
            output += input1[j]
        elif input1[j].isupper():
            asc = ord(input1[j]) + 13
            asc = asc % 91
            if asc < 65:
                asc += 65
            input2 = chr(asc)
            output += input2
        elif input1[j].islower():
            asc = ord(input1[j]) + 13
            asc = asc % 123
            if asc < 97:
                asc += 97
            input2 = chr(asc)
            output += input2
        j = j + 1
print(output)
