list1 = []
i = 0
J = 0
string = ""
while string != "GEN" and string != "gen":
    string = input("Enter a string:\n>>")
    list1.append(string)
    i = i + 1
for j in range(0, i-1):
    print("<p>"+str(list1[j])+"</p>")


