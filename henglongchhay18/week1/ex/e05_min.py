print("Enter a number:")
num1=input()
print("Enter second number:")
num2=input()
if num1<num2:
    print("Result: "+num1+" < "+num2)
elif num2<num1:
    print("Result: "+num2+" < "+num1)
else:
    print("Result: "+num2+" == "+num1)