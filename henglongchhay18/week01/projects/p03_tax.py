amount = input("Please enter your amount:\n>>")
while True:
    if float(amount) % 1 != 0 or float(amount) < 0:
        amount = input("Number is incorrect, try again.\nPlease enter your amount:\n>>")
    elif float(amount) % 1 == 0:
        break
    elif amount == "" or amount.isalpha():
        amount = input("Number is incorrect, try again.\nPlease enter your amount:\n>>")
rate = input("Please enter tax rate:\n>>")
while True:
    if float(rate) % 1 != 0 or float(rate) < 0:
        rate = input("Rate is incorrect, try again.\nPlease enter tax rate:\n>>")
    elif float(amount) % 1 == 0:
        break
    elif rate == "" or amount.isalpha():
        rate = input("Number is incorrect, try again.\nPlease enter your amount:\n>>")
TAX = int(amount)*int(rate)/100
'{:.2f}'.format(TAX)
NET = int(amount) - float(TAX)
'{:.2f}'.format(NET)
print("===== ===== ===== ===== =====\nAMOUNT: "+amount+"\nRATE: "+rate+"%\n===== ===== ===== ===== =====\nTAX: "+str(TAX)+"\nNET: "+str(NET)+"\n===== ===== ===== ===== =====")