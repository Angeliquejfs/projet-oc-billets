# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Bonjour, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
New_variable=50
nouvelle_variable=3

# Fonction d'un nombre qui multiplie par 10

def multiply_by_ten(number):
    """
    Cette fonction multiplie un nombre par 10.
    si paire renvoie 10
    si impaire renvoie 5

    :param number: Le nombre à multiplier.
    :return: Le résultat de la multiplication.
    """
    if number % 2 == 0:
        return number * 10
    else:
        return number * 5

multiply_by_ten(20)






