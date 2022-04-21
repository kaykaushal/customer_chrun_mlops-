# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import pandas as pd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #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
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def check_os_dir():
    print(os.getcwd())
    print(os.path.abspath('./data'))
    df = pd.read_csv('./data/bank_data.csv')
    print(df.head())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    check_os_dir()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
