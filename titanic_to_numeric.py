# titanic_to_numeric.py

# Converts the nonnumeric data in the titanic datasets to numerical form.
#
# Note: This requires pandas which is not included with the base Python distribution.

import pandas as pd

def strings_to_nums(df):

    def sex_to_num(string):
        if string == 'male':
            return -1
        elif string == 'female':
            return 1

    def age_to_num(string):
        if string == 'UNDER 21':
            return -1
        elif string == '21 to 39':
            return 0
        elif string == 'OVER 40':
            return 1

    def embarked_to_num(string):
        if string == 'C':
            return -1
        elif string == 'Q':
            return 0
        elif string == 'S':
            return 1

    def boat_to_num(boolean):
        if boolean == False:
            return -1
        elif boolean == True:
            return 1

    def body_to_num(boolean):
        if boolean == False:
            return -1
        elif boolean == True:
            return 1

    df.sex = df.sex.apply(sex_to_num)
    df.age = df.age.apply(age_to_num)
    df.embarked = df.embarked.apply(embarked_to_num)
    df.boat = df.boat.apply(boat_to_num)
    df.body = df.body.apply(body_to_num)
    data = df.as_matrix()
    return data

training_df = pd.read_csv('titanic_train.csv', encoding='utf_8_sig')
training_data = strings_to_nums(training_df)

testing_df = pd.read_csv('titanic_test.csv', encoding='utf_8_sig')
testing_data = strings_to_nums(testing_df)

training_df.to_csv('titanic_numeric_train.csv', index=False)
testing_df.to_csv('titanic_numeric_test.csv', index=False)
