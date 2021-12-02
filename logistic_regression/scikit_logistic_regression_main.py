
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


def main():
    data = pd.read_csv('train.csv')
    print(data.shape)
    print(data['SibSp'].head())
    print(data.columns)


    # test_data = pd.DataFrame(data,columns=['Survived','Pclass','Sex'])
    # draw_dot(test_data)
    used_feature = ['Pclass', 'Sex', 'Age','Embarked']
    # data = data_clean(data)
    # print(data['Sex'].head())
    build_module(data,used_feature)

    # cabin_data = pd.DataFrame(data, columns=['Survived', 'Cabin'])
    # print(cabin_data.head())
    # print(cabin_data.describe())
    #
    # embarked_data = pd.DataFrame(data, columns=['Survived', 'Embarked'])
    # print(embarked_data.head())
    # print(embarked_data.describe())
    #
    # print(embarked_data.columns)
    # embarked_data = data_clean(embarked_data)


    #
    # analyze_feature(data)



def analyze_feature(data):
    class_data = pd.DataFrame(data, columns=['Survived', 'Pclass'])
    sex_data = pd.DataFrame(data, columns=['Survived', 'Sex'])
    sibsp_data = pd.DataFrame(data, columns=['Survived', 'SibSp'])
    parch_data = pd.DataFrame(data, columns=['Survived', 'Parch'])
    age_data = pd.DataFrame(data, columns=['Survived', 'Age'])
    ticket_data = pd.DataFrame(data, columns=['Survived', 'Ticket'])
    fare_data = pd.DataFrame(data, columns=['Survived', 'Fare'])
    cabin_data = pd.DataFrame(data, columns=['Survived', 'Cabin'])
    embarked_data = pd.DataFrame(data, columns=['Survived', 'Embarked'])
    # draw_sibsp_data_bar(sibsp_data)
    # draw_discrete_feature_to_survived_data_bar(sibsp_data,'SibSp')
    # draw_discrete_feature_to_survived_data_bar(class_data,'Pclass')
    # draw_discrete_feature_to_survived_data_bar(sex_data,'Sex')
    # draw_discrete_feature_to_survived_data_bar(parch_data,'Parch')
    # draw_discrete_feature_to_survived_data_bar(age_data,'Age')
    # draw_discrete_feature_to_survived_data_bar(ticket_data,'Ticket')
    # draw_discrete_feature_to_survived_data_bar(fare_data,'Fare')
    # draw_discrete_feature_to_survived_data_bar(embarked_data,'Embarked')
    # draw_discrete_feature_to_survived_data_bar(cabin_data,'Cabin')

    draw_discrete_feature_to_survived_rate_line(sibsp_data, 'SibSp')  # not sure
    draw_discrete_feature_to_survived_rate_line(class_data, 'Pclass')  # sure
    draw_discrete_feature_to_survived_rate_line(sex_data, 'Sex')  # sure
    draw_discrete_feature_to_survived_rate_line(parch_data, 'Parch')  # not sure
    # draw_discrete_feature_to_survived_rate_line(ticket_data,'Ticket')
    draw_discrete_feature_to_survived_rate_line(embarked_data,'Embarked') # sure
    # draw_discrete_feature_to_survived_rate_line(cabin_data,'Cabin')

    draw_continuous_feature_to_survived_data_line(age_data, 'Age')
    draw_continuous_feature_to_survived_data_line(fare_data, 'Fare')
    # draw_continuous_feature_to_survived_data_line(ticket_data, 'Ticket')

def build_module(train_data,used_feature):
    data = train_data
    # pick feature 'Pclass' and 'Sex' for logistic regression
    train_X = pd.DataFrame(data, columns=used_feature)
    train_X = data_clean(train_X)
    print(train_X.isna)
    # checkout if has nan value
    # print(train_X[train_X['Age'].isnull().values==True].head())
    y = pd.DataFrame(data, columns=['Survived'])
    lr = LogisticRegression(random_state=0).fit(train_X, y)

    # use module to predict test data
    test_data = pd.read_csv('test.csv')
    test_X = pd.DataFrame(test_data, columns=used_feature)
    test_X = data_clean(test_X)
    output_y = lr.predict(test_X)

    # save result
    result = pd.DataFrame(test_data, columns=['PassengerId'])
    result['Survived'] = pd.Series(output_y)
    print(result.head())

    pd.DataFrame.to_csv(result, path_or_buf='result.csv', index=False)
    print('输出结果完成')

def data_clean(data):
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({
            'male': 1,
            'female': 0
        })
    # fill the missing age with mean value
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(value=data['Age'].mean())

    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna(value=data['Embarked'].value_counts().idxmax())
        data['Embarked'] = translate_enum(data['Embarked'])
    return data

def translate_enum(series):
    count = series.unique()
    index = 0
    for i in count:
        index += 1
        series = series.replace(i,index)
    return series

def draw_dot(data):
    survived_0 = data[data['Survived'] == 0]
    survived_1 = data[data['Survived'] == 1]
    plt.plot(survived_0['Pclass'],survived_0['Sex'],'o',color='r')
    plt.plot(survived_1['Pclass'],survived_1['Sex'],'o',color='b')
    plt.xlabel('Pclass')
    plt.ylabel('Sex')
    plt.show()

# draw survived line for continuous value
def draw_continuous_feature_to_survived_data_line(feature_data,feature_name):
    feature = sorted(feature_data[feature_name].unique())
    survived_0 = feature_data[feature_data['Survived'] == 0]
    survived_1 = feature_data[feature_data['Survived'] == 1]
    survived_count_0 = survived_0[feature_name].value_counts().sort_index()
    survived_count_1 = survived_1[feature_name].value_counts().sort_index()
    survived_rate = survived_count_1/(survived_count_1+survived_count_0)
    # plt.plot(survived_count_0.index,survived_count_0, 's-',label='no_survived', color='r')
    # plt.plot(survived_count_1.index,survived_count_1, 'o-',label='survived', color='b')
    plt.plot(survived_rate.index,survived_rate, 'o-',label='survived', color='g')

    plt.legend()
    plt.xlabel(feature_name)
    plt.ylabel('nums')
    plt.show()


# draw survived rate bar for discrete value
def draw_discrete_feature_to_survived_rate_line(feature_data,feature_name):
    feature = sorted(feature_data[feature_name].unique())
    survived_0 = feature_data[feature_data['Survived'] == 0]
    survived_1 = feature_data[feature_data['Survived'] == 1]
    survived_count_0 = survived_0[feature_name].value_counts().sort_index()
    survived_count_1 = survived_1[feature_name].value_counts().sort_index()
    survived_rate = survived_count_1/(survived_count_1+survived_count_0)
    # 数据补齐
    for i in feature:
        if i not in survived_count_0:
            survived_count_0[i] = 0
        if i not in survived_count_1:
            survived_count_1[i] = 0
    bar_weight = 0.3
    x_index = np.arange(len(feature))
    plt.plot(survived_rate.index, survived_rate, label='survived', color='r')
    plt.legend()
    plt.xticks(x_index + bar_weight / 2, feature)
    plt.xlabel(feature_name)
    plt.ylabel('nums')
    plt.show()

# draw survived bar for discrete value
def draw_discrete_feature_to_survived_data_bar(feature_data,feature_name):
    feature = sorted(feature_data[feature_name].unique())
    survived_0 = feature_data[feature_data['Survived'] == 0]
    survived_1 = feature_data[feature_data['Survived'] == 1]
    survived_count_0 = survived_0[feature_name].value_counts().sort_index()
    survived_count_1 = survived_1[feature_name].value_counts().sort_index()
    # 数据补齐
    for i in feature:
        if i not in survived_count_0:
            survived_count_0[i] = 0
        if i not in survived_count_1:
            survived_count_1[i] = 0
    bar_weight = 0.3
    x_index = np.arange(len(feature))
    plt.bar(x_index, survived_count_0, label='no_survived', color='r', width=bar_weight)
    plt.bar(x_index + bar_weight, survived_count_1, label='survived', color='b', width=bar_weight,
            tick_label=feature_name)
    plt.legend()
    plt.xticks(x_index + bar_weight / 2, feature)
    plt.xlabel(feature_name)
    plt.ylabel('nums')
    plt.show()
    # 验证数据是否都显示
    sum = 0
    for i in feature:
        sum+=survived_count_1[i]+survived_count_0[i]
        print(i,feature_name, "survived rate:", survived_count_1[i] / (survived_count_1[i] + survived_count_0[i]))
    print('total record:',sum)
if __name__ == '__main__':
    main()