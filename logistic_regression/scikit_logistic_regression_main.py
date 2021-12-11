
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn.preprocessing as preprocessing


def main():
    pd.set_option('display.width', 130)
    pd.set_option('display.max_columns', 130)
    pd.set_option('display.max_colwidth', 130)
    data = pd.read_csv('train.csv')
    # print(data.shape)
    # print(data['SibSp'].head())
    # print(data.columns)


    # test_data = pd.DataFrame(data,columns=['Survived','Pclass','Sex'])
    # draw_dot(test_data)
    used_feature = ['Pclass', 'Sex', 'Age','Embarked','SibSp','Parch']
    used_feature_regex = 'Age_.*|Pclass_.*|Cabin_.*|Embarked_.*|SibSp|Parch|Fare_.*|Sex_.*'
    # print(data['Cabin'].info())
    data = data_clean(data)

    lr = build_module_regex(data,used_feature_regex)

    used_feature = pd.DataFrame(data).filter(regex=used_feature_regex).columns.values
    # print(data[data['Cabin'].notnull()].value_counts())
    # data = set_cabin_type(data)
    # print(data['Cabin'].head())
    # print(data['Sex'].head())
    # lr = build_module(data,used_feature)

    theta = pd.DataFrame({"colums":used_feature,"coef":list(lr.coef_.T)})
    print(theta)

    # view data with graph
    # analyze_feature(data)

# scaling feature to [-1,1]
def scaling_data(data,feature):
    scaler = preprocessing.StandardScaler()
    for i in feature:
        param = scaler.fit(data[i].values.reshape(-1,1))
        data[i+'_scaled'] = scaler.fit_transform(data[i].values.reshape(-1,1),param)
    return data
# change 'Cabin' feature to 'Yes' or 'No'

# set missing age whit random forest
def set_missing_ages(data):
    age_data = pd.DataFrame(data,columns=['Age','Fare','Parch','SibSp','Pclass'])
    known_age = age_data[age_data['Age'].notnull()].values
    unknown_age = age_data[age_data['Age'].isnull()].values

    y = known_age[:,0] # 年龄列
    X = known_age[:,1:] # 其他特征列
    # print(age_data.value_counts())
    # 随机森林进行拟合
    rf = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rf.fit(X,y)
    # 进行预测
    predict_age = rf.predict(unknown_age[:,1::])
    data.loc[data['Age'].isnull(),'Age'] = predict_age
    return data

def set_cabin_type(data):
    data.loc[data['Cabin'].notnull(),'Cabin']='Yes'
    data.loc[data['Cabin'].isnull(),'Cabin']='No'
    return data
def set_one_hot_encode(data):
    dummies_Cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')

    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')

    data = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    data.drop(['Cabin', 'Name', 'Ticket', 'Embarked', 'Sex', 'Pclass'], axis=1, inplace=True)
    return data
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
    draw_continuous_feature_to_survived_data_line(fare_data, 'Fare') # not sure
    # draw_continuous_feature_to_survived_data_line(ticket_data, 'Ticket')

# use regex to pick feature
def build_module_regex(train_data,regex):
    used_feature = pd.DataFrame(train_data).filter(regex=regex).columns.values
    return build_module(train_data,used_feature)

def build_module(train_data,used_feature):
    data = train_data
    # pick feature 'Pclass' and 'Sex' for logistic regression
    # use filer to pick feature
    train_X = pd.DataFrame(data, columns=used_feature)
    # train_X = data_clean(train_X)
    # checkout if has nan value
    # print(train_X[train_X['Age'].isnull().values==True].head())
    y = pd.DataFrame(data, columns=['Survived'])
    lr = LogisticRegression(random_state=0).fit(train_X, y)

    # use module to predict test data
    test_data = pd.read_csv('test.csv')
    test_X = data_clean(test_data)
    test_X = pd.DataFrame(test_X, columns=used_feature)
    # test_X = set_cabin_type(test_X)
    output_y = lr.predict(test_X)

    # save result
    result = pd.DataFrame(test_data, columns=['PassengerId'])
    result['Survived'] = pd.Series(output_y)

    pd.DataFrame.to_csv(result, path_or_buf='result.csv', index=False)
    print('输出结果完成')

    return lr

def data_clean(data):

    # if 'Sex' in data.columns:
    #     data['Sex'] = data['Sex'].map({
    #         'male': 1,
    #         'female': 0
    #     })
    # fill the missing age with mean value
    # if 'Age' in data.columns:
    #     data['Age'] = data['Age'].fillna(value=data['Age'].mean())

    if 'Fare' in data.columns:
        data['Fare'] = data['Fare'].fillna(value=data['Fare'].mean())

    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna(value=data['Embarked'].value_counts().idxmax())
        # data['Embarked'] = translate_enum(data['Embarked'])
    print(data.head())
    data = set_cabin_type(data)
    data = set_missing_ages(data)
    data = set_one_hot_encode(data)
    data = scaling_data(data,['Age','Fare'])

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
    plt.ylabel('survived rate')
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