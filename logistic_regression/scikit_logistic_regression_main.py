
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


def main():
    data = pd.read_csv('train.csv')
    print(data.shape)
    print(data.head())
    print(data.columns)

    # class_data = pd.DataFrame(data,columns=['Survived','Pclass'])
    # draw_class_data_bar(class_data)

    # sex_data = pd.DataFrame(data,columns=['Survived','Sex'])

    # draw_sex_data_bar(sex_data)

    # test_data = pd.DataFrame(data,columns=['Survived','Pclass','Sex'])
    # draw_dot(test_data)

    # pick feature 'Pclass' and 'Sex' for logistic regression
    train_X = pd.DataFrame(data,columns=['Pclass','Sex','Age'])
    train_X = data_clean(train_X)

    # checkout if has nan value
    # print(train_X[train_X['Age'].isnull().values==True].head())

    y = pd.DataFrame(data,columns=['Survived'])
    lr = LogisticRegression(random_state=0).fit(train_X,y)

    # use module to predict test data
    test_data = pd.read_csv('test.csv')
    test_X = pd.DataFrame(test_data,columns=['Pclass','Sex','Age'])
    test_X = data_clean(test_X)
    output_y = lr.predict(test_X)

    # save result
    result = pd.DataFrame(test_data,columns=['PassengerId'])
    result['Survived'] = pd.Series(output_y)
    print(result.head())

    pd.DataFrame.to_csv(result,path_or_buf='result.csv',index=False)
    print('输出结果完成')
def data_clean(data):
    data['Sex'] = data['Sex'].map({
        'male': 1,
        'female': 0
    })
    data['Age'] = data['Age'].fillna(value=data['Age'].mean())
    return data

def draw_dot(data):
    survived_0 = data[data['Survived'] == 0]
    survived_1 = data[data['Survived'] == 1]
    plt.plot(survived_0['Pclass'],survived_0['Sex'],'o',color='r')
    plt.plot(survived_1['Pclass'],survived_1['Sex'],'o',color='b')
    plt.xlabel('Pclass')
    plt.ylabel('Sex')
    plt.show()

# 明显3等舱的存活率更低，1等舱存活率最高
def draw_class_data_bar(class_data):
    x = sorted(class_data['Pclass'].unique())
    survived_0 = class_data[class_data['Survived'] == 0]
    survived_1 = class_data[class_data['Survived'] == 1]
    survived_count_0 = sorted(survived_0['Pclass'].value_counts())
    survived_count_1 = sorted(survived_1['Pclass'].value_counts())
    bar_weight = 0.3
    x_index = np.arange(len(x))
    plt.bar(x_index,survived_count_0,label='no_survived',color='r',width=bar_weight)
    plt.bar(x_index+bar_weight,survived_count_1,label='survived',color='b',width=bar_weight,tick_label=['1','2','3'])
    plt.legend()
    plt.xticks(x_index+bar_weight/2,x)
    plt.xlabel('Pclass')
    plt.ylabel('nums')
    plt.show()
    # 验证数据是否都显示
    print("1 class survived rate:",survived_count_1[0]/(survived_count_1[0]+survived_count_0[0]))
    print("2 class survived rate:",survived_count_1[1]/(survived_count_1[1]+survived_count_0[1]))
    print("3 class survived rate:",survived_count_1[2]/(survived_count_1[2]+survived_count_0[2]))
#
def draw_sex_data_bar(sex_data):
    x = sorted(sex_data['Sex'].unique())
    survived_0 = sex_data[sex_data['Survived'] == 0]
    survived_1 = sex_data[sex_data['Survived'] == 1]
    survived_count_0 = sorted(survived_0['Sex'].value_counts())
    survived_count_1 = sorted(survived_1['Sex'].value_counts())
    bar_weight = 0.3
    x_index = np.arange(len(x))
    plt.bar(x_index,survived_count_0,label='no_survived',color='r',width=bar_weight)
    plt.bar(x_index+bar_weight,survived_count_1,label='survived',color='b',width=bar_weight)
    plt.legend()
    plt.xticks(x_index+bar_weight/2,x)
    plt.xlabel('Pclass')
    plt.ylabel('nums')
    plt.show()
    # 验证数据是否都显示
    print("female survived rate:",survived_count_1[0]/(survived_count_1[0]+survived_count_0[0]))
    print("male survived rate:",survived_count_1[1]/(survived_count_1[1]+survived_count_0[1]))

if __name__ == '__main__':
    main()