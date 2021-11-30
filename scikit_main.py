import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def main():
    data = pd.read_csv('housing_price_init.csv')
    print(data.shape)
    y = data['price']
    x = data.drop('price', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    # x_train = x_train.reshape(-1,1)
    lm = linear_model.LinearRegression()
    lm.fit(X=x_train, y=y_train)

    print("截距：", lm.coef_)
    print("斜率：", lm.intercept_)

    draw_result(lm.intercept_, lm.coef_, data)


def draw_result(o0, o1, data):
    x = np.linspace(20, 2000, 40)
    y = o0 + x * o1
    plt.plot(x, y, color='red')
    draw_dot(data)


def draw_dot(data):
    # draw
    plt.scatter(data['area'], data['price'])
    plt.xlabel('area')
    plt.ylabel('price')
    plt.show()


if __name__ == '__main__':
    main()
