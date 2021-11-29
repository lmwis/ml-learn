import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    columns = ['area', 'price']
    # data = pd.read_csv("housing_price.csv",header=None,delimiter=r"\s+",names=columns)

    data = pd.read_csv("housing_price_init.csv")
    print(data.head())
    print(data.describe())
    draw_dot(data)

    # init
    a = 0.0000003
    o0 = 1
    o1 = 0
    sum1 = 0
    sum2 = 0
    m = len(data)
    # middle value
    os0=[]
    os1=[]
    jo=[]
    # start 梯度下降
    for times in range(100):
        # 对oj的微分
        for index,row in data.iterrows():
            sum1 += cal_y(row['area'],o0,o1) - row['price']
            sum2 += (cal_y(row['area'],o0,o1) - row['price']) * row['area']
        o0 = o0 - a * sum1 / m
        o1 = o1 - a * sum2 / m
        # init
        sum1 = 0
        sum2 = 0
        os0.append(o0)
        os1.append(o1)
        # cost function
        jo_sum = 0
        for index,row in data.iterrows():
            jo_sum += pow(cal_y(row['area'],o0,o1) - row['price'],2)
        jo.append(jo_sum/2/m)
    print("o0:",o0)
    print("o1:",o1)
    print(cal_y(13,o0,o1))
    draw_o(os0,os1)
    draw_jo(os0,os1,jo)
    draw_result(o0,o1,data)

def draw_result(o0,o1,data):
    x = np.linspace(20,2000,40)
    y = o0+x*o1
    plt.plot(x,y,color='red')
    draw_dot(data)

def draw_o(os0,os1):
    plt.plot(range(0,len(os0)),os0,color='red')
    plt.plot(range(0,len(os1)),os1,color='blue')
    plt.xlabel('times')
    plt.ylabel('thta')
    plt.show()

def draw_jo(o0,o1,y):
    # 定义坐标轴
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(o0, o1, y)  # 绘制散点图
    plt.show()

def draw_dot(data):
    # draw
    plt.scatter(data['area'], data['price'])
    plt.xlabel('area')
    plt.ylabel('price')
    plt.show()

def cal_y(x,o0,o1):
    return o0+o1*x
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

