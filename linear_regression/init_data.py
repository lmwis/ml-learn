
import pandas as pd
import random


def main():
    target_o1 = 10.89
    target_o0 = 324.12
    xs=[]
    ys=[]
    for i in range(20,2000,2):
        param = random.uniform(-2, 2)
        o1_float = round(param, 2)
        print(o1_float)
        xs.append(i)
        ys.append(round(cal_y(target_o0,target_o1+o1_float,i),2))

    data={'area':xs,'price':ys}
    df = pd.DataFrame(data=data)
    print(df.head())
    pd.DataFrame.to_csv(df, path_or_buf='housing_price_init.csv', index=None)

def cal_y(o0,o1,x):
    print(o1)
    return o0+x*o1

if __name__ == '__main__':
    main()