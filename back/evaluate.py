
import pandas as pd
import sklearn.linear_model

df = pd.read_csv('../test.csv')
df = df.dropna()

print(df.head(20))

rows = ['ARM', 'FOREARM', 'LEG', 'THIGH', 'SHOULDER', 'TRUNK', 'PELVIS']
rows_real = [x + '_REAL' for x in rows]

print(df[rows].std())

diff = pd.DataFrame(columns=rows, data=[])

for i in range(len(rows)):
    diff[rows[i]] = (df[rows_real[i]] - df[rows[i]]).abs()

print(diff)
print(diff.mean())

diff2 = pd.DataFrame(columns=rows, data=[])

for i in range(len(rows)):
    linear = sklearn.linear_model.LinearRegression(fit_intercept=False)
    linear = linear.fit(df[rows[i]].values.reshape((-1, 1)), df[rows_real[i]].values)
    diff2[rows[i]] = (df[rows_real[i]] - linear.predict(df[rows[i]].values.reshape((-1, 1)))).abs()
    print(rows[i], linear.coef_, linear.intercept_)

print(diff2)
print(diff2.mean())

per = pd.DataFrame(columns=rows, data=[])

for i in range(len(rows)):
    per[rows[i]] = diff2[rows[i]] / df[rows_real[i]]

print(per.mean())
