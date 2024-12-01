import pandas as pd
from itertools import chain
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score

life_expectancy = pd.read_excel("./xlsx/data.xlsx", sheet_name="Данные", header=None, skiprows=lambda x: x in chain(range(20), [i for i in range(19, 976) if i not in range(19, 976, 9)], [190, 217, 307, 316, 325, 406, 478, 550, 622, 649, 694, 703, 757, 766, 784, 829, 838, 865, 892, 964, 973]), usecols="AL", names=["life_expectancy"])
min_cost_of_living = pd.read_excel("./xlsx/vpm_sub_2021-2025.xlsx", sheet_name="2023", header=None, skiprows=lambda x: x in chain(range(8), [26, 29, 30, 40, 49, 57, 72, 75, 76, 81, 92, 104], range(105, 113)), usecols="B", names=["min_cost"])

def swap_rows(df, i1, i2):
	a, b = df.iloc[i1, :].copy(), df.iloc[i2, :].copy()
	df.iloc[i1, :], df.iloc[i2, :] = b, a
	return df
life_expectancy = swap_rows(life_expectancy, 75, 76)

X = sm.add_constant(life_expectancy, prepend=False)
Y = min_cost_of_living
mod = sm.OLS(Y, X)
res = mod.fit()

pred = life_expectancy.map(lambda x: res.params["life_expectancy"] * x + res.params["const"]).rename(columns={"life_expectancy": "min_cost_pred"})
dataf = pd.concat([min_cost_of_living, life_expectancy], axis=1)

cor = dataf["life_expectancy"].corr(dataf["min_cost"])
r2 = r2_score(y_true=dataf["min_cost"], y_pred=pred["min_cost_pred"])

ax = sns.scatterplot(data=dataf, x="life_expectancy", y="min_cost")
ax.annotate(f'Коэффициент корреляции: {cor:.5f} |=> мера связи - слабая', xy=(0.01, 0.9), xycoords='axes fraction', color='blue', fontstyle='italic')
ax.annotate(f'Достоверность аппроксимации R^2: {r2:.5f}', xy=(0.01, 0.85), xycoords='axes fraction', color='red', fontstyle='italic')
sns.lineplot(x=dataf["life_expectancy"], y=pred["min_cost_pred"])
plt.show()