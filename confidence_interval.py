import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from scipy import stats

general = pd.read_excel("./xlsx/оsn-pok-nauka_02-2024.xlsx", sheet_name="2", header=None, skiprows=chain(range(9), range(564, 582)), usecols="A:C", names=["Activity_name", "Id", "Count"]).drop_duplicates(subset=["Activity_name", "Count"])
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
group_index = [i for i in general.index if general["Id"].loc[i] in ascii_uppercase]

def random_selection(df):
	return df.loc[df.index % 3 == 0]

def confidence_interval(mx, sigma, n):
	answer = {}
	const = (sigma / np.sqrt(n))
	percentiles = [0.9, 0.95, 0.99]
	for per in percentiles:
		z = stats.norm.ppf(per, 0, 1)
		answer[f"{per * 100}%"] = [mx - z * const, mx + z * const]
	return answer

simple_sample = []
stratified_sample = []
for i in range(len(group_index)):
	per = general.loc[group_index[i]]["Count"] / general.loc[0]["Count"]
	if (per < 0.01):
		continue
	simple_sample.append(general.loc[(general.index >= group_index[i] + 1) & (general.index < group_index[i+1])])
	stratified_sample.append(random_selection(general.loc[(general.index >= group_index[i] + 1) & (general.index < group_index[i+1])].reset_index(drop=True)))
general_mean = np.mean(pd.concat(simple_sample).reset_index(drop=True)["Count"])
simple_sample = random_selection(pd.concat(simple_sample).reset_index(drop=True))
stratified_sample = pd.concat(stratified_sample).reset_index(drop=True)

simple_mean = np.mean(simple_sample["Count"])
stratified_mean = np.mean(stratified_sample["Count"])

simple_intervals = confidence_interval(simple_mean, np.var(simple_sample["Count"]), len(simple_sample))
stratified_intervals = confidence_interval(stratified_mean, np.var(stratified_sample["Count"]), len(stratified_sample))

print(f"general_mean: {general_mean:.3f}", f"simple_mean: {simple_mean:.3f}", f"simple_intervals: {simple_intervals}", f"stratified_mean: {stratified_mean:.3f}", f"stratified_intervals: {stratified_intervals}", sep="\n")