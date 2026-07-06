# %%
import pandas as pd

df1 = pd.read_csv("/home/gergo/projects/SS26_ml_prozess/data/gt_2011.csv")
df2 = pd.read_csv("/home/gergo/projects/SS26_ml_prozess/data/gt_2012.csv")
df3 = pd.read_csv("/home/gergo/projects/SS26_ml_prozess/data/gt_2013.csv")
df4 = pd.read_csv("/home/gergo/projects/SS26_ml_prozess/data/gt_2014.csv")
df5 = pd.read_csv("/home/gergo/projects/SS26_ml_prozess/data/gt_2015.csv")
dfs = [df1, df2, df3, df4, df5]
df = pd.concat(dfs, axis=0)
# timestamps = pd.date_range(start="2011-01-01", periods=len(df), freq="h")
# df["timestamp"] = timestamps
# df.to_csv("/home/gergo/projects/SS26_ml_prozess/data/gt_total.csv")
target_columns = ("NOX",)
cols = df.columns
print(len([col for col in cols if col not in target_columns]))
# %%

("_".join(target_columns))
