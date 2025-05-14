import pandas as pd, glob

rows = []
for csv in glob.glob('runs/lambda_*/metrics.csv'):
    df = pd.read_csv(csv, names=['epoch','lam','dice','ter'])
    last = df.sort_values('epoch').iloc[-1]     # 最后一个 epoch
    rows.append(last)

df = pd.DataFrame(rows)
best = df.sort_values(['ter','dice'], ascending=[True, False]).iloc[0]

print(df[['lam','dice','ter']])
print(f"\n>>> 最优 λ = {best.lam}")