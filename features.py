import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("Transfer-Value-Predict/scouting_reports.csv")

# 2. Keep only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# 3. Drop columns with all NaN values
numeric_df = numeric_df.dropna(axis=1, how="all")

# 4. Impute missing values (mean)
imputer = SimpleImputer(strategy="mean")
numeric_df_imputed = pd.DataFrame(
    imputer.fit_transform(numeric_df),
    columns=numeric_df.columns
)

# 5. Separate predictors and target
y = numeric_df_imputed["value"]
X = numeric_df_imputed.drop(columns=["value"])

# 6. Remove highly correlated features
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.85
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

print(f"Highly correlated features to drop ({len(to_drop)}):")
print(to_drop)

X_reduced = X.drop(columns=to_drop)

print(f"Original features: {X.shape[1]}")
print(f"Reduced features: {X_reduced.shape[1]}")

# 7. Random Forest for feature importance
rf = RandomForestRegressor(random_state=42, n_estimators=500)
rf.fit(X_reduced, y)

importances = pd.DataFrame({
    "Feature": X_reduced.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# 8. Print top 20
print("\nTop 20 Important Features:")
print(importances.head(20))

# 9. Plot
plt.figure(figsize=(10,6))
plt.barh(importances["Feature"].head(20), importances["Importance"].head(20), color="steelblue")
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance (Gini)")
plt.title("Top 20 Important Features for Transfer Value")
plt.tight_layout()
plt.show()






# OUTPUT for old csv
# Highly correlated features to drop (46):
# ['Goals + Assists', 'Non-Penalty Goals', 'Penalty Kicks Attempted', 'xG: Expected Goals', 'npxG: Non-Penalty xG', 'npxG + xAG', 'Shots Total', 'Shots on Target', 'Non-Penalty Goals - npxG', 'Passes Attempted', 'Total Passing Distance', 'Progressive Passing Distance', 'Passes Completed (Short)', 'Passes Attempted (Short)', 'Passes Completed (Medium)', 'Passes Attempted (Medium)', 'Passes Attempted (Long)', 'xA: Expected Assists', 'Key Passes', 'Passes into Final Third', 'Live-ball Passes', 'Throw-ins Taken', 'Inswinging Corner Kicks', 'Outswinging Corner Kicks', 'Shot-Creating Actions', 'SCA (Live-ball Pass)', 'SCA (Dead-ball Pass)', 'GCA (Live-ball Pass)', 'Tackles Won', 'Dribblers Tackled', 'Dribbles Challenged', 'Challenges Lost', 'Tkl+Int', 'Touches', 'Touches (Def Pen)', 'Touches (Def 3rd)', 'Touches (Mid 3rd)', 'Touches (Att 3rd)', 'Touches (Att Pen)', 'Touches (Live-Ball)', 'Successful Take-Ons', 'Times Tackled During Take-On', 'Carries', 'Progressive Carrying Distance', 'Carries into Final Third', 'Passes Received']
# Original features: 108
# Reduced features: 62

# Top 20 Important Features:
#                       Feature  Importance
# 28              SCA (Take-On)    0.154713
# 32      Goal-Creating Actions    0.072014
# 15           Passes Completed    0.048400
# 21              Through Balls    0.042314
# 0                       Goals    0.039435
# 48    Total Carrying Distance    0.032585
# 49  Carries into Penalty Area    0.031334
# 61               Aerials Lost    0.028525
# 51               Dispossessed    0.025438
# 16    Passes Completed (Long)    0.021422
# 47         Take-Ons Attempted    0.020487
# 22                   Switches    0.019775
# 6         Progressive Carries    0.019269
# 31     SCA (Defensive Action)    0.019258
# 42                     Blocks    0.017777
# 54                Fouls Drawn    0.017242
# 46                     Errors    0.016887
# 1                     Assists    0.015969
# 8      Progressive Passes Rec    0.015662
# 11      Average Shot Distance    0.014061









# OUTPUT for new CSV
# Highly correlated features to drop (47):
# ['Goals + Assists', 'Non-Penalty Goals', 'Penalty Kicks Attempted', 'xG: Expected Goals', 'npxG: Non-Penalty xG', 'npxG + xAG', 'Shots Total', 'Shots on Target', 'Non-Penalty Goals - npxG', 'Passes Attempted', 'Total Passing Distance', 'Progressive Passing Distance', 'Passes Completed (Short)', 'Passes Attempted (Short)', 'Passes Completed (Medium)', 'Passes Attempted (Medium)', 'Passes Attempted (Long)', 'xA: Expected Assists', 'Key Passes', 'Passes into Final Third', 'Live-ball Passes', 'Throw-ins Taken', 'Inswinging Corner Kicks', 'Outswinging Corner Kicks', 'Shot-Creating Actions', 'SCA (Live-ball Pass)', 'SCA (Dead-ball Pass)', 'GCA (Live-ball Pass)', 'Tackles Won', 'Dribblers Tackled', 'Dribbles Challenged', 'Challenges Lost', 'Tkl+Int', 'Touches', 'Touches (Def Pen)', 'Touches (Def 3rd)', 'Touches (Mid 3rd)', 'Touches (Att 3rd)', 'Touches (Att Pen)', 'Touches (Live-Ball)', 'Successful Take-Ons', 'Times Tackled During Take-On', 'Carries', 'Total Carrying Distance', 'Progressive Carrying Distance', 'Carries into Final Third', 'Passes Received']
# Original features: 109
# Reduced features: 62

# Top 20 Important Features:
#                       Feature  Importance
# 16           Passes Completed    0.079249
# 33      Goal-Creating Actions    0.054730
# 1                       Goals    0.046236
# 0                         age    0.034595
# 43                     Blocks    0.031726
# 49  Carries into Penalty Area    0.031619
# 17    Passes Completed (Long)    0.026053
# 48         Take-Ons Attempted    0.025902
# 22              Through Balls    0.024188
# 12      Average Shot Distance    0.023953
# 20           Dead-ball Passes    0.023282
# 27             Passes Offside    0.020227
# 8          Progressive Passes    0.020016
# 9      Progressive Passes Rec    0.018297
# 54                Fouls Drawn    0.018212
# 7         Progressive Carries    0.017138
# 42          Tackles (Att 3rd)    0.016576
# 47                     Errors    0.016550
# 50                Miscontrols    0.016525
# 59            Ball Recoveries    0.016260