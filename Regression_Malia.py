import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as ols
import statsmodels.api as sm
import sklearn.model_selection as skm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.ensemble import \
     (RandomForestRegressor as RF,
      GradientBoostingRegressor as GBR)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/maliakema/Desktop/BS data/Project/ObeseOrNot.csv')
print("Malia Kema")
X = df.drop(columns =['Gender', 'Height', 'Weight', 'Age', 'familyHistoryOverWeight','BMI','Classification'])
y=df['Classification']

df['NumberMainMeals'] = df['NumberMainMeals'].round(decimals=0)
df['FreqVegConsump'] = df['FreqVegConsump'].round(decimals=0)
df['WaterIntake'] = df['WaterIntake'].round(decimals=1)
df['PhysicalActivityFreq'] = df['PhysicalActivityFreq'].round(decimals=0)
df['TimeUsineTech'] = df['TimeUsingTech'].round(decimals=1)
df['BMI'] = df['BMI'].round(decimals=1)

(X_train,
 X_test,
 y_train,
 y_test) = skm.train_test_split(X,
                                y,
                                test_size=0.3,
                                random_state=0)
np.mean(y_train)

#Regression 
reg = DTR(max_depth=3)
reg.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(reg,
          feature_names=X.columns,
          ax=ax)
plt.show()
ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)
print(ccp_path)
#Prune the tree
kfold = skm.KFold(3,
                  shuffle=True,
                  random_state=10)
grid = skm.GridSearchCV(reg,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)

best_ = grid.best_estimator_
print("Regression Pruned")
np.mean((y_test - best_.predict(X_test))**2)
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(reg,
          feature_names=X.columns,
          ax=ax)
plt.show()

print("Bagging")

bag_Ob = RF(max_features=X_train.shape[1],
                n_estimators=500,
                random_state=0).fit(X_train, y_train)
y_hat_bag = bag_Ob.predict(X_test)
np.mean((y_test - y_hat_bag)**2)

RF_Ob = RF(max_features=3,
               random_state=0).fit(X_train, y_train)
y_hat_RF = RF_Ob.predict(X_test)
print("Random Forrest")
np.mean((y_test - y_hat_RF)**2)

#Boosting
boost_Ob = GBR(n_estimators=5000,
                   learning_rate=0.001,
                   max_depth=3,
                   random_state=0)
boost_Ob.fit(X_train, y_train)
y_hat_boost = boost_Ob.predict(X_test)
print("Boosting")
np.mean((y_test - y_hat_boost)**2)

print("RESULTS: Single(.205), Bagging (.093), RF (.088), Boosting(.141). Bagging and Random Forest outperformed a single decision tree.The single decision tree is often prone to overfitting which can result in a higher MSE.")
print("The Random Forest performs slightly better than bagging which is likely due to the additional randomness. Boosting seems to result in a higher MSE compared to Random Forest.I would pick either Bagging or Random Forest.")

#RFE Feature Selection process
selector = RFE(estimator = RF(), n_features_to_select =5)
selector.fit(X_train, y_train)
select_ft = X_train.columns[selector.support_]
print(select_ft)

#Anova
formula = 'Classification ~ C(NumberMainMeals) + C(FreqHighCalorie) + C(FreqVegConsump) + C(EatingBetweenMeals) + C(Smoker) + C(CountCalories) + C(PhysicalActivityFreq) + C(CommuteType) + C(AlcholIntake)'
model = ols(formula, data=df).fit()

anova_table = sm.stats.anova_lm(model)
print(anova_table)

#Correlations
correlation_table = df.corr(method ='pearson')
print(correlation_table)
correlation_with_y = correlation_table['Classification']
print('Correlation of Binary Classification')
print(correlation_with_y)
