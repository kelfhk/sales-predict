import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score

'''
Preprocessing data
'''
print("Preprocessing data...\n")
# read data from csv
train = pd.read_csv('data/train.csv', parse_dates = ['date'], infer_datetime_format = True, dayfirst = True)

# aggregate day sales of shop-items in the same year-month
monthly_sales_df = train.groupby([train['date'].apply(lambda x: x.strftime('%Y-%m')), 'item_id','shop_id']).sum(numeric_only=True).reset_index()
monthly_sales_df = monthly_sales_df.rename(columns = {'item_cnt_day': 'item_cnt_month'})

# get the pivot tables with year-month as columns
monthly_sales_df = monthly_sales_df.pivot_table(index=['item_id','shop_id'], columns = 'date', values = 'item_cnt_month', fill_value = 0).reset_index()

# replace nan values with 0
monthly_sales_df.fillna(0, inplace = True)

'''
Preparing training set and testing set data
'''
print("Preparing training/testing set...\n")
    
year_month_list = monthly_sales_df.columns.values[2:]
num_year_month = len(year_month_list)

# shuffle rows and randomly select range of time of sales to be fitted into training
monthly_sales_df = monthly_sales_df.sample(frac = 1)

X = []
y = []

X.extend(monthly_sales_df.loc[:, year_month_list[:-5]].values)
X.extend(monthly_sales_df.loc[:, year_month_list[1:-4]].values)
X.extend(monthly_sales_df.loc[:, year_month_list[2:-3]].values)
X.extend(monthly_sales_df.loc[:, year_month_list[3:-2]].values)

y.extend(monthly_sales_df.loc[:, year_month_list[-5]].values)
y.extend(monthly_sales_df.loc[:, year_month_list[-4]].values)
y.extend(monthly_sales_df.loc[:, year_month_list[-3]].values)
y.extend(monthly_sales_df.loc[:, year_month_list[-2]].values)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# checking the shapes
print("Shape of x_train :", X_train.shape)
print("Shape of x_valid :", X_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_valid :", y_test.shape)

'''
Building and training model
'''
print("Building model...\n")

# use the best parameters (found by experiments) to fit training sets into light gradient boost machine model
best_params = {'boosting_type': 'dart', 'learning_rate': 0.05, 'max_depth': 2, 'min_data_in_leaf': 2, 'n_estimators': 500, 'num_leaves': 4, 'reg_alpha': 0.05, 'reg_lambda': 0.01}

lgb = LGBMRegressor(**best_params)
lgb.fit(X_train, y_train, eval_metric="l1")

score = np.mean(cross_val_score(lgb, X_test, y_test, scoring='neg_mean_absolute_error'))
print(f"Cross validated mean absolute error (-ve): {score}")


'''
Saving model
'''
model_save_path = 'model/model.txt'

    
# save new model if the new model has higher cross validated score than previous model
try:
    with open('model/model_score.txt') as f:
        prev_score = float(next(f))
except:
    prev_score = -np.inf
    
if score > prev_score:
    # save the model
    lgb.booster_.save_model(model_save_path)
    
    # update the new cross validated score
    with open('model/model_score.txt', 'w') as f:
        f.write(str(score))

    print(f"Model saved in {model_save_path}")
else:
    print("Model is not saved")
