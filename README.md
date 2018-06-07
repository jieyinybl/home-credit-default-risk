# Home Credit Default Risk

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

[Home Credit](http://www.homecredit.net/) strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

Click [here](https://www.kaggle.com/c/home-credit-default-risk) for the original task in **Kaggle**.

# Data

The data can be downloaded from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data).

It includes:

- **application_{train|test}.csv**

    - This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
    - Static data for all applications. One row represents one loan in our data sample.

- **bureau.csv**

    - All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
    - For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

- **bureau_balance.csv**

    - Monthly balances of previous credits in Credit Bureau.
    - This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

- **POS_CASH_balance.csv**

    - Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
    - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

- **credit_card_balance.csv**

    - Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
    - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

- **previous_application.csv**

    - All previous applications for Home Credit loans of clients who have loans in our sample.
    - There is one row for each previous application related to loans in our data sample.

- **installments_payments.csv**

    - Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
    - There is a) one row for every payment that was made plus b) one row each for missed payment.
    - One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

- **HomeCredit_columns_description.csv**

    - This file contains descriptions for the columns in the various data files.
    
![Diagram of tables](./images/home_credit.png)

# Metric

Model is evaluated on [area under the ROC curve (ROC_AUC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

# Model

## Light GBM

I added up [LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html) as that I noticed on Kaggle this would achieve higher AUC_ROC score (about 0.78) compared to RandomForestClassifier (auc_roc at about 0.73):


> [1000]	train's auc: 0.872276	val's auc: 0.782073
>
> [2000]	train's auc: 0.925911	val's auc: 0.781464
>
> [3000]	train's auc: 0.957165	val's auc: 0.780306
>
> Fold  1 AUC: 0.780306
>
> [1000]	train's auc: 0.874167	val's auc: 0.783202
>
> [2000]	train's auc: 0.926376	val's auc: 0.782785
>
> [3000]	train's auc: 0.956695	val's auc: 0.780828
>
> Fold  2 AUC: 0.780828
>
> [1000]	train's auc: 0.872797	val's auc: 0.782933
>
> [2000]	train's auc: 0.924618	val's auc: 0.782177
>
> [3000]	train's auc: 0.955913	val's auc: 0.780925
>
> Fold  3 AUC: 0.780925
>
> [1000]	train's auc: 0.87251	val's auc: 0.780649
>
> [2000]	train's auc: 0.925375	val's auc: 0.78074
>
> [3000]	train's auc: 0.956294	val's auc: 0.779591
>
> Fold  4 AUC: 0.779591
>
> [1000]	train's auc: 0.873406	val's auc: 0.779406
>
> [2000]	train's auc: 0.925119	val's auc: 0.778319
>
> [3000]	train's auc: 0.956101	val's auc: 0.777131
>
> Fold  5 AUC: 0.777131 
>



Next, let's explain the steps of training with LightGBM.

**Load data**

```
train_data = gmb.Dataset(df)
# OR 
train_data = gmb.Dataset(data, label=label)
```

**Booster parameters**

```
param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}

```

**Metric parameters**

Here we can set it to `auc`.
```
param['metric'] = 'auc'
```

## RandomForestClassifier

Just to test some classic models on the data. The Notebook tries to train a `RandomForestClassifier`.

Without parameter tuning, the RandomForestClassifier reaches a ROC_AUC about 0.71.

|Fold | Training ROC_AUC | Validation ROC_AUC |
|:---:| :---------------:| :-----------------:|
|1 | 1.000000 | 0.711845 |
|2 | 1.000000 | 0.713323
|3 | 1.000000 | 0.713551
|4 | 1.000000 | 0.712408
|5 | 1.000000 | 0.715944

As we can see that the ROC_AUC is far better on the training set than the validation set. This shows that the model is over-fitting the training data. We will try some regulation parameters with GridSearchCV:

```
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

params = {'n_estimators': [100, 200],
          'min_samples_leaf': [5, 15]}

rf_clf2 = RandomForestClassifier(random_state=42)
clf2 = GridSearchCV(rf_clf2, params)
X_train, X_test, y_train, y_test = train_test_split(tr, y, test_size=0.3, random_state=42)
clf2.fit(X_train, y_train)
```
| Training ROC_AUC | Validation ROC_AUC |
| :---------------:| :-----------------:|
| 0.999785 | 0.733740

The model still over-fits the training data. However, the performance on the validation set shows that the regulation parameters helps.

And the best parameters of the above grid search is: 
```
{'min_samples_leaf': 5, 'n_estimators': 100}
```
