import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Tassandro\Downloads\Cadastro_clientes.csv").set_index('ID')
plt.figure(figsize=(15,5))
df['Income'].hist(bins = 10, rwidth = 0.90)

from sklearn.model_selection import train_test_split

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.20, random_state = 61658)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'max_leaf_nodes':[3, 4, 5, 6, 7, 8],
    'min_samples_leaf':[5, 10, 20], 
}

model = GridSearchCV(
    DecisionTreeRegressor(random_state=61658),
    params,
    cv = 10,
    scoring='neg_mean_squared_error',
    verbose = 10,
    n_jobs = 1,
     
)

model.fit(X_tr, y_tr)

model.best_params_

(model.best_score_*(-1))**0.50

from sklearn.metrics import mean_squared_error
mean_squared_error(y_ts, model.predict(X_ts))**0.50

##**Avaliando os resultados**

plt.figure(figsize=(15,5))
plt.hist(y_ts - model.predict(X_ts), bins=10, rwidht=0.90)

from sklearn.tree import plot_tree

plt.figure(figsize=(15,7))
plot_tree(
    model.best_estimator_,
    feature_names = X_tr.columns,
    filled = True,
    node_ids = True,
    rounded = True,
    impurity = False,
    precision = 2
);
