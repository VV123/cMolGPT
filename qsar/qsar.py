import numpy as np
from sklearn.model_selection import RepeatedKFold, train_test_split
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

name = 'EGFR'
# name = 'HTR1A'
# name = 'S1PR1'
X = np.load('npy/{}_X.npy'.format(name))
y = np.load('npy/{}_y.npy'.format(name))
print(X.shape)
print(y.shape)
Ntree = 500
nBatches = 16
nCpU = 16
pH = 7.4
rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=128)  # 20 cross validation


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)
print(y_train.shape)
print(y_test.shape)


model = lgb.LGBMRegressor(n_estimators=Ntree, n_jobs=nCpU, subsample=0.8, colsample_bytree=0.8,
                                  subsample_freq=3)
model.fit(X_train, y_train)

Y_pred_test = model.predict(X_test)
R_test = pearsonr(y_test, Y_pred_test)[0]
mse_test = mean_squared_error(y_test,Y_pred_test)
rmse_test = np.sqrt(mse_test)
print(mse_test)
print(R_test)


X_rnn = np.load('./npy/{}_RNN_X.npy'.format(name))
Y_rnn_pred = model.predict(X_rnn)
np.save('./npy/y/{}_RNN_Y.npy'.format(name),Y_rnn_pred)

X_trans = np.load('./npy/{}_Transformer_X.npy'.format(name))
Y_trans_pred = model.predict(X_trans)
np.save('./npy/y/{}_Transformers_Y.npy'.format(name),Y_trans_pred)
