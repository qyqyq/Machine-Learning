
import numpy as np
import scipy.stats as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor as GBR


def SquaredLoss_NegGradient(y, y_pred):
    return y - y_pred

def HuberLoss_NegGradient(y, y_pred, alpha):
    diff = y - y_pred
    delta = st.scoreatpercentile(np.abs(diff), alpha * 100)
    g = np.where(np.abs(diff) > delta, delta * np.sign(diff), diff)
    return g


class GradientBoosting(object):
    def __init__(self, M, base_learner, learning_rate=1.0, loss="square", alpha=0.9):
        self.M = M
        self.base_learner = base_learner
        self.learning_rate = learning_rate
        self.loss = loss
        self.alpha = alpha

    def fit(self, X, y):
        init_learner = self.base_learner
        y_pred = init_learner.fit(X, y).predict(X)
        self.base_learner_total = [init_learner]
        for m in range(self.M):
            if self.loss == "square":
                response = SquaredLoss_NegGradient(y, y_pred)
            elif self.loss == "huber":
                response = HuberLoss_NegGradient(y, y_pred, self.alpha)

            base_learner = self.base_learner
            y_pred += base_learner.fit(X, response).predict(X) * self.learning_rate
            self.base_learner_total.append(base_learner)

        return self


    def predict(self, X):
        pred = np.array([self.base_learner_total[m].predict(X) * self.learning_rate for m in range(1, self.M + 1)])
        pred = np.vstack((self.base_learner_total[0].predict(X), pred))
        pred_final = np.sum(pred, axis=0)
        return pred_final


if __name__ == "__main__":
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    
   
    model_s = GradientBoosting(M=1000, base_learner=DecisionTreeRegressor(), learning_rate=0.1, loss="square")
    model_s.fit(X_train, y_train)
    pred_s = model_s.predict(X_test)
    rmse_s = np.sqrt(mean_squared_error(y_test, pred_s))
    print('RMSE_SQUARELOSS: ', rmse_s)
   
    '''
    model_h = GradientBoosting(M=1000, base_learner=DecisionTreeRegressor(), learning_rate=0.1, loss="huber")
    model_h.fit(X_train, y_train)
    pred_h = model_h.predict(X_test)
    rmse_h = np.sqrt(mean_squared_error(y_test, pred_h))
    print('RMSE_HUBERLOSS: ', rmse_h)
     '''
    
    gbr = GBR()
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)
    gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))
    print('GBR_RMSE: ', gbr_rmse)
