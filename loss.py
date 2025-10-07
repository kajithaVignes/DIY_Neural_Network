import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):
    def forward(self, y, yhat):
        yhat = yhat.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return np.mean((y - yhat)**2)

    def backward(self,y,yhat):
        yhat = yhat.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return 2 * (yhat - y) / y.shape[0]

class CELoss(Loss):
    def forward(self, y, yhat):
        yhat = yhat - np.max(yhat, axis=1, keepdims=True)  # stabilité numérique
        log_softmax = yhat - np.log(np.sum(np.exp(yhat), axis=1, keepdims=True))
        return -np.sum(y * log_softmax, axis=1).mean()

    def backward(self, y, yhat):
        # Softmax
        yhat = yhat - np.max(yhat, axis=1, keepdims=True)
        softmax = np.exp(yhat) / np.sum(np.exp(yhat), axis=1, keepdims=True)
        return softmax - y


class BCE(Loss):
    def forward(self, y, yhat):
        eps = 1e-12
        yhat = np.clip(yhat, eps, 1 - eps)
        return np.mean(-(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))

    def backward(self, y, yhat):
        eps = 1e-12
        yhat = np.clip(yhat, eps, 1 - eps)
        return -((y / yhat - (1 - y) / (1 - yhat)) / y.shape[0])