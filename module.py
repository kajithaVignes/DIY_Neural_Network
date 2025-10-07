import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


class Linear(Module):

    def __init__(self, input, output,biais=False):
        super(Linear, self).__init__()
        #matrice de poids pour la couche linéaire
        self._W = np.random.randn(input, output) * np.sqrt(2. / input)
        self._W_gradient = np.zeros_like(self._W)
        self.biais_exist=biais
        #si on ajoute un biais
        if self.biais_exist:
            self._bias = np.zeros((1, output))
            self._bias_gradient = np.zeros_like(self._bias)
        else:
            self._bias = None
            self._bias_gradient = None



    def zero_grad(self):
        #on remet le gradient à 0
       self._W_gradient=np.zeros_like(self._W)
       if self.biais_exist:
           self._bias_gradient=np.zeros_like(self._bias)

    def forward(self, X):
        self.input=X
        output= X@self._W
        if self.biais_exist:
            output+=self._bias
        return output


    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._W -= gradient_step*self._W_gradient
        if self.biais_exist:
            self._bias-=gradient_step*self._bias_gradient

    def backward_update_gradient(self, delta):
        self._W_gradient+= self.input.T@delta/ self.input.shape[0]
        if self.biais_exist:
            self._bias_gradient+=np.sum(delta,axis=0,keepdims=True) / self.input.shape[0]

    def backward_delta(self, delta):
        delta=np.clip(delta,-1,1)
        return delta@(self._W.T)



class Sequentiel(object):

    def __init__(self, loss, *modules):
        if len(modules) == 0:
            raise ValueError("No module")
        self.loss = loss
        self.modules = modules

    def forward(self, X):
        for module in self.modules:
            X = module.forward(X)
        return X

    def backward(self, y, yhat,gradient_step=1e-3):

        delta = self.loss.backward(y, yhat)
        #propage l'erreur en arrière(calcule les deltas)
        for module in reversed(self.modules):
            module.backward_update_gradient(delta)
            delta = module.backward_delta(delta)
            module.update_parameters(gradient_step)
            module.zero_grad()


    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def update_parameters(self, gradient_step=1e-3):
        for module in self.modules:
            module.update_parameters(gradient_step)

class Optim(object):

    def __init__(self,net,loss,eps):
        self.net=net
        self.loss=loss
        self.eps=eps

    def step(self,batch_x,batch_y):
      batch_yhat = self.net.forward(batch_x)
      self.net.zero_grad()
      loss_value = self.loss.forward(batch_y, batch_yhat)
      self.net.backward(batch_y, batch_yhat, self.eps)
      for module in self.net.modules:
        module.zero_grad()
        module.update_parameters(self.eps)
      return loss_value



def SGD(net, loss, X, Y, batch_size, epochs, eps):
    """
    Stochastic Gradient Descent avec split, losses, et métriques sklearn.
    """
    # 1. Split train/test
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[indices[:split_idx]], Y[indices[:split_idx]]
    X_test, y_test = X[indices[split_idx:]], Y[indices[split_idx:]]

    optim = Optim(net, loss, eps)
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        epoch_loss = []
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            loss_value = optim.step(batch_x, batch_y)
            epoch_loss.append(loss_value)

        # Compute losses
        train_losses.append(np.mean(epoch_loss))
        test_losses.append(loss.forward(y_test, net.forward(X_test)))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, train loss: {train_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses, X_test, y_test