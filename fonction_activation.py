import numpy as np
from module import Module
class TanH(Module):

    def __init__(self):
        super(TanH,self).__init__()

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        self.input=X
        return np.tanh(X)

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, delta):
        ## Calcul la derivee de l'erreur
        return delta*(1-(np.tanh(self.input))**2)


class Sigmoide(Module):
    def __init__(self):
        super(Sigmoide,self).__init__()

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        self.input=X
        return 1 / (1 + np.exp(-X))

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, delta):
        ## Calcul la derivee de l'erreur
      sigmoid = 1 / (1 + np.exp(-self.input))
      return delta * sigmoid * (1 - sigmoid)