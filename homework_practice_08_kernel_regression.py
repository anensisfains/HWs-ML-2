import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression class
    """

    def __init__(
        self,
        lr=0.01,
        regularization=1.0,
        tolerance=1e-2,
        max_iter=1000,
        batch_size=64,
        kernel_scale=1.0,
    ):
        """
        :param lr: learning rate
        :param regularization: regularization coefficient
        :param tolerance: stopping criterion for square of euclidean norm of weight difference
        :param max_iter: stopping criterion for iterations
        :param batch_size: size of the batches used in gradient descent steps
        :parame kernel_scale: length scale in RBF kernel formula
        """

        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.loss_history: list[float] = []
        self.kernel = RBF(kernel_scale)

    def calc_grad(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating gradient for x and y dataset
        :param x: features array
        :param y: targets array
        """
        batch_idx = np.random.choice(np.arange(x.shape[0]), self.batch_size, replace = False)
        x_st = x[batch_idx, :]
        y_st = y[batch_idx]
        
        grad1 = x_st.T@(x_st@self.w - y_st) 
        grad2 = self.regularization*(x@self.w) # считаем по всей выборке, иначе не получится по размерностям
        return grad1 + grad2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров с помощью градиентного спуска
        :param x: features array
        :param y: targets array
        :return: self
        """
        self.train = x
        K = self.kernel(x)
        self.w = np.random.rand(K.shape[0])
        
        for i in range(self.max_iter):
            grad = self.calc_grad(K, y)
            w_new = self.w - self.lr*grad
            if np.linalg.norm(w_new - self.w)**2 < self.tolerance:
                break
            self.w = w_new
            
        return self

    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров через аналитическое решение
        :param x: features array
        :param y: targets array
        :return: self
        """
        self.train = x
        K = self.kernel(x)
        self.w = np.linalg.inv(K + self.regularization * np.identity(K.shape[0])) @ y
        
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        x_kernel = self.kernel(x, self.train)
        return x_kernel @ self.w
