import numpy as np

from typing import Callable

from itertools import permutations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import cauchy


class RandomFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func
        
    def fit(self, X, y=None):
        X_pca = X   # сначала тут было преобразование X в X_pca, но потом убрал его в пайплайн
        
        # нужно для синт. данных, т.к. там поменьше наблюдений
        if (X_pca.shape[0] * (X_pca.shape[0] - 1)) / 2 > 10**6: 
            idx = np.random.choice(X_pca.shape[0], (10**6, 2), replace=True)
        else:
            idx = np.array(list(permutations(np.arange(X_pca.shape[0]), 2)))
            
        resids = (X_pca[idx[:, 0]] - X_pca[idx[:, 1]]) ** 2
        sigma = np.median(resids.sum(axis=1))

        self.w = np.random.normal(loc=0, scale=1/np.sqrt(sigma), size=(self.n_features, self.new_dim))
        self.b = np.random.uniform(low=-np.pi, high=np.pi, size=self.n_features)

        return self

    def transform(self, X, y=None):
        X_pca = X                   # тут тоже было преобразование, но потом убрал его в пайплайн
        return self.func(X_pca @ self.w.T + self.b)

class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        X_pca = X 
        idx = np.random.choice(X_pca.shape[0], (10**6, 2), replace=True)
        resids = (X_pca[idx[:, 0]] - X_pca[idx[:, 1]]) ** 2
        sigma = np.median(resids.sum(axis=1))
        
        if self.n_features == self.new_dim:
            G = np.random.normal(0, 1, size=(self.n_features, self.new_dim))
            Q, R = np.linalg.qr(G)
            S = np.diag(np.sqrt(np.random.chisquare(df=self.new_dim, size=self.new_dim)))
            self.w = (1 / np.sqrt(sigma)) * (S @ Q)
            self.b = np.random.uniform(low=-np.pi, high=np.pi, size=self.n_features)
            
        elif self.n_features < self.new_dim:
            G = np.random.normal(0, 1, size=(self.new_dim, self.new_dim))
            Q, R = np.linalg.qr(G)
            S = np.diag(np.sqrt(np.random.chisquare(df=self.new_dim, size=self.new_dim)))
            self.w = ((1 / np.sqrt(sigma)) * (S @ Q))[:self.n_features]
            self.b = np.random.uniform(low=-np.pi, high=np.pi, size=self.n_features)
            
        else:
            final = []
            num_iterations = self.n_features // self.new_dim
            for el in range(num_iterations):
                G = np.random.normal(0, 1, size=(self.new_dim, self.new_dim))
                S = np.diag(np.sqrt(np.random.chisquare(df=self.new_dim, size=self.new_dim)))
                Q, R = np.linalg.qr(G)
                final.append((1 / np.sqrt(sigma)) * (S @ Q))
            if self.n_features % self.new_dim != 0:
                G = np.random.normal(0, 1, size=(self.new_dim, self.new_dim))
                S = np.diag(np.sqrt(np.random.chisquare(df=self.new_dim, size=self.new_dim)))
                Q, R = np.linalg.qr(G)
                final.append(((1 / np.sqrt(sigma)) * (S @ Q))[:self.n_features % self.new_dim])
            self.w = np.vstack(final)
            self.b = np.random.uniform(low=-np.pi, high=np.pi, size=self.n_features)
        return self

class SignRFF(RandomFeatureCreator):
    
    def transform(self, X):
        X_pca = X
        product = -X_pca @ self.w.T + self.b
        return np.sign(product)
        
class RandomLaplaceFeatureCreator(RandomFeatureCreator):
    
    def fit(self, X, y=None):
        X_pca = X 
        idx = np.random.choice(X_pca.shape[0], (10**6, 2), replace=True)
        resids = (X_pca[idx[:, 0]] - X_pca[idx[:, 1]]) ** 2
        sigma = np.median(resids.sum(axis=1))
        self.w = np.random.laplace(loc=0, scale=1/np.sqrt(sigma), size=(self.n_features, self.new_dim))
        self.b = np.random.uniform(low=-np.pi, high=np.pi, size=self.n_features)
        return self
        

class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=RandomFeatureCreator,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        if classifier_params is None:
            classifier_params = {}
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=n_features, new_dim=new_dim, func=func
        )
        self.pipeline = None
        

    def fit(self, X, y):
        if not self.use_PCA:
            pipeline_steps: list[tuple] = [  
            ('feature_creator', self.feature_creator),
            ('classifier', self.classifier)
            ]
        else:
            pipeline_steps: list[tuple] = [
            ('pca', PCA(n_components = self.new_dim)),
            ('feature_creator', self.feature_creator),
            ('classifier', self.classifier)
            ]
            
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
