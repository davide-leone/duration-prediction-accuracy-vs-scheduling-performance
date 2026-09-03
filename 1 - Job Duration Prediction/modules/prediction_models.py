import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import deque, defaultdict

### NORMALIZED POLYNOMIAL

class NAGPolynomialRegressor:
    '''
    
    This is the actual on-line implementation from the paper: 
    “Improving Backfilling by using Machine Learning to Predict Running Times”
    
    '''
    def __init__(self, degree=2, alpha=0.01, eta=1e-8, gamma=0.9):
        self.degree = degree
        self.alpha = alpha  # L2 regularization strength
        self.eta = eta  # Learning rate
        self.gamma = gamma  # NAG momentum factor
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.w = None  # Weight vector
        self.v = None  # Velocity for NAG updates

    def transform_features(self, X):
        X_poly = self.poly.fit_transform(X)
        return self.scaler.fit_transform(X_poly)  # Standardize features

    def fit(self, X, y):
        X_poly = self.transform_features(X)
        n_samples, n_features = X_poly.shape

        if self.w is None:
            self.w = np.zeros(n_features)
            self.v = np.zeros(n_features)

        for i in range(n_samples):
            xi, yi = X_poly[i], y[i]
            
            # Compute Lookahead Gradient (NAG step)
            w_ahead = self.w - self.gamma * self.v  # Lookahead step
            gradient = -2 * (yi - np.dot(xi, w_ahead)) * xi + 2 * self.alpha * w_ahead
            
            # Clip extreme values to prevent instability
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Update velocity and weights
            self.v = self.gamma * self.v + self.eta * gradient
            self.w -= self.v

    def predict(self, X):
        X_poly = self.scaler.transform(self.poly.transform(X))
        return np.maximum(np.dot(X_poly, self.w), 1).astype(int)  # Ensure non-negative predictions


class RidgePolynomialRegressor:
    '''

    Since we have all the data available we can use an off-line implementation which uses Ridge regression
    
    '''
    def __init__(self, degree=2, alpha=0.01, max_time_limit=86400):
        self.degree = degree
        self.alpha = alpha  # L2 regularization strength
        self.max_time_limit = max_time_limit
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)  # Standardize features
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)  # Standardize using the same scaler
        partial = self.model.predict(X_scaled)

        # Ensure predictions stay in valid range (0 to max_time_limit seconds)
        result = np.where(partial < 1, 1, np.where(partial > self.max_time_limit, self.max_time_limit, partial)).astype(int)
        
        return result  # Returns an array of integers


class OnlineRidgePolynomialRegressor:
    def __init__(self, degree=2, alpha=0.01, max_time_limit=86400,
                 batch_size=100, max_history=None):
        """
        degree: polynomial degree
        alpha: ridge L2 regularization
        batch_size: retrain after this many new samples
        max_history: keep at most this many past samples (None = keep all)
        """
        self.degree = degree
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_history = max_history
        self.max_time_limit = max_time_limit
        
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

        # Buffers for incremental data
        self.X_buffer = []
        self.y_buffer = []
        self.total_seen = 0
        self._is_fit = False

    def _clip_result(self, preds):
        preds = np.where(preds < 1, 1, preds)
        preds = np.where(preds > self.max_time_limit, self.max_time_limit, preds)
        return preds.astype(int)

    def fit(self, X, y):
        """Initial full training."""
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)
        self._is_fit = True

    def partial_fit(self, x_new, y_new):
        """
        Add one (or more) new samples.
        Retrains automatically when batch_size is reached.
        """
        x_new = np.atleast_2d(x_new)
        y_new = np.array(y_new, ndmin=1)

        self.X_buffer.append(x_new)
        self.y_buffer.append(y_new)
        self.total_seen += len(y_new)

        # Automatic retraining
        if len(self.X_buffer) >= self.batch_size:
            self._retrain()

    def _retrain(self):
        """Retrains full model using all historical data or window."""
        X_batch = np.vstack(self.X_buffer)
        y_batch = np.concatenate(self.y_buffer)

        # Optionally limit stored history
        if self.max_history is not None and len(y_batch) > self.max_history:
            X_batch = X_batch[-self.max_history:]
            y_batch = y_batch[-self.max_history:]

        # Full refit
        self.fit(X_batch, y_batch)

        # Clear buffer
        self.X_buffer = []
        self.y_buffer = []

    def predict(self, X):
        if not self._is_fit:
            raise RuntimeError("Model has not been fit yet.")
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self._clip_result(self.model.predict(X_scaled))


### DECISION TREE
        
class OnlineDecisionTreeRegressor:
    def __init__(self, batch_size=100, max_history=None, max_time_limit=86400, **tree_kwargs):
        """
        batch_size: retrain after this many new samples
        max_history: keep at most this many past samples
        tree_kwargs: passed to DecisionTreeRegressor (e.g., max_depth)
        """
        self.batch_size = batch_size
        self.max_history = max_history
        self.max_time_limit = max_time_limit

        self.tree_kwargs = tree_kwargs
        self.model = DecisionTreeRegressor(**tree_kwargs)

        self.X_buffer = []
        self.y_buffer = []
        self._is_fit = False

    def _clip(self, y):
        y = np.where(y < 1, 1, y)
        y = np.where(y > self.max_time_limit, self.max_time_limit, y)
        return y.astype(int)

    def fit(self, X, y):
        self.model = DecisionTreeRegressor(**self.tree_kwargs)
        self.model.fit(X, y)
        self._is_fit = True

    def partial_fit(self, x_new, y_new):
        x_new = np.atleast_2d(x_new)
        y_new = np.array(y_new, ndmin=1)

        self.X_buffer.append(x_new)
        self.y_buffer.append(y_new)

        if len(self.X_buffer) >= self.batch_size:
            self._retrain()

    def _retrain(self):
        X_batch = np.vstack(self.X_buffer)
        y_batch = np.concatenate(self.y_buffer)

        if self.max_history is not None and len(y_batch) > self.max_history:
            X_batch = X_batch[-self.max_history:]
            y_batch = y_batch[-self.max_history:]

        self.fit(X_batch, y_batch)
        self.X_buffer = []
        self.y_buffer = []

    def predict(self, X):
        if not self._is_fit:
            raise RuntimeError("Model has not been fit yet.")
        return self._clip(self.model.predict(X))


### KNN

class OnlineKNNRegressor:
    def __init__(self, k=5, max_buffer=200000):
        self.k = k
        self.max_buffer = max_buffer
        self.buffer_X = deque(maxlen=max_buffer)
        self.buffer_y = deque(maxlen=max_buffer)

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=self.k, weights='distance'))
        ])

        self.is_fitted = False

    def partial_fit(self, X, y):
        for xi, yi in zip(X, y):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)

        X_arr = np.array(self.buffer_X)
        y_arr = np.array(self.buffer_y)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)


class OnlineKNNRegressor_UserLocal:
    def __init__(self, k=5, max_buffer=200000, user_index=0):
        """
        Parameters
        ----------
        k : int
            Number of neighbors
        max_buffer : int
            Maximum number of samples kept in memory
        user_index : int
            Column index of the user ID in X
        """
        self.k = k
        self.max_buffer = max_buffer
        self.user_index = user_index

        self.buffer_X = deque(maxlen=max_buffer)
        self.buffer_y = deque(maxlen=max_buffer)

        self.scaler = StandardScaler()
        self.is_fitted = False

    def partial_fit(self, X, y):
        # Append new samples
        for xi, yi in zip(X, y):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)

        # Convert buffer to arrays
        X_arr = np.asarray(self.buffer_X)
        y_arr = np.asarray(self.buffer_y)

        # Scale features
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        # Extract user IDs
        user_arr = X_arr[:, self.user_index].astype(int)

        # Group indices by user (O(N), done once)
        user_indices = defaultdict(list)
        for i, u in enumerate(user_arr):
            user_indices[u].append(i)

        # Global fallback model
        self.global_model = KNeighborsRegressor(
            n_neighbors=self.k,
            weights="distance"
        )
        self.global_model.fit(X_scaled, y_arr)

        # Per-user KNN models
        self.user_models = {}

        for u, idx in user_indices.items():
            k_eff = min(self.k, len(idx))
            knn = KNeighborsRegressor(
                n_neighbors=k_eff,
                weights="distance"
            )
            knn.fit(X_scaled[idx], y_arr[idx])
            self.user_models[u] = knn

        # Store for prediction
        self.X_scaled = X_scaled
        self.y_arr = y_arr
        self.user_arr = user_arr

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)

        preds = np.empty(len(X), dtype=float)

        for i, (xi, xi_s) in enumerate(zip(X, X_scaled)):
            user_id = int(xi[self.user_index])
            model = self.user_models.get(user_id, self.global_model)
            preds[i] = model.predict([xi_s])[0]

        return preds


### KNN CLASSIFIER

from sklearn.neighbors import KNeighborsClassifier

class OnlineKNNClassifier:
    def __init__(self, k=5, max_buffer=200000):
        self.k = k
        self.max_buffer = max_buffer

        self.buffer_X = deque(maxlen=max_buffer)
        self.buffer_y = deque(maxlen=max_buffer)

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=k,
                weights="distance",
                metric="euclidean"
            ))
        ])

        self.is_fitted = False

    def partial_fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        for xi, yi in zip(X, y):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)

        X_arr = np.asarray(self.buffer_X)
        y_arr = np.asarray(self.buffer_y)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict_proba(X)
