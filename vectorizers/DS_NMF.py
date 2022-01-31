from os import scandir
import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import spdiags
from utils_Zhu_Ghodsi import ZG_number_of_topics

class DS_NMF(NMF):
    """Class to allow diagonal scaling of NMF
    Inherits from NMF (class): sklearn.decomposition.NMF
    and its docstring is given here for convenience.
    DS_NMF has one additional argument and 3 additional attributes, which are support 4 optional
    diagonal scalings.
    
    Diagaonl Scaling Non-Negative Matrix Factorization (NMF).
    
    As this class is a subclass of sklearn.decomposition.NMF we give it's document string here with the 
    extensions provided by DS_NMF, which include the diagonal scaling (scale_type) argument as well as 
    support for using Zhu-Ghodsi is used to estimate and automatically the number of components.
    
    
    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.
    The objective function is:
        .. math::
            0.5 * ||X - WH||_{loss}^2 + alpha * l1_{ratio} * ||vec(W)||_1
            + alpha * l1_{ratio} * ||vec(H)||_1
            + 0.5 * alpha * (1 - l1_{ratio}) * ||W||_{Fro}^2
            + 0.5 * alpha * (1 - l1_{ratio}) * ||H||_{Fro}^2
    Where:
    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)
    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)
    The generic norm :math:`||X - WH||_{loss}` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.
    The objective function is minimized with an alternating minimization of W
    and H.
    Read more in the :ref:`User Guide <NMF>`.
    Parameters
    ----------
    n_components : int, default=None
        Number of components, if n_components is not set all features
        are kept. if n_components<=0 then Zhu-Ghodsi is used to select the number of components.
        In this case the -n_components elbow index is used. Setting this to 0 gives the fewest number
        of components. For typical data 0, -1, or -2 are good choices.  
        
    scale_type: str, default=None,  
        Type of diagonal scaling to use. One of four options are supported.
        'RS': row scaling (rows of X are scaled to sum to 1.)
        'CS': column scaling (colums of X are scaled to sum to 1.)
        'NL': (Normalized Laplacian) rows are scaled by the square roots of the 
               row marginals and similarily the columns. This scaling is a generalization
               inspired by the normalized Laplacian and often improves NMF ability to find
               clusters.
        'RSCS': Employ both row and column scaling, this normalization is the non-logarithm 
                pointwise mutual information. (Note X is also scaled by the sum of counts to 
                complete this scaling.)


    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Default: None.
        Valid options:
        - `None`: 'nndsvd' if n_components <= min(n_samples, n_features),
          otherwise random.
        - `'random'`: non-negative random matrices, scaled with:
          sqrt(X.mean() / n_components)
        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)
        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)
        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)
        - `'custom'`: use custom matrices W and H
    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:
        'cd' is a Coordinate Descent solver.
        'mu' is a Multiplicative Update solver.
        .. versionadded:: 0.17
           Coordinate Descent solver.
        .. versionadded:: 0.19
           Multiplicative Update solver.
    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.
        .. versionadded:: 0.19
    tol : float, default=1e-4
        Tolerance of the stopping condition.
    max_iter : int, default=200
        Maximum number of iterations before timing out.
    random_state : int, RandomState instance or None, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.
    alpha : float, default=0.
        Constant that multiplies the regularization terms. Set it to zero to
        have no regularization.
        .. versionadded:: 0.17
           *alpha* used in the Coordinate Descent solver.
    l1_ratio : float, default=0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        .. versionadded:: 0.17
           Regularization parameter *l1_ratio* used in the Coordinate Descent
           solver.
    verbose : int, default=0
        Whether to be verbose.
    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.
        .. versionadded:: 0.17
           *shuffle* parameter used in the Coordinate Descent solver.
    regularization : {'both', 'components', 'transformation', None}, \
                     default='both'
        Select whether the regularization affects the components (H), the
        transformation (W), both or none of them.
        .. versionadded:: 0.24
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.
    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.
    scale_type: str
        Type of diagonal scaling to use. One of four options are supported.
    row_sums: ndarray of shape (n_samples,)
        giving the row marginals
    col_sums: ndarray of shape (,n_features)
        giving the column marginals 
    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.
    n_iter_ : int
        Actual number of iterations.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    Args:
    """    
    def __init__(self, n_components=None,  *, scale_type=None,init='warn', solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False):
        """initialization method of DS_NMF (diagonal scaling NMF)

        Args:
            n_components ([type], optional): [description]. Defaults to None.
            scale_type ([type], optional): [description]. Defaults to None.
            init (str, optional): [description]. Defaults to 'warn'.
            solver (str, optional): [description]. Defaults to 'cd'.
            beta_loss (str, optional): [description]. Defaults to 'frobenius'.
            tol ([type], optional): [description]. Defaults to 1e-4.
            max_iter (int, optional): [description]. Defaults to 200.
            random_state ([type], optional): [description]. Defaults to None.
            alpha ([type], optional): [description]. Defaults to 0..
            l1_ratio ([type], optional): [description]. Defaults to 0..
            verbose (int, optional): [description]. Defaults to 0.
            shuffle (bool, optional): [description]. Defaults to False.
        """        
        self.scale_type = scale_type

        super().__init__(n_components=n_components,  init=init, solver=solver,
                 beta_loss=beta_loss, tol=tol, max_iter=max_iter,
                 random_state=random_state, alpha=alpha, l1_ratio=l1_ratio, verbose=verbose,
                 shuffle=shuffle)
        
    def fit(self, X, y=None, **params):
        """Wrapper for a NMF model for the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def fit_transform(self, X, y=None, W=None, H=None):
        """wrapper for NMF.fit_transform to perform pre and post scaling, and whose docstring
        is given here.
        Learn a NMF model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.
        H : array-like of shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # preform diagonal scaling if requested
        if self.scale_type: 
            X = self.scale_counts(X)
        if self.n_components<=0:
            self.n_components=ZG_number_of_topics(X,elbow_index=-self.n_components,n_topics_upper_bound=max(X.shape))
        W = super().fit_transform(X,y,W,H)
        # unwind the diagonal scaling if done
        if self.scale_type:
            W, self.components_ = self.unscale_WH(W,self.components_)
        return W
    
    def transform(self, X, y=None, W=None, H=None):
        """wrapper for NMF.transform (docstring given below) to perform pre and post scaling
        Transform the data X according to the fitted NMF model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """   
        # preform diagonal scaling if requested
        if self.scale_type:
            X = self.scale_counts(X)
        W = super().transform(X,y,W,H)
        # unwind the diagonal scaling if done
        if self.scale_type:
            W, self.components_ = self.unscale_WH(W,self.components_)
        return W
    
    
    def unscale_WH(self,W,H):
        """unscale W and H to undo any scaling applied to the original matrix.

        Args:
            W : array-like of shape (n_samples, n_components)
            H : array-like of shape (n_components, n_samples)
        """  
        eps = np.finfo(float).eps
        row_sums = self.row_sums + eps
        col_sums = self.col_sums + eps
        if self.scale_type=='NL':
            row_sums = np.sqrt(row_sums) 
            col_sums = np.sqrt(col_sums)
        if self.scale_type != 'CS':
            dr=row_sums
            m = W.shape[0]
            Dr = spdiags(dr.transpose(),0,m,m)
            W = Dr*W
        if self.scale_type != 'RS':
            dc=col_sums
            n = H.shape[1]
            Dc = spdiags(dc,0,n,n)
            H = H*Dc
        if self.scale_type == 'RSCS':
            # Scale factor so matrix is entries converge as sample size increases
            scale = 1.0/np.sqrt(row_sums.sum())
            W *= scale
            H *= scale      
        return(W,H)
    
    def scale_counts(self,A,row_sums=None,col_sums=None):
        '''scale the matrixt of counts via NL, RS, CS,
        'NL' normalized Laplacian
        'RS' row stochastic
        'CS' column stochastic
        'RSCS' pointwise mutual information
        '''
        if self.scale_type is None:
            return A
        if self.scale_type.lower()=="none":
            return A
        eps = np.finfo(float).eps
        if row_sums is None:
            row_sums = A.sum(axis=1)
        if col_sums is None:
            col_sums = A.sum(axis=0)
        self.row_sums = row_sums
        self.col_sums = col_sums
        if self.scale_type=='NL':
           row_sums = np.sqrt(row_sums) 
           col_sums = np.sqrt(col_sums)
        if self.scale_type != 'CS':
            dr=1.0/(row_sums+eps)
            m=A.shape[0]
            Dr = spdiags(dr.transpose(),0,m,m)
            A = Dr*A
        if self.scale_type != 'RS':
            dc=1.0/(col_sums+eps)
            n = A.shape[1]
            Dc = spdiags(dc,0,n,n)
            A = A*Dc
        if self.scale_type == 'RSCS':
            # Scale factor so matrix is entries converge as sample size increases
            A=A*row_sums.sum()
        return A
  
# import numpy as np
# X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])  
# model_DS = DS_NMF(n_components=2, scale_type='NL', init='random', random_state=0)
# W_DS = model_DS.fit_transform(X)
# H_DS = model_DS.components_
# print(X-W_DS@H_DS)