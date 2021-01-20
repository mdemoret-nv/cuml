"""
This file gathers Scikit-Learn code that would otherwise
require a version-dependent import from the sklearn library
"""

# Original authors from Sckit-Learn:
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.


from functools import wraps
from ....common.base import Base
from ..utils.validation import check_X_y
from ....thirdparty_adapters import check_array
from cuml.internals.api_decorators import mirror_args

class BaseEstimator(Base):

    def __init_subclass__(cls):
        orig_init = cls.__init__

        import inspect
        import numpydoc.docscrape
        from cuml.common.doc_utils import CumlDocString

        orig_sig = inspect.signature(orig_init)

        @wraps(orig_init)
        def init(self, /, *args, handle=None, verbose=False, output_type=None, **kwargs):
            handle = kwargs['handle'] if 'handle' in kwargs else None
            verbose = kwargs['verbose'] if 'verbose' in kwargs else False
            output_type = kwargs['output_type'] if 'output_type' in kwargs \
                else None
            Base.__init__(self, handle=handle, verbose=verbose,
                          output_type=output_type)
            for param in ['handle', 'verbose', 'output_type']:
                if param in kwargs:
                    del kwargs[param]
            orig_init(self, *args, **kwargs)

        base_sig = inspect.signature(Base.__init__)
        sig = inspect.signature(init)
        unwrap_sig = inspect.signature(init, follow_wrapped=False)

        base_doc = CumlDocString(inspect.getdoc(Base))

        new_params = list(inspect.signature(init).parameters.values())
        new_doc = CumlDocString(inspect.getdoc(cls))
        new_doc_params = new_doc["Parameters"]

        insert_idx = 0

        # First, find the place to insert keyword args. Will be last place before VAR_KEYWORD
        for insert_idx, param in enumerate(new_params):
            if (param.kind == param.VAR_KEYWORD):
                insert_idx -= 1
                break

        insert_idx += 1

        for param in unwrap_sig.parameters.values():            
            if (param.kind != param.KEYWORD_ONLY):
                continue

            if (param.name not in sig.parameters.keys()):
                # Insert into new_params
                new_params.insert(insert_idx, param.replace())
                insert_idx += 1

            new_doc.add_parameter(base_doc.get_parameter(param.name), update=True)
        
        init.__signature__ = sig.replace(parameters=new_params)

        cls.__init__ = init
        cls.__doc__ = str(new_doc)

    @classmethod
    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            Else, the attribute must already exist and the function checks
            that it is equal to `X.shape[1]`.
        """
        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
        else:
            if not hasattr(self, 'n_features_in_'):
                raise RuntimeError(
                    "The reset parameter is False but there is no "
                    "n_features_in_ attribute. Is this estimator fitted?"
                )
            if n_features != self.n_features_in_:
                raise ValueError(
                    'X has {} features, but this {} is expecting {} features '
                    'as input.'.format(n_features, self.__class__.__name__,
                                       self.n_features_in_)
                )

    def _validate_data(self, X, y=None, reset=True,
                       validate_separately=False, **check_params):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """

        if y is None:
            if self._get_tags()['requires_y']:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
            X = check_array(X, **check_params)
            out = X
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = check_array(X, **check_X_params)
                y = check_array(y, **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get('ensure_2d', True):
            self._check_n_features(X, reset=reset)

        return out


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)

        y : ndarray of shape (n_samples,), default=None
            Target values.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)
