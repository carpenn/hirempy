import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from .layer import hirem_layer

def fit(object, *args, **kwargs):
    """Dispatch function for fitting methods based on object class."""
    if isinstance(object, hirem_layer):
        return fit_layer_glm(object, *args, **kwargs)
    else:
        raise NotImplementedError(f"fit method not implemented for object of type {type(object)}")

def fit_layer_glm(layer, obj, formula, training=False, fold=None):
    """Fits a generalized linear model (GLM) layer in a hierarchical reserving model.

    Args:
        layer: The hirem_layer object to be fitted.
        obj: The hirem object containing the data and other layers.
        formula: The formula for the GLM model.
        training: Boolean indicating whether to use training data (default: False, uses observed data).
        fold: Fold number for cross-validation (if not None, excludes data from that fold).

    Returns:
        The fitted hirem_layer object.
    """

    print(f"Fitting layer_glm for {layer.name}...")

    layer.formula = formula  # Store the formula in the layer object

    data = obj.data_training.copy()  # Use a copy to avoid unintended side effects

    if not training:
        data = obj.data_observed.copy()

    if fold is not None:
        # Filter out data from the specified fold for cross-validation
        data = data.query("cv_fold != @fold")

    # Apply filter function to the data
    data_filtered = data[layer.filter_func(data.copy())]

    if layer.transformation is not None:
        # Apply transformation if defined
        label = formula.split('~')[0].strip()
        data_filtered[label] = layer.transformation.transform(data_filtered[label])

    # Handle weights
    weights_vec = None
    if hasattr(obj, 'weights'):
        weights_vec = obj.weights[data_filtered[obj.weight_var]]

    # Fit the GLM model using statsmodels
    if layer.method_options.family == 'gamma':
        family = sm.families.Gamma(link=sm.families.links.log)
    else:
        family = layer.method_options

    model = glm(formula=formula, data=data_filtered, family=family, weights=weights_vec)
    layer.fit = model.fit()  # Store the fitted model in the layer

    # Calculate balance correction (if applicable)
    if hasattr(obj, 'balance_var'):
        balance_correction = {}
        for group, group_data in data_filtered.groupby(obj.balance_var):
            balance_correction[group.name] = group_data[layer.name].sum() / model.predict(group_data).sum()

        # Handle potential missing levels in balance variable
        if len(balance_correction) != len(data_filtered[obj.balance_var].unique()):
            missing_levels = set(data_filtered[obj.balance_var].unique()) - set(balance_correction.keys())
            for level in missing_levels:
                balance_correction[level] = 1
            balance_correction = {k: v for k, v in sorted(balance_correction.items())}  # Sort for consistency

        layer.balance_correction = balance_correction

    # Calculate sigma or shape parameter (depending on family)
    if layer.method_options.family == 'gaussian':
        layer.sigma = np.std(layer.fit.resid)
    elif layer.method_options.family == 'gamma':
        from scipy.stats import gamma  # Assuming you have scipy installed
        shape_result = gamma.fit(layer.fit.model.endog, layer.fit.fittedvalues)
        layer.shape = shape_result[0]
        layer.shape_se = shape_result[1]

    return layer