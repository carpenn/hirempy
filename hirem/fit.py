import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from .layer import hirem_layer
from .layer import hirem_get_layer_pos

import patsy

def fit_hirem(obj, training=False, fold=None, weights=None, weight_var=None, balance_var=None, **formulas):
    """Fits one or multiple layers of the hierarchical reserving model.

    Args:
        obj: The hirem object.
        training: Whether to fit on training data (True) or observed data (False).
        fold: Index of the fold for cross-validation (None for all records).
        weights: Optional. Vector of weights.
        weight_var: Optional. Name of the weight variable in the data.
        balance_var: Optional. Name of the variable for balance correction.
        **formulas: Keyword arguments where keys are layer names and values are formulas.
    Returns:
        The fitted hirem object.
    """

    obj.balance_var = balance_var
    obj.weight_var = weight_var
    obj.weights = weights

    for layer_name, formula in formulas.items():
        try:
            index = hirem_get_layer_pos(obj, layer_name)
        except ValueError as e:
            print(f"Warning: Skipping layer '{layer_name}': {e}")
            continue

        obj.layers[index] = fit(obj.layers[index], obj, formula, training, fold)

    return obj

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
        design_info = patsy.ModelDesc.from_formula(formula)
        terms = design_info.lhs + design_info.rhs
        label = terms[0].name()
        data_filtered[label] = layer.transformation.transform(data_filtered[label])

    # Handle weights
    weights_vec = None
    if hasattr(obj, 'weights') and obj.weights is not None:
        weights_vec = obj.weights[data_filtered[obj.weight_var]]

    # Fit the GLM model using statsmodels
    if layer.method_options.family == 'gamma':
        family = sm.families.Gamma(link=sm.families.links.log)
    else:
        family = layer.method_options

    model = glm(formula=formula, data=data_filtered, family=family, weights=weights_vec).fit()
    layer.fit = model

    # Calculate balance correction (if applicable)
    if hasattr(obj, 'balance_var') and obj.balance_var is not None:
        balance_correction = {}
        for group, group_data in data_filtered.groupby(obj.balance_var):
            balance_correction[group] = group_data[layer.name].sum() / model.predict(group_data).sum()

        # Handle potential missing levels in balance variable
        if len(balance_correction) != len(data_filtered[obj.balance_var].unique()):
            missing_levels = set(data_filtered[obj.balance_var].unique()) - set(balance_correction.keys())
            for level in missing_levels:
                balance_correction[level] = 1
            balance_correction = dict(sorted(balance_correction.items()))  # Sort for consistency

        layer.balance_correction = balance_correction

    # Calculate sigma or shape parameter (depending on family)
    if layer.method_options.family == 'gaussian':
        layer.sigma = np.std(model.resid)
    elif layer.method_options.family == 'gamma':
        from scipy.stats import gamma  # Assuming you have scipy installed
        shape_result = gamma.fit(model.model.endog, model.fittedvalues)
        layer.shape = shape_result[0]
        layer.shape_se = shape_result[1]

    return layer