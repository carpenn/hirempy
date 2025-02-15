from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.formula.api import glm
from scipy.stats import gamma
from .layer import hirem_layer
from .layer import hirem_get_layer_pos

import logging

logger = logging.getLogger(__name__)

def fit_hirem(obj: Any, 
           training: bool = False, 
           fold: Optional[int] = None, 
           weights: Optional[np.ndarray] = None, 
           weight_var: Optional[str] = None, 
           balance_var: Optional[str] = None, 
           **formulas: Dict[str, str]) -> Any:
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

def fit(object: Any, *args: Any, **kwargs: Any) -> Any:
    """Dispatch function for fitting methods based on object class.
    
    Args:
        object: The object to fit (typically a hirem_layer instance)
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        
    Returns:
        The fitted object
        
    Raises:
        NotImplementedError: If fitting method is not implemented for the object type
    """
    """Dispatch function for fitting methods based on object class."""
    if isinstance(object, hirem_layer):
        return fit_layer_glm(object, *args, **kwargs)
    else:
        raise NotImplementedError(f"fit method not implemented for object of type {type(object)}")

def _get_model_family(family_name: str) -> sm.families.Family:
    """Helper function to get the appropriate model family.
    
    Args:
        family_name: Name of the family ('gamma' or 'gaussian')
        
    Returns:
        statsmodels Family instance
        
    Raises:
        ValueError: If family_name is not supported
    """
    if family_name == 'gamma':
        return sm.families.Gamma(link=sm.families.links.log)
    elif family_name == 'gaussian':
        return sm.families.Gaussian()
    else:
        raise ValueError(f"Unsupported family: {family_name}")

def _calculate_balance_correction(data: pd.DataFrame, 
                               balance_var: str, 
                               layer_name: str, 
                               predictions: np.ndarray) -> Dict[Any, float]:
    """Calculate balance correction factors.
    
    Args:
        data: Input DataFrame
        balance_var: Name of the balance variable
        layer_name: Name of the layer
        predictions: Model predictions
        
    Returns:
        Dictionary of balance correction factors
    """
    balance_correction = {}
    for group, group_data in data.groupby(balance_var):
        balance_correction[group] = group_data[layer_name].sum() / predictions[group_data.index].sum()
    
    # Handle missing levels
    all_levels = set(data[balance_var].unique())
    missing_levels = all_levels - set(balance_correction.keys())
    for level in missing_levels:
        balance_correction[level] = 1.0
    
    return dict(sorted(balance_correction.items()))

def fit_layer_glm(layer: hirem_layer, 
                 obj: Any, 
                 formula: str, 
                 training: bool = False, 
                 fold: Optional[int] = None) -> hirem_layer:
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

    logger.info(f"Fitting layer_glm for {layer.name}...")
    
    # Validate inputs
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Formula must be a non-empty string")
        
    if not hasattr(layer, 'name') or not layer.name:
        raise ValueError("Layer must have a valid name")

    layer.formula = formula  # Store the formula in the layer object

    # Get the appropriate dataset
    data = obj.data_observed if not training else obj.data_training
    data = data.copy()  # Create a single copy for modifications

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
        try:
            weights_vec = obj.weights[data_filtered[obj.weight_var]-1]
        except Exception as e:
            logger.warning(f"Failed to apply weights: {str(e)}")
            weights_vec = None

    try:
        # Get the appropriate model family
        family = _get_model_family(layer.method_options.family) if hasattr(layer.method_options, 'family') else layer.method_options
        
        # Fit the GLM model
        model = glm(formula=formula, 
                   data=data_filtered, 
                   family=family, 
                   weights=weights_vec).fit()
    except Exception as e:
        raise RuntimeError(f"Failed to fit GLM model: {str(e)}")
    layer.fit = model

    # Calculate balance correction if applicable
    if hasattr(obj, 'balance_var') and obj.balance_var is not None:
        try:
            predictions = model.predict(data_filtered)
            layer.balance_correction = _calculate_balance_correction(
                data_filtered, 
                obj.balance_var, 
                layer.name, 
                predictions
            )
        except Exception as e:
            logger.warning(f"Failed to calculate balance correction: {str(e)}")

    # Calculate distribution parameters
    try:
        if hasattr(layer.method_options, 'family'):
            if layer.method_options.family == 'gaussian':
                layer.sigma = np.std(model.resid)
            elif layer.method_options.family == 'gamma':
                shape_result = gamma.fit(model.model.endog, model.fittedvalues)
                layer.shape = shape_result[0]
                layer.shape_se = shape_result[1]
    except Exception as e:
        logger.warning(f"Failed to calculate distribution parameters: {str(e)}")

    return layer