class HiremTransformation:
    """
    Represents a transformation for use in a hierarchical reserving model.

    Args:
        name: The name of the transformation.
        formula: A function that returns a string representation of the transformation.
        transform: A function that applies the transformation to data.
        inverse_transform: A function that applies the inverse transformation to data.
        dtransform: A function that calculates the derivative of the transformation.
    """

    def __init__(self, name, formula, transform, inverse_transform, dtransform):
        self.name = name
        self.formula = formula
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.dtransform = dtransform

    def __str__(self):
        return f"HiremTransformation: {self.name}"

# Define specific transformation objects
hirem_transformation_identity = HiremTransformation(
    'identity',
    lambda x: x,  # Formula for identity is simply the input itself
    lambda x: x,
    lambda x: x,
    lambda x: 1  # Derivative of identity is 1
)

hirem_transformation_log = HiremTransformation(
    'log',
    lambda x: f"log({x})",
    lambda x: np.log(x),
    lambda x: np.exp(x),
    lambda x: 1 / x
)

hirem_transformation_log_neg = HiremTransformation(
    'log',
    lambda x: f"log({x})",
    lambda x: -np.log(x),
    lambda x: np.exp(-x),
    lambda x: 1 / x
)

hirem_transformation_logit = HiremTransformation(
    'logit',
    lambda x: f"log({x} / (1 - {x}))",
    lambda x: np.log(x / (1 - x)),
    lambda x: np.exp(x) / (1 + np.exp(x)),
    lambda x: 1 / (x * (1 - x))
)