class hirem:
    """
    Initiates the construction of a hierarchical reserving model composed of a sequence of layers.

    Args:
        data: The input data for constructing, evaluating and testing the hierarchical reserving model.
    """

    def __init__(self, data):
        self.data = data
        self.data_observed = data
        self.data_training = data
        self.layers = []
        self.layers_dict = {}
        self.updaters = []
        self.update_after = []

    def __str__(self):
        output = "Hierarchical reserving model\n\n"
        output += "Data:\n----------\n"
        output += f"Observed: {len(self.data_observed)} records\n"
        if hasattr(self, 'data_validation'):
            output += f"- Training: {len(self.data_training)} records\n"
            output += f"- Validation: {len(self.data_validation)} records\n"
        if hasattr(self, 'data_test'):
            output += f"Test: {len(self.data_test)} records\n"
        output += f"\nLayers ({len(self.layers)}):\n----------\n"
        for i, layer in enumerate(self.layers):
            output += f"* {layer}\n"
        return output