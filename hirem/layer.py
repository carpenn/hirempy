from collections import namedtuple

# Define a namedtuple for family options (mimicking R's family object)
Family = namedtuple("Family", ["family", "link"])

class hirem_layer:
    def __init__(self, name, method, method_class, method_options, filter_func=None, transformation=None):
        self.name = name
        self.method = method
        self.method_options = method_options
        self.formula = None # Add formula attribute

        if filter_func is None:
            self.filter_func = lambda x: [True] * len(x) if hasattr(x, '__len__') else True # Handle cases where x is not iterable
        else:
            self.filter_func = filter_func

        self.transformation = transformation

    def __str__(self):
        if self.transformation is not None:
            output = f"{self.transformation.name}({self.name})"
        else:
            output = self.name

        output += f": {self.method} "

        if self.__class__.__name__ == 'layer_glm' and isinstance(self.method_options, Family):
            output += f"{self.method_options.family} ({self.method_options.link}) "

        output += "\n"
        if self.formula is not None:
            output += str(self.formula) # Assuming formula has a __str__ method
        return output

def hirem_layer_func(obj, name, method, method_class, method_options, filter_func=None, transformation=None):
    layer = hirem_layer(name, method, method_class, method_options, filter_func, transformation)
    obj.layers.append(layer)
    obj.layers_dict[name] = layer #Store layers in a dictionary for faster access
    return obj

def layer_glm(obj, name, family, filter_func=None, transformation=None):
    return hirem_layer_func(obj, name, 'glm', 'layer_glm', family, filter_func, transformation)

def layer_gbm(obj, name, distribution, n_trees=500, interaction_depth=2, n_minobsinnode=10,
              shrinkage=0.1, bag_fraction=0.5, cv_folds=0, select_trees='perf', filter_func=None, transformation=None):

    options = {
        'shrinkage': shrinkage,
        'n_minobsinnode': n_minobsinnode,
        'interaction_depth': interaction_depth,
        'n_trees': n_trees,
        'select_trees': select_trees,
        'bag_fraction': bag_fraction,
        'distribution': distribution,
        'cv': cv_folds
    }

    return hirem_layer_func(obj, name, 'gbm', 'layer_gbm', options, filter_func, transformation)

def hirem_get_layer_pos(obj, layer_name):
    if not obj.layers:
        raise ValueError('There are no layers defined for the supplied hierarchical reserving model.')

    if layer_name not in obj.layers_dict:
        raise ValueError(f'Layer {layer_name} not found.')

    #Find the index of the layer
    for i, layer in enumerate(obj.layers):
        if layer.name == layer_name:
            return i
    return None #Should never get here as the layer is in the dictionary