import pandas as pd

def split_data(obj, observed=None, validation=None, cv_fold=None):
    if observed is not None:
        if callable(observed):
            obj.data_observed = observed(obj.data)
        else:
            obj.data_observed = observed.copy()  # Important to copy to avoid unintended side effects

    if cv_fold is not None:
        # Create a new column for fold assignment using stratified sampling
        obj.data_observed["cv_fold"] = pd.qcut(obj.data_observed.sample(frac=1), cv_fold, labels=False) + 1
        obj.cv_fold = cv_fold

    if validation is not None:
        if isinstance(validation, (float, int)):
            # Random sampling for validation split
            validation_prop = min(1, max(0, validation))  # Ensure validation proportion is between 0 and 1
            subset = obj.data_observed.sample(frac=validation_prop, random_state=0)  # Set random state for reproducibility
            obj.data_validation = subset.copy()
            obj.data_training = obj.data_observed.drop(subset.index).copy()
        elif callable(validation):
            obj.data_validation = validation(obj.data_observed).copy()
            obj.data_training = fsetdiff(obj.data_observed.copy(), obj.data_validation, all=True)  # Efficient set difference using pandas
        else:
            obj.data_validation = validation.copy()
            obj.data_training = fsetdiff(obj.data_observed.copy(), obj.data_validation, all=True)  # Efficient set difference using pandas

    else:
        obj.data_training = obj.data_observed.copy()

    return obj
    