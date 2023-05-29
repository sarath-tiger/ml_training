import os
import pickle

# import tarfile
import warnings

# import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import randint

# from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from my_ml.logger import log_initialize, logging

# from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
log_initialize(os.path.basename(__file__))


def income_cat_proportions(data):
    """
    This module will add a column "income_cat" for the input dataframe

    Args:
        Dataframe

    Returns:
        Dataframe with added income_cat column
    """
    logging.info("getting income cat proportions")
    return data["income_cat"].value_counts() / len(data)


def housing_pre_process_eda(config):
    """
    This module has pre processing and EDA algorithm

    Args:
        config: A dictionary with split_data_path (splitted data path)

    Returns:
        Boolean
    """
    logging.info("Starting housing preprocess & EDA")
    logging.info("split_data_path--> {}".format(config["split_data_path"]))
    housing = pd.read_csv(os.path.join(config["split_data_path"], "housing.csv"))
    logging.info("Reading the required data files")
    strat_test_set = pd.read_csv(
        os.path.join(config["split_data_path"], "strat_test_set.csv")
    )
    strat_train_set = pd.read_csv(
        os.path.join(config["split_data_path"], "strat_train_set.csv")
    )
    # train_set = pd.read_csv(os.path.join(config["split_data_path"], "train_set.csv"))
    test_set = pd.read_csv(os.path.join(config["split_data_path"], "test_set.csv"))
    logging.info("Generating income cat proportions")
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    logging.info("dropping income cat column from strat train and test set")
    housing = strat_train_set.copy()
    logging.info("Generating the scatter plot")
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    logging.info("Getting corellated columns ")
    # corr_matrix = housing.corr()
    # corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    logging.info("dropping median house value from strat train set")
    strat_test_set.to_csv(
        os.path.join(config["split_data_path"], "strat_test_set.csv"), index=False
    )
    strat_train_set.to_csv(
        os.path.join(config["split_data_path"], "strat_train_set.csv"), index=False
    )
    logging.info("Saving required data to csv files")
    housing_labels = strat_train_set["median_house_value"].copy()
    logging.info("Initializing imputer module")
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)
    housing_num.to_csv(
        os.path.join(config["split_data_path"], "housing_num.csv"), index=False
    )

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    housing_labels.to_csv(
        os.path.join(config["split_data_path"], "housing_labels.csv"), index=False
    )
    housing_prepared.to_csv(
        os.path.join(config["split_data_path"], "housing_prepared.csv"), index=False
    )
    logging.info("Completed housing preprocess & EDA")
    return True


def housing_model_build(config):
    """
    This module has pre processing and EDA algorithm

    Args:
        config: A dictionary with split_data_path (splitted data path) and ml_model_path (path to save model)

    Returns:
        Boolean
    """

    logging.info("Starting building models")
    housing_labels = pd.read_csv(
        os.path.join(config["split_data_path"], "housing_labels.csv")
    )
    housing_prepared = pd.read_csv(
        os.path.join(config["split_data_path"], "housing_prepared.csv")
    )
    strat_test_set = pd.read_csv(
        os.path.join(config["split_data_path"], "strat_test_set.csv")
    )
    os.makedirs(config["ml_model_path"], exist_ok=True)
    logging.info("Initializing linear regression model")
    lin_reg = LinearRegression()
    logging.info("Fitting data to LR model")
    lin_reg.fit(housing_prepared, housing_labels)
    logging.info("Saving LR model as pickle in {}".format(config["ml_model_path"]))
    pickle.dump(
        lin_reg, open(os.path.join(config["ml_model_path"], "lin_reg.pkl"), "wb")
    )
    logging.info("Initializing decision tree regression model")
    tree_reg = DecisionTreeRegressor(random_state=42)
    logging.info("Fitting data to decision tree regression model")
    tree_reg.fit(housing_prepared, housing_labels)
    logging.info(
        "Saving tree model as pickle file in {}".format(config["ml_model_path"])
    )
    pickle.dump(
        tree_reg, open(os.path.join(config["ml_model_path"], "tree_reg.pkl"), "wb")
    )

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    logging.info("Initializing random forest regressor with ramdomized search CV")
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    logging.info("Fitting data for randmoized search CV")
    rnd_search.fit(housing_prepared, housing_labels)
    pickle.dump(
        rnd_search, open(os.path.join(config["ml_model_path"], "rnd_search.pkl"), "wb")
    )
    # cvres = rnd_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    # print(np.sqrt(-mean_score), params)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    logging.info("Initializing Random Forest Regressor model with grid search CV")
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    logging.info("Fitting data for grid search")
    grid_search.fit(housing_prepared, housing_labels)
    logging.info("Getting best params for Random Forest Regressor model")
    print(grid_search.best_params_)
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    # print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    logging.info("Initializing the final model with best parameter")
    final_model = grid_search.best_estimator_
    logging.info(
        "Dumping the final model as pickle file in {}".format(config["ml_model_path"])
    )
    pickle.dump(
        final_model,
        open(os.path.join(config["ml_model_path"], "final_model.pkl"), "wb"),
    )
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    y_test.to_csv(os.path.join(config["split_data_path"], "y_test.csv"), index=False)

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    housing_num = pd.read_csv(
        os.path.join(config["split_data_path"], "housing_num.csv")
    )
    logging.info("Generating csv files for test and validation")
    imputer.fit(housing_num)
    # print(X_test_num.head())
    # print(housing_num.head())
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))
    X_test_prepared.to_csv(
        os.path.join(config["split_data_path"], "X_test_prepared.csv"), index=False
    )
    logging.info("train script completed..!!")

    return True
