import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, learning_curve, LearningCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)


# ==================================================================================================================== #


def target(x):
    return np.vectorize(lambda t: 1.0 / (1.0 + 10.0 * t ** 2))(x)
    # return np.vectorize(lambda t: np.exp(-0.5*t)*np.sin(2*np.pi*t))(x)


def ground_truth(rng: np.random.default_rng, x: np.array, noise: bool = False, seed: int = None, noise_scale: float = 0.1) -> pd.DataFrame:

    y = target(x)

    if noise:
        if seed is None or (not isinstance(noise_scale, float)):  # -- only checking for type, not valid values
            raise ValueError('If `noise is True` then `seed` must be a non-negative integer and `noise_scale` must be a non-negative float.')
        for k in range(y.shape[0]):
            y[k] += rng.normal(loc=0.0, scale=noise_scale * y[k])

    return (
        pd
        .DataFrame({
            'x': x,
            'y': y
        })
        .sort_values(by=['x'])
        .reset_index(drop=True)
    )


def uniform_nodes(n, start=-1.0, stop=1.0):
    return np.linspace(start=start, stop=stop, num=n)


# ==================================================================================================================== #


def run() -> None:

    # ---------------------------------------------------------------------------------------------------------------- #
    # -- SETTINGS -- #
    # ---------------------------------------------------------------------------------------------------------------- #
    seed = 42
    NOISE = True
    # NOISE = False
    rng = np.random.default_rng(seed=seed)

    # ---------------------------------------------------------------------------------------------------------------- #
    # -- DATA GENERATION -- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Ground truth at uniform nodes -- #
    nobs = 1_000
    start = -2.0
    stop = -start
    lower_train_limit = -1.0
    upper_train_limit = -lower_train_limit

    df_gt = ground_truth(rng=rng, x=uniform_nodes(n=nobs, start=start, stop=stop), noise=NOISE, seed=seed, noise_scale=0.08)
    df_train = df_gt[(df_gt.x >= lower_train_limit) & (df_gt.x <= upper_train_limit)]
    df_vldn = df_gt[(df_gt.x < lower_train_limit) | (df_gt.x > upper_train_limit)]

    # ---------------------------------------------------------------------------------------------------------------- #
    # -- POLYNOMIAL REGRESSION -- #
    # ---------------------------------------------------------------------------------------------------------------- #
    max_poly_degree = 25
    degrees = range(1, max_poly_degree + 1)
    grid_search_splits = 5
    shuffle = True

    regression_pipeline = Pipeline([
        ('polynomial_features', PolynomialFeatures(include_bias=False)),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    param_grid = [
        {'polynomial_features__degree': degrees}
    ]

    cv = KFold(n_splits=grid_search_splits, shuffle=shuffle, random_state=seed)

    grid_search = GridSearchCV(
        regression_pipeline,
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        return_train_score=True
    )

    # -- Model fit -- #
    grid_search.fit(df_train[['x']], df_train[['y']])
    df_cv_res = pd.DataFrame(grid_search.cv_results_)

    # -- Make the CV results into a data frame with the minimum necessary information. -- #
    time_cols = [_ for _ in df_cv_res.columns.values if _.endswith('_time')]
    cols = [_ for _ in df_cv_res.columns.values if _ not in (time_cols + ['params'])]
    df_cv_res_lean = df_cv_res[cols]
    df_cv_res_lean = df_cv_res_lean.rename(columns={'param_polynomial_features__degree': 'Degree'})
    df_cv_res_lean = df_cv_res_lean[['Degree', 'rank_test_score'] + [_ for _ in df_cv_res_lean.columns.values if (_.startswith('mean_') or _.startswith('std_'))]]

    # ---------------------------------------------------------------------------------------------------------------- #
    # -- VALIDATION ERROR -- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- For each degree, fit the model on the entire training data set and evaluate on the validation set. -- #
    validation_error = []
    for d in df_cv_res_lean.Degree:
        regression_pipeline = Pipeline([
            ('polynomial_features', PolynomialFeatures(degree=d, include_bias=False)),
            ('regression', LinearRegression(fit_intercept=True))
        ])
        regression_pipeline.fit(X=df_train[['x']], y=df_train[['y']])
        y_vldn = regression_pipeline.predict(df_vldn[['x']])
        validation_error.append(mean_squared_error(df_vldn[['y']], y_vldn))

    # ---------------------------------------------------------------------------------------------------------------- #
    # -- PLOTTING -- #
    # ---------------------------------------------------------------------------------------------------------------- #
    markersize = 6

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(df_cv_res_lean.Degree, -df_cv_res_lean.mean_train_score, 'b--', marker='.', markersize=markersize, label='Avg training error (over folds)')
    ax[0].plot(df_cv_res_lean.Degree, -df_cv_res_lean.mean_test_score, 'k--', marker='.', markersize=markersize, label='Avg test error (over folds)')
    ax[0].grid()
    ax[0].set_xticks(df_cv_res_lean.Degree.values)
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel('average training and test error from grid search')

    ax[1].plot(df_cv_res_lean.Degree, np.log(validation_error), 'r--', marker='.', markersize=markersize, label='Validation error in log y-scale')
    ax[1].grid()
    ax[1].set_xticks(df_cv_res_lean.Degree.values)
    ax[1].legend(loc='upper left')
    plt.xlabel('polynomial degree')
    plt.ylabel('log(validation error)')

    # ---------------------------------------------------------------------------------------------------------------- #

    plt.show()


# ==================================================================================================================== #


if __name__ == '__main__':
    run()
