"""Climate change and wildfire."""
import analysis


def models_building() -> None:
    """Build models: wildfire(size, frequency) - climate(temperature(max, min, mean), precipitation)

    Note: There will be 20 figures after running this function.
    """
    analysis_ca.analise_wildfire()
    analysis_ca.temp_time(
        max_guess=(16.847, 0.017182, 187.1, -0.649, 0.009168, 1731, 0.000122, 79.56),
        min_guess=(11.92, 0.017182, 188, -0.58, 0.009046, 1737, 0.00002, 54),
        mean_guess=(14.915, 0.017165, 182.8, -0.545, 0.00924, 1799, 0.000001, 57.67),
        test_range=(0.95, 1.05))
    analysis_ca.prcp_time(
        guess=(1.725, 0.0172, 375.58, -0.32274, 0.006797, 1930, -0.000001, 2.17),
        test_range=(0.99, 1.01))


def future_prediction() -> None:
    """Predict the future climate and wildfire frequency based on the best models we have gotten.

    Note: The model that fits wildfire frequency - temperature best is quadratic model.
          The model that fits wildfire frequency - precipitation best is logarithm model.
          There will be 10 figures after running this function.
    """
    analysis_ca.predict_freq_temp_qua(
        (2020, 2039),
        (16.847, 0.017182, 187.1, -0.649, 0.009168, 1731, 0.000122, 79.56),
        (11.92, 0.017182, 188, -0.58, 0.009046, 1737, 0.00002, 54),
        (14.915, 0.017165, 182.8, -0.545, 0.00924, 1799, 0.000001, 57.67))
    analysis_ca.predict_freq_prcp_log(
        2020, 2039,
        (1.725, 0.0172, 375.58, -0.32274, 0.006797, 1930, -0.000001, 2.17))


if __name__ == '__main__':
    analysis_ca = analysis.StateDataAnalysis('CA', (1994, 2013),
                                             'new_ca_climate.csv',
                                             'new_wildfire_data.csv')

    # Please uncomment only one of following functions every time,
    # or there will be too many figures.

    # wildfire_models_building()
    # future_prediction()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['data_prep', 'figure', 'models', 'numpy', 'typing', 'datetime'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E9999', 'R1705', 'C0200']
    })
