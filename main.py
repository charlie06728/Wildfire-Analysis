"""Climate change and wildfire."""
import data_prep
from analysis import StateDataAnalysis

# data_prep.select_cols('wildfire_data.csv', 'wildfire.csv', ['STATE', 'FIRE_YEAR', 'DISCOVERY_DOY', 'FIRE_SIZE'])

if __name__ == '__main__':
    analysis_ca = StateDataAnalysis('CA', 1994, 2013, 'ca_climate.csv', 'wildfire_data3.csv')
    # analysis_ca.analise_wildfire()
    analysis_ca.temp_time(max_guess=(16.847, 0.017182, 187.1, -0.649, 0.009168, 1731, 0.000122, 79.56),
                          min_guess=(11.92, 0.017182, 188, -0.58, 0.009046, 1737, 0.00002, 54),
                          mean_guess=(14.915, 0.017165, 182.8, -0.545, 0.00924, 1799, 0.000001, 57.67),
                          test_range=(0.95, 1.05))
    analysis_ca.prcp_time(guess=(1.725, 0.0172, 375.58, -0.32274, 0.006797, 1930, -0.000001, 2.17),
                          test_range=(0.99, 1.01))
    # analysis_ca.predict_freq_temp_exp(2020, 2039,
    #                                   (16.847, 0.017182, 187.1, -0.649, 0.009168, 1731, 0.000122, 79.56),
    #                                   (11.92, 0.017182, 188, -0.58, 0.009046, 1737, 0.00002, 54),
    #                                   (14.915, 0.017165, 182.8, -0.545, 0.00924, 1799, 0.000001, 57.67))
    # analysis_ca.predict_freq_prcp_log(2020, 2039, (1.725, 0.0172, 375.58, -0.32274, 0.006797, 1930, -0.000001, 2.17))
