# plotting libraries
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np

csfont = {'fontname':'cmr10'}
plt.rcParams.update({'font.size': 22})




variables = ['mort_30day_hosp_z',
            'child_ec_se_county',
            'child_high_exposure_county',
            'child_exposure_county',
            'ec_grp_mem_high_county',
            'cs00_seg_inc',
            'pop_d_2000_1980',
            'exposure_grp_mem_county',
            'bias_grp_mem_high_county',
            'hhinc00',
            'cs_labforce',
            'cs_fam_wkidsinglemom']


inflow_y = [1.128272, 0.527972, 0.509775, 0.482833, 0.385636, 0.319593, 0.257645, 0.250013, 0.192881, 0.148001, 0, 0]
outflow_y = [1.677961, 0.489639, 0.336419, 1.241721, 0.138468, 0.290575, 0, 0.194291, 0.127374, 0, 0.251971, 0.139069]


# creating the bar plot
fig = plt.figure(figsize = (10, 5))
plt.style.use('ggplot')
width = 0.4
x = np.arange(len(variables))

# plot data in grouped manner of bar type
plt.bar(x-0.2, inflow_y, width, label='migration inflow coefficient')
plt.bar(x+0.2, outflow_y, width, label='migration outflow coefficient')


xtick_labels = [xtick.upper() for xtick in variables]
plt.xticks(x, xtick_labels, rotation=32, ha='right', fontsize=10, **csfont)
plt.yticks(fontsize=10, **csfont)
plt.title("Migration Regression Coefficients", fontsize=18, **csfont)
plt.legend(loc='best', prop = {"family": 'cmr10' })
plt.xlabel("Dependent Variable Name", fontsize=16, **csfont)
plt.ylabel("Coefficient Magnitude", fontsize=16, **csfont)

# save figure
plt.savefig('./migration_regression_coefficients.png', dpi=300, bbox_inches="tight")
