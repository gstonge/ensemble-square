import pandas as pd
import numpy as np
import datetime
from scorepi import *
from epiweeks import Week
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import datetime
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import matplotlib as mpl
import random
import sys

import warnings
warnings.filterwarnings('ignore')

def main(argv):

    it = str(argv[1])
    numsamp = int(float(argv[2]))

    locations = pd.read_csv('./dat/locations.csv',dtype={'location':str})

    # raw score
    modelsall = ['JHU_IDD-CovidSP','NotreDame-FRED', 'UTA-ImmunoSEIRS', 'UVA-adaptive', 'USC-SIkJalpha', 
                'UVA-EpiHiper', 'UNCC-hierbin', 'MOBS_NEU-GLEAM_COVID']
    #modelsall = ['USC-SIkJalpha', 'UVA-EpiHiper', 'UNCC-hierbin', 'MOBS_NEU-GLEAM_COVID']

    rd =17 
    numsamps = numsamp

    start_week = Week(2023, 16)
    max_date = pd.to_datetime('2023-09-01')

    #loclist = list(predictions.location.unique())
    #loclist.remove('US')

    samplewisdf = pd.DataFrame()

    predictionsall = pd.DataFrame()
    i=0
    for model in modelsall:
        #print(model)
        predictions = pd.read_parquet(f'./dat/{model}_rd{rd}_trajectories.pq')
        predictions['Model'] = model
        predictions['trajectory_id'] = predictions['type_id'] + 100*i
        predictionsall = pd.concat([predictionsall, predictions])
        i += 1
        
    loclist = list(predictionsall.location.unique())
    loclist.remove('US')

    for loc in loclist:
        for scenario in ['A', 'B', 'C', 'D', 'E', 'F']:
            #scenario = 'B'
            location = loc
            target = 'hosp'
            incidence = True

            if target == 'hosp':
                target_obs = 'hospitalization'
            else:
                target_obs = target_obs

            observations = pd.read_parquet(f"./dat/truth_{'inc' if incidence else 'cum'}_{target_obs}.pq")
            observations['date'] = pd.to_datetime(observations['date'])


            predictionsfilt = predictionsall[(predictionsall.scenario_id == scenario + '-2023-04-16') & \
                                        (predictionsall.location == location) & \
                                        (predictionsall.target == 'inc ' + target)  & \
                                        (predictionsall.target_end_date <= observations.date.unique().max()) & \
                                        (predictionsall.target_end_date >= pd.to_datetime(start_week.startdate()))]

            observations = observations[(observations['date'] >= pd.to_datetime(start_week.startdate())) & \
                                        (observations['date'] <= predictionsfilt.target_end_date.unique().max())]   

            #filter location
            observations = observations[observations['location'] == location]

            #aggregate to weekly
            observations = observations.groupby(['location', pd.Grouper(key='date', freq='W-SAT')]).sum().reset_index()



            a = list(predictionsfilt.trajectory_id.unique())
            samps = random.sample(a, numsamps)

            predictionsfilt = predictionsfilt[predictionsfilt.trajectory_id.isin(samps) ]

            y = np.array(observations.value)
            X = np.array([np.array(predictionsfilt[predictionsfilt.trajectory_id == i].value) \
                            for i in predictionsfilt.trajectory_id.unique()])

            quantiles=[0.01,0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.60,
                        0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.975,0.99]

            Q = np.quantile(X,quantiles,axis=0)

            WIS = np.zeros(X.shape[1])
            for i in range(len(quantiles) // 2):
                interval_range = 100*(quantiles[-i-1]-quantiles[i])
                alpha = 1-(quantiles[-i-1]-quantiles[i])
                IS = interval_score(y,Q[i],Q[-i-1],interval_range)
                WIS += IS['interval_score']*alpha/2
            WIS += 0.5*np.abs(Q[len(quantiles) // 2 +1] - y)

            WIS = np.mean(WIS) / (len(quantiles) // 2 + 0.5)


            if int(loc) <10:
                loc_conv = loc[1]
            else:
                loc_conv = loc  

            newrow = pd.DataFrame({'Label': 'Scenario '+ scenario, 'location':loc_conv, 'sampleWIS':WIS, 
                                'target':target, 'num_samples':numsamps}, index=[0])

            samplewisdf = pd.concat([samplewisdf, newrow])



    samplewisdf = samplewisdf.reset_index()
    samplewisdf = samplewisdf.drop(columns=['index'])   

                
    samplewisdf = pd.merge(samplewisdf, locations, how = 'inner', on = 'location')


    samplewisdf.to_pickle(f'./samplewis/sampledwis_ensemble_rd17_hosp_it{it}_samp{numsamps}.pkl')
    

    print("done")


if __name__ == "__main__":
    main(sys.argv)

