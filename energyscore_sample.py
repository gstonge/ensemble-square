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
#from numba import njit
from energyscore_fcn import energyscore


import warnings
warnings.filterwarnings('ignore')




def main(argv):

    it = str(argv[1])
    numsamp = int(float(argv[2]))

    locations = pd.read_csv('./dat/locations.csv',dtype={'location':str})

    
    # lump all trajectories to together - this forms the ensemble model
  
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

    energyscoresdf = pd.DataFrame()

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

            a = list(predictionsfilt.trajectory_id.unique())
            samps = random.sample(a, numsamps)

            predictionsfilt = predictionsfilt[predictionsfilt.trajectory_id.isin(samps) ]
            
            for i in [predictionsfilt.type_id.unique()[0]]:
                pfilt = predictionsfilt[predictionsfilt.trajectory_id == i]

            
            observations = observations[(observations['date'] >= pd.to_datetime(start_week.startdate())) & \
                                        (observations['date'] <= predictionsfilt.target_end_date.unique().max())]   

            #filter location
            observations = observations[observations['location'] == location]

            #aggregate to weekly
            observations = observations.groupby(['location', pd.Grouper(key='date', freq='W-SAT')]).sum().reset_index()

            #transform to Observation object
            observations = Observations(observations)


            y = np.array(observations.value)
            X = [np.array(predictionsfilt[predictionsfilt.trajectory_id == i].value) for i in predictionsfilt.trajectory_id.unique()]
            
            ES = energyscore(np.array(X),y)

            #energyscores[loc][scenario] = ES

            if int(loc) <10:
                loc_conv = loc[1]
            else:
                loc_conv = loc  

            newrow = pd.DataFrame({'Label': 'Scenario '+ scenario, 'location':loc_conv, 'energyscore':ES, 
                                'target':target, 'num_samples':numsamps}, index=[0])

            energyscoresdf = pd.concat([energyscoresdf, newrow])
            
    energyscoresdf = energyscoresdf.reset_index()
    energyscoresdf = energyscoresdf.drop(columns=['index'])   

                
    energyscoresdf = pd.merge(energyscoresdf, locations, how = 'inner', on = 'location')


    energyscoresdf.to_pickle(f'./samplescores/energyscore_ensemble_sample_rd17_hosp_it{it}_samp{numsamps}.pkl')

    print('done')

if __name__ == "__main__":
    main(sys.argv)


