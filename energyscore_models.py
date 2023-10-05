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

    locations = pd.read_csv('./dat/locations.csv',dtype={'location':str})

    # raw score
    modelsall = ['JHU_IDD-CovidSP','NotreDame-FRED', 'UTA-ImmunoSEIRS', 'UVA-adaptive', 'USC-SIkJalpha', 
                'UVA-EpiHiper', 'UNCC-hierbin', 'MOBS_NEU-GLEAM_COVID']

    rd =17 

    start_week = Week(2023, 16)

    #loclist = list(predictions.location.unique())
    #loclist.remove('US')

    energyscoresdf = pd.DataFrame()

    #loclist = ['02', '01', '05', '04', '06', '08', '09']

    for model in modelsall:
        print(model)
        predictions = pd.read_parquet(f'./dat/{model}_rd{rd}_trajectories.pq')
        loclist = list(predictions.location.unique())
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
                

                predictionsfilt = predictions[(predictions.scenario_id == scenario + '-2023-04-16') & \
                                            (predictions.location == location) & \
                                            (predictions.target == 'inc ' + target)  & \
                                            (predictions.target_end_date <= observations.date.unique().max()) & \
                                            (predictions.target_end_date >= pd.to_datetime(start_week.startdate()))]

                for i in predictionsfilt.type_id.unique():
                    pfilt = predictionsfilt[predictionsfilt.type_id == i]


                observations = observations[(observations['date'] >= pd.to_datetime(start_week.startdate())) & \
                                            (observations['date'] <= pfilt.target_end_date.unique().max())]

                #filter location
                observations = observations[observations['location'] == location]

                #aggregate to weekly
                observations = observations.groupby(['location', pd.Grouper(key='date', freq='W-SAT')]).sum().reset_index()

                #transform to Observation object
                observations = Observations(observations)


                check = list(observations.date) == list(pfilt.target_end_date)
                if check == False:
                    print('dates do not match')


                N = len(predictionsfilt.type_id.unique())
                M = len(observations)

                # first term
                observations['target_end_date'] = observations['date']
                pfilt = predictionsfilt.merge(observations, how='left', on=['location', 'target_end_date'])
                pfilt['diff_sq']=(pfilt.value_x - pfilt.value_y)**2

                ES1 = 1/N*sum(np.sqrt(np.array(pfilt.groupby(['type_id']).sum()['diff_sq'])))
                
                

                # second term
                
                cross = predictionsfilt.merge(predictionsfilt, how='cross')
                crossfilt = cross.loc[(cross['target_end_date_x'] == cross['target_end_date_y']) ]
                
                crossfilt['diff_sq'] = (crossfilt.value_x - crossfilt.value_y)**2
                ES2 = 1/(2*N**2) * sum(np.sqrt(np.array(crossfilt.groupby(['type_id_x', 'type_id_y']).sum()['diff_sq'])))
                

                ES = ES1 - ES2
                
                if int(loc) <10:
                    loc_conv = loc[1]
                else:
                    loc_conv = loc 

                #energyscores[loc][scenario] = ES


                newrow = pd.DataFrame({'Model':model , 'Label': 'Scenario '+ scenario, 'location':loc_conv, 'energyscore':ES, 
                                    'target':target}, index=[0])

                energyscoresdf = pd.concat([energyscoresdf, newrow])
            
    energyscoresdf = energyscoresdf.reset_index()
    energyscoresdf = energyscoresdf.drop(columns=['index'])   

                
    energyscoresdf = pd.merge(energyscoresdf, locations, how = 'inner', on = 'location')

    energyscoresdf.to_pickle('./energyscore_raw_hosp_rd17_models.pkl')


   

    print('done')

if __name__ == "__main__":
    main(sys.argv)



