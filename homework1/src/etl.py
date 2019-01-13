import utils
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    # First convert times to pandas timestamp
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    mortality['timestamp'] = pd.to_datetime(mortality['timestamp'])
    
    # Set the index date for the dead
    mortality = mortality.assign(indx_date=lambda x: x.timestamp - pd.DateOffset(days=30))[['patient_id', 'indx_date']]
    
    df_dead = pd.merge(events, mortality, on='patient_id', how='inner')

    df_alive = events[~events['patient_id'].isin(set(df_dead['patient_id']))]

    # Do groupby for df_alive to extract max event date
    alive_idx = df_alive.groupby('patient_id')['patient_id','timestamp'].max()
    
    alive_idx = pd.DataFrame(alive_idx).rename(columns={'timestamp': 'indx_date'})
    
    indx_date = pd.concat([alive_idx, mortality])
    
    # Save to disk
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    join = pd.merge(events, indx_date, left_on='patient_id', right_on='patient_id')
    join['timestamp'] = pd.to_datetime(join['timestamp'])
    join['indx_date'] = pd.to_datetime(join['indx_date'])
    join['timedelta'] = (join['indx_date'] - join['timestamp']).apply(lambda x: x.days)
    join = join[(join['timedelta'] <= 2000) & (join['timedelta'] >= 0)]
    
    
    filtered_events = join[['patient_id', 'event_id', 'value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    columns=['patient_id', 'feature_id', 'feature_value']
    lab = filtered_events_df[filtered_events_df['event_id'].str.contains('LAB')]
    lab['event_id'] = lab['event_id'].map(feature_map_df.set_index('event_id')['idx'])
    lab = lab.dropna(subset=['value'])
    agg_lab = lab.groupby(['patient_id', 'event_id'], as_index=False).count()
    agg_lab.rename(columns={'event_id':'feature_id'}, inplace=True)
    agg_lab_max = agg_lab.groupby(['feature_id'], as_index=False).agg({"value":"max"})
    merged_lab = pd.merge(agg_lab, agg_lab_max, left_on="feature_id", right_on="feature_id")
    merged_lab['feature_value'] = merged_lab['value_x'] / merged_lab['value_y']
    merged_lab = merged_lab[columns]
    
    dg = filtered_events_df[filtered_events_df['event_id'].str.contains('DRUG')
                                           | filtered_events_df['event_id'].str.contains('DIAG')]
    dg['event_id'] = dg['event_id'].map(
        feature_map_df.set_index('event_id')['idx'])
    dg = dg.dropna(subset=['value'])
    agg_dg = dg.groupby(['patient_id','event_id'], as_index=False).agg({"value":"sum"})
    agg_dg.rename(columns={'event_id': 'feature_id', 'value':'feature_value'}, inplace=True)
    agg_dg_max = agg_dg.groupby(['feature_id'], as_index=False).agg({"feature_value":"max"})
    merged = pd.merge(agg_dg, agg_dg_max, left_on="feature_id", right_on="feature_id")
    merged['feature_value'] = merged['feature_value_x'] / merged['feature_value_y']
    merged = merged[columns]
    aggregated_events = pd.concat([merged_lab, merged])
    
    # Save to disk
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    #Features dict
    # Perfect application of python's defaultdict 
    from collections import defaultdict
    patient_features = defaultdict(list)
    for _, row in aggregated_events.iterrows():
        patient_features[row['patient_id']].append( (row['feature_id'], row['feature_value']) )
    
    # Mortality dict
    patients = aggregated_events.groupby(['patient_id'], as_index=False).count()
    mapping = pd.merge(patients, mortality[['patient_id', 'label']], on='patient_id', how='left').fillna(0)[['patient_id', 'label']]
    zipped = list(zip(mapping['patient_id'], mapping['label']))
    mortality = dict(zipped)

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    del1 = ''
    del2 = ''

    for key, value in sorted(patient_features.items()):
        del1 += str(int(key)) + ' ' + str(mortality[key]) + ' '
        del2 += str(mortality[key]) + ' '
        value = sorted(value)
        for v in value:
            del1 += str(int(v[0])) + ':' + str(format(v[1], '.6f')) + ' '
            del2 += str(int(v[0])) + ':' + str(format(v[1], '.6f')) + ' '
        del1 += '\n'
        del2 += '\n'
        
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    deliverable1.write(bytes((del2),'UTF-8')) #Use 'UTF-8'
    deliverable2.write(bytes((del1),'UTF-8'))

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()
