
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


def preprocess(X_df):
    df = X_df.copy()
    
    categories = ['foosdfpfkusacimwkcsosbicdxkicaua',
 'MISSING',
 'lmkebamcaaclubfxadlmueccxoimlema',
 'usilxuppasemubllopkaafesmlibmsdf',
 'ewpakwlliwisiwduibdlfmalxowmwpci',
 'sddiedcslfslkckwlfkdpoeeailfpeds']
    df[[f'channel_sales: {x}' for x in categories]] = df['channel_sales'].apply(lambda x: pd.Series([1 if y == x else 0 for y in categories]))
    df.drop('channel_sales', axis = 1, inplace = True)
    
    categories = ['lxidpiddsbxsbosboudacockeimpuepw',
 'kamkkxfxxuwbdslkwifmmcsiusiuosws',
 'ldkssxwpmemidmecebumciepifcamkci',
 'MISSING']
    df[[f'origin_up: {x}' for x in categories]] = df['origin_up'].apply(lambda x: pd.Series([1 if y == x else 0 for y in categories]))
    df.drop('origin_up', axis = 1, inplace = True)
    
    columns = ['cons_12m',
 'cons_gas_12m',
 'cons_last_month',
 'forecast_cons_12m',
 'forecast_cons_year',
 'forecast_discount_energy',
 'forecast_meter_rent_12m',
 'forecast_price_energy_off_peak',
 'forecast_price_energy_peak',
 'forecast_price_pow_off_peak',
 'has_gas',
 'imp_cons',
 'margin_gross_pow_ele',
 'margin_net_pow_ele',
 'nb_prod_act',
 'net_margin',
 'num_years_antig',
 'pow_max',
 'has_electricity',
 'average_off_peak_fix',
 'average_peak_fix',
 'average_mid_peak_fix',
 'average_6m_off_peak_fix',
 'average_6m_peak_fix',
 'average_6m_mid_peak_fix',
 'average_3m_off_peak_fix',
 'average_3m_peak_fix',
 'average_3m_mid_peak_fix',
 'months_active',
 'channel_sales: foosdfpfkusacimwkcsosbicdxkicaua',
 'channel_sales: MISSING',
 'channel_sales: lmkebamcaaclubfxadlmueccxoimlema',
 'channel_sales: usilxuppasemubllopkaafesmlibmsdf',
 'channel_sales: ewpakwlliwisiwduibdlfmalxowmwpci',
 'channel_sales: sddiedcslfslkckwlfkdpoeeailfpeds',
 'origin_up: lxidpiddsbxsbosboudacockeimpuepw',
 'origin_up: kamkkxfxxuwbdslkwifmmcsiusiuosws',
 'origin_up: ldkssxwpmemidmecebumciepifcamkci',
 'origin_up: MISSING']
    columns_removed = ['channel_sales: foosdfpfkusacimwkcsosbicdxkicaua',
 'channel_sales: MISSING',
 'channel_sales: lmkebamcaaclubfxadlmueccxoimlema',
 'channel_sales: usilxuppasemubllopkaafesmlibmsdf',
 'channel_sales: ewpakwlliwisiwduibdlfmalxowmwpci',
 'channel_sales: sddiedcslfslkckwlfkdpoeeailfpeds',
 'origin_up: lxidpiddsbxsbosboudacockeimpuepw',
 'origin_up: kamkkxfxxuwbdslkwifmmcsiusiuosws',
 'origin_up: ldkssxwpmemidmecebumciepifcamkci',
 'origin_up: MISSING']
    for column in columns_removed:
        columns.remove(column)
        
    for column in columns:
        df[f'{column} Edited'] = df[column]
        df.drop(column,axis = 1, inplace = True)
    
    return df

from sklearn.base import BaseEstimator, TransformerMixin

class feature_transformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        
        return preprocess(X)
