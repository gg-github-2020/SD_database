from ctypes import alignment
from lib2to3.pgen2.token import NEWLINE
import streamlit as st  
import pandas as pd
import numpy as np
from itertools import chain

import os
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code
st.header("Welcome to Supermind desgin's collective intelligence Knowledge Graph")
gif_html = get_img_with_href('super.jpg', 'https://www.supermind.design/')
st.sidebar.markdown(gif_html, unsafe_allow_html=True)

# st.sidebar.image('super.jpg',caption='Supermind Design',use_column_width=True)
db = pd.read_csv('db_updated.csv')
# st.write(db)

process = st.sidebar.multiselect('Process',['Sense', 'Remember','Decide','Create','Learn'])
augmentation = st.sidebar.multiselect('Augmentation',['Connect','Curate','Collaborate','Compute'])
module = st.sidebar.multiselect('Module',['Illuminate network',	'Incentivize',	'Feed',	'Collaborate_M'])
group = st.sidebar.multiselect('Group',['Community',	'Market',	'Ecosystem',	'Democratic "voting"',	'Democracy'])
sector = st.sidebar.multiselect('Specific Sector',['Consumer / retail',	'Healthcare',	'Public sector, NGO',	
            'Manufacturing hardw., Infra',	'High Tech (software)',	'Financial services',	'Professional services',	
            'Media, telco, entertainment',	'Agriculture',	'Energy, nat. resources',	'Education and academia',	'Supply chain, real estate'])

if process or augmentation or module or group or sector:
    #filter by selection in sidebar consider only those rows which have 1,2 values in all columns
    cols = process + augmentation + module + group + sector
    # dfhat = db.iloc[np.where(db[np.where(~db[p].isna()) for p in process].contains([1,2])) & np.where(db[augmentation].contains([1,2])) & np.where(db[module].contains([1,2])) & np.where(db[group].contains([1,2])) & np.where(db[sector].contains([1,2]))]
    dfhat = db
    for p in cols:
        dfhat = dfhat[dfhat[p].isin([1,2])]
    dfhat.sort_values(by=cols,ascending=False,inplace=True)
    for row in dfhat.iterrows():
        with st.expander(row[1]['Who / What'] + ': '+row[1]['Use case']):
            st.markdown(', '.join([key+ ':'+ dict({1:'+',2:'++'})[int(value)] for key,value in dict(row[1][cols]).items() if value != np.nan]))
            st.write(row[1]['Description'])

    dfhat.to_csv('db_download.csv')
    with open('db_download.csv', 'rb') as f:
        st.sidebar.download_button('Download filtered Data', f, file_name='db_download.csv')
# st.sidebar.download_link(dfhat.to_csv('db_download.csv'))



            