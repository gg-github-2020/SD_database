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
gif_html = get_img_with_href('super.jpg', 'https://www.supermind.design/')
st.sidebar.markdown(f'<div style="text-align: center"> {gif_html} </div>', unsafe_allow_html=True)

st.sidebar.write("#")

# st.sidebar.image('super.jpg',caption='Supermind Design',use_column_width=True)
db = pd.read_csv('data (3).csv')
st.header('Supermind.design database output:')
# st.write(db)

process = st.sidebar.multiselect('Process',['Sense', 'Remember','Decide','Create','Learn'])
module = st.sidebar.multiselect('Module',['Illuminate network',	'Incentivize',	'Feed',	'Collaborate'])
group = st.sidebar.multiselect('Group',['Community',	'Market',	'Ecosystem',	'Democracy'])
augmentation = st.sidebar.multiselect('Augmentation',['Connect','Curate','Collaborate ','Compute'])
sector = st.sidebar.multiselect('Specific Sector',['Consumer / retail',	'Healthcare',	'Public sector, NGO',	
            'Manufacturing hardw., Infra',	'High Tech (software)',	'Financial services',	'Professional services',	
            'Media, telco, entertainment',	'Agriculture',	'Energy, nat. resources',	'Education and academia',	'Supply chain, real estate'])

if process or augmentation or module or group or sector:
    #filter by selection in sidebar consider only those rows which have 1,2 values in all columns
    cols = process + augmentation +module+ group + sector
    # dfhat = db.iloc[np.where(db[np.where(~db[p].isna()) for p in process].contains([1,2])) & np.where(db[augmentation].contains([1,2])) & np.where(db[module].contains([1,2])) & np.where(db[group].contains([1,2])) & np.where(db[sector].contains([1,2]))]
    dfhat = db
    if len(dfhat) == 0:
        st.write('No data found')
    else:
        for p in cols:
            dfhat[p] = dfhat[p].apply(lambda x: float(str(x).split('"')[-1]))
            dfhat = dfhat[dfhat[p].isin([1,2])]
        dfhat.sort_values(by=cols,ascending=False,inplace=True)
        for row in dfhat.iterrows():
            with st.expander(row[1]['Who / What'] + ': ' + row[1]['Use case']):
                st.markdown(', '.join([key+ ':'+ dict({1:'+',2:'++'})[int(value)] for key,value in dict(row[1][cols]).items() if value != np.nan]))
                st.write(row[1]['Description'])
    # dfhat.drop(columns=['Unnamed: 0'],inplace=True)
    dfhat.to_csv('db_download.csv', index=False)
    with open('db_download.csv', 'rb') as f:
        st.sidebar.download_button('Download filtered Data', f, file_name='db_download.csv')
    st.sidebar.write('Copyright © Supermind.design Creative Commons (share, adapt, credit) license')
    # st.sidebar.download_link(dfhat.to_csv('db_download.csv'))



            