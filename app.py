from ctypes import alignment
from lib2to3.pgen2.token import NEWLINE
import streamlit as st  
import pandas as pd
import numpy as np
import re

import os
import base64

from streamlit_agraph import agraph, Node, Edge, Config
st.set_page_config(layout="wide", page_title="SuperMind Design")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# @st.cache(allow_output_mutation=True)
# def get_img_with_href(local_img_path, target_url):
#     img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
#     bin_str = get_base64_of_bin_file(local_img_path)
#     html_code = f'''
#         <a href="{target_url}">
#             <img src="data:image/{img_format};base64,{bin_str}" />
#         </a>'''
#     return html_code
# gif_html = get_img_with_href('super.jpg', 'https://www.supermind.design/')
# st.sidebar.markdown(f'<div style="text-align: center"> {gif_html} </div>', unsafe_allow_html=True)

st.sidebar.write("#")

# st.sidebar.image('super.jpg',caption='Supermind Design',use_column_width=True)
@st.cache(allow_output_mutation=True)
def clean_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1RviBVCNh5FaYaNjoMAgCncRBHBFtfyt6XXaKO4f4Wek/edit?usp=sharing"
    url_1 = sheet_url.replace('/edit?usp=sharing', '/export?format=csv&gid=0')
    df = pd.read_csv(url_1, header=[1])
    def apply_func(x):
        if re.search(r"\+{2}", str(x)) or re.search(r"\*{2}", str(x)):
            return 2
        elif re.search(r"^[^\+]*\+[^\+]*$", str(x)) or re.search(r"^[^\+]*\+[^\+]*$", str(x)):
            return 1
        else:
            return np.nan
    for col in df.columns:
        if col not in ['Who / What', 'Use case', 'Description']:
            df[col] = df[col].apply(apply_func)
    return df

db = clean_data()
st.header('Supermind.design database output:')
# st.write(db)

col1, col2 = st.sidebar.columns(2)
# button = col1.button('Graph (Beta)')
button = st.sidebar.radio('Select a View:', ['Table','Graph'], index=0)
# button2 = col2.button('View Table')

process = st.sidebar.multiselect('Process',['Sense', 'Remember','Decide','Create','Learn'])
module = st.sidebar.multiselect('Module',['Illuminate network',	'Incentivize',	'Feed',	'Collaborate'])
group = st.sidebar.multiselect('Group',['Community',	'Market',	'Ecosystem',	'Democracy'])
augmentation = st.sidebar.multiselect('Augmentation',['Connect','Curate','Collaborate ','Compute'])
sector = st.sidebar.multiselect('Specific Sector',['Consumer / retail',	'Healthcare',	'Public sector, NGO',	
            'Manufacturing hardw., Infra',	'High Tech (software)',	'Financial services',	'Professional services',	
            'Media, telco, entertainment, hospitality',	'Agriculture',	'Energy, nat. resources',	'Education and academia',	'Supply chain, real estate'])


cols = process + augmentation +module+ group + sector

@st.cache(allow_output_mutation=True)
def graph(db):
    
    nodes = []
    edges = []
    dbnew = db[db[cols].notnull().any(axis=1)]
    # only select rows where there is no null in cols
    for col in cols:
        dbnew = dbnew[dbnew[col].notnull()]
    
    # dbnew = dbnew[dbnew[cols].notnull().any(axis=1)]
    dbnew1 = dbnew[cols].sum(axis=1)
    # dbnew['topics'] = [dict() for i in range(len(dbnew))]
    # for i in range(len(dbnew)):
    #     dbnew.iloc[i]['topics'] = {col:dbnew.iloc[i][col] for col in cols if dbnew.iloc[i][col] > 0}
    
    dbnew['score'] = dbnew1
    dbnew = dbnew[['Who / What', 'Use case', 'Description', 'score']][dbnew['score'] > 0]
    dbnew = dbnew.sort_values(by=['score', 'Who / What'], ascending=False)
    # dbnew = dbnew.groupby(['Who / What']).sum().reset_index()
    check = 50 if len(dbnew)//8 > 50 else len(dbnew)
    dbnew = dbnew[:check]
    # st.write(dbnew)
    ls = []
    for i in range(len(dbnew)):
        sr = dbnew.iloc[i]['Who / What']
        sc = dbnew.iloc[i]['score']
        if sr not in ls:
            # st.write(db.iloc[i]['Who / What'])
            ls.append(sr)
            nodes.append(Node(id=sr, label= sr, size= 15, color= nodeColor))
    mx = dbnew['score'].max() + 2
    for i in range(len(dbnew)):
        for j in range(i+1, len(dbnew)):
            if abs(dbnew.iloc[i]['score'] - dbnew.iloc[j]['score']) < 1:
                edges.append(Edge(source=dbnew.iloc[i]['Who / What'], target=dbnew.iloc[j]['Who / What'], length=(2*mx - dbnew.iloc[i]['score'] - dbnew.iloc[j]['score'])*150, color=edgeColor, type="CURVE_SMOOTH"))
    
    config = Config(width=1000, height=800, directed=False, collapsible=False,  nodeHighlightBehavior=True, highlightColor="#F7A7A6", highlightFontSize=20, highlightFontWeight="bold", node={'labelProperty': 'label', 'color': nodeColor, 'size': 30, 'highlightStrokeColor': "SAME"}, link={'highlightColor': edgeColor})
    return {'nodes':nodes, 'edges':edges, 'config':config}
    
  
if button == 'Graph':
    # val = st.slider('Select multiplier for edge length', min_value=100, max_value=600, value=150, step=10)

    # nodeColor = st.color_picker('Pick A Color for Nodes', '#fff333')

    # edgeColor = st.color_picker('Pick A Color for Edges', '#9999CC')

    # NodeSize = st.slider('Select multiplier for node size', min_value=1, max_value=20, value=10, step=1)

    # canvasLength = st.slider('Select multiplier for canvas length', min_value=500, max_value=2000, value=1000, step=10)
    
    nodeColor= '#fff333'
    edgeColor= '#9999CC'
    
    dic = graph(db)
    
    nodes, edges, config = dic['nodes'], dic['edges'], dic['config']
    # nodes = [node for node in nodes if node.id in  or node.id in [edge.to for edge in edges if edge.source in topic]]
    return_value = agraph(nodes=nodes,
                    edges=edges, 
                    config=config)  
    if return_value:
        db_fltr = db[db['Who / What'] == return_value].iloc[0]
        with st.expander(db_fltr['Who / What'] + ': ' + db_fltr['Use case']):
                    # st.markdown(', '.join([key+ ':'+ dict({1:'+',2:'++'})[int(value)] for key,value in dict(db_fltr[]).items() if value != np.nan]))
            st.write(db_fltr['Description'])
    
    
    
else:
    if process or augmentation or module or group or sector:
        #filter by selection in sidebar consider only those rows which have 1,2 values in all columns
       
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



            