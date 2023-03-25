import streamlit as st  
import pandas as pd
import numpy as np
import re
import os
import base64
import pickle
from rank_bm25 import BM25Okapi

from streamlit_agraph import agraph, Node, Edge, Config
st.set_page_config(layout="wide", page_title="SuperMind Design")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

names = ['Sense', 'Remember','Decide','Create','Learn','Illuminate network',	'Incentivize',	'Feed',	'Collaborate_','Community',	'Market',	'Ecosystem',	'Democracy','Connect','Curate','Collaborate_','Compute','Consumer / retail',	'Healthcare',	'Public sector, NGO',
         'Manufacturing hardw., Infra',	'High Tech (software)',	'Financial services',	'Professional services',	
            'Media, telco, entertainment, hospitality',	'Agriculture',	'Energy, nat. resources',	'Education and academia',	'Supply chain, real estate' ]


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
    df.columns = [re.sub(r"Collaborate ", r"Collaborate_", col).strip() for col in df.columns]
    df['Case'] = ''
    #concatenate the column name to Who / What column where ++ or + is present
    for col in names:
        df['Case'] = df['Case'] + df[col].apply(lambda x: str(col) + ' ' if x in [1,2] else '')
    df['text_details'] = df['Case'] + ' ' + df['Description'] + ' ' + df['Use case'] + ' ' + df['Who / What']
    return df

def get_data():
    if not os.path.exists('data.csv'):
        db = clean_data()
        db.to_csv('data.csv', index=False)
    db = pd.read_csv('data.csv')
    if len(db) == len(pd.read_csv("https://docs.google.com/spreadsheets/d/1RviBVCNh5FaYaNjoMAgCncRBHBFtfyt6XXaKO4f4Wek/edit?usp=sharing".replace('/edit?usp=sharing', '/export?format=csv&gid=0'), header=[1])):
        pass
    else:
        db = clean_data()
        db.to_csv('data.csv', index=False)
    db.columns = [re.sub(r"Collaborate ", r"Collaborate_", col).strip() for col in db.columns]
    return db
db = get_data()
st.header('Supermind.design database output:')
# st.write(db)

def clean_text(text):
    dfhat = pd.read_csv('data.csv')
    dfhat['Case'] = ''
    #concatenate the column name to Who / What column where ++ or + is present
    for col in names:
        dfhat['Case'] = dfhat['Case'] + dfhat[col].apply(lambda x: str(col) + ' ' if x in [1,2] else '')
    dfhat['text_details'] = dfhat['Case'] + ' ' + dfhat['Description'] + ' ' + dfhat['Use case'] + ' ' + dfhat['Who / What']
    docs = dfhat['text_details']
    docs = [doc.split(" ") for doc in docs]
    bm25 = BM25Okapi(docs)
    pickle.dump(bm25, open('bm25.pkl', 'wb'))
    return bm25

def get_docs_embeddings():
    db = get_data()
    db.to_csv('data.csv', index=False)
    if len(db) == len(get_data()) and os.path.exists('doc_embeddings_dict.pkl'):
        with open('bm25.pkl', 'rb') as f:
            doc_embeddings_dict = pickle.load(f)
    else:
        doc_embeddings_dict = clean_text(db)
    return doc_embeddings_dict

def get_similar_docs(query,corpus, top_n=5):
    doc_embeddings_dict = get_docs_embeddings()
    tokenized_query = query.split(" ")
    similar_items = doc_embeddings_dict.get_top_n(tokenized_query,corpus , n=top_n)
    return similar_items  

col1, col2 = st.sidebar.columns(2)
# button = col1.button('Graph (Beta)')
button = st.sidebar.radio('Select a View:', ['Table','Graph'], index=0)
# button2 = col2.button('View Table')
query = st.sidebar.text_input('Search', value='', key=None, type='default')
process = st.sidebar.multiselect('Process',['Sense', 'Remember','Decide','Create','Learn'])
module = st.sidebar.multiselect('Module',['Illuminate network',	'Incentivize',	'Feed',	'Collaborate'])
group = st.sidebar.multiselect('Group',['Community',	'Market',	'Ecosystem',	'Democracy'])
augmentation = st.sidebar.multiselect('Augmentation',['Connect','Curate','Collaborate ','Compute'])
sector = st.sidebar.multiselect('Specific Sector',['Consumer / retail',	'Healthcare',	'Public sector, NGO',	
            'Manufacturing hardw., Infra',	'High Tech (software)',	'Financial services',	'Professional services',	
            'Media, telco, entertainment, hospitality',	'Agriculture',	'Energy, nat. resources',	'Education and academia',	'Supply chain, real estate'])


cols = process + [w if w != 'Collaborate ' else 'Collaborate_' for w in augmentation ] +module+ group + sector


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
    if query != '':
        idxs = [np.where(db['text_details'] == doc)[0][0] for doc in get_similar_docs(query, list(db['text_details']), 5)]
        db = db.loc[idxs, :]
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
    
    dfhat = db
    if query != '':
        idxs = [np.where(dfhat['text_details'] == doc)[0][0] for doc in get_similar_docs(query, list(dfhat['text_details']), 15)]
        dfhat = dfhat.loc[idxs, :]
        
    # dfhat = db.iloc[np.where(db[np.where(~db[p].isna()) for p in process].contains([1,2])) & np.where(db[augmentation].contains([1,2])) & np.where(db[module].contains([1,2])) & np.where(db[group].contains([1,2])) & np.where(db[sector].contains([1,2]))]
    for p in cols:
        dfhat[p] = dfhat[p].apply(lambda x: float(str(x).split('"')[-1]))
        dfhat = dfhat[dfhat[p].isin([1,2])]
    dfhat.sort_values(by=cols,ascending=False,inplace=True)
    if len(dfhat) == 0:
            st.write('No data found')
    for row in dfhat.iterrows():
        with st.expander(row[1]['Who / What'] + ': ' + row[1]['Use case']):
            st.markdown(', '.join([key+ ':'+ dict({1:'+',2:'++'})[int(value)] for key,value in dict(row[1][cols]).items() if value != np.nan]))
            st.write(row[1]['Description'])
        # dfhat.drop(columns=['Unnamed: 0'],inplace=True)
    dfhat.to_csv('db_download.csv', index=False)
    with open('db_download.csv', 'rb') as f:
        st.sidebar.download_button('Download filtered Data', f, file_name='db_download.csv', key=None, mime='text/csv')
    st.sidebar.write('Copyright © Supermind.design Creative Commons (share, adapt, credit) license')
        # st.sidebar.download_link(dfhat.to_csv('db_download.csv'))



            