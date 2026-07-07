import streamlit as st
import pandas as pd
import numpy as np
import re
from ast import literal_eval
import openai

st.set_page_config(layout="wide", page_title="SuperMind Design")

from streamlit_agraph import agraph, Node, Edge, Config

# ---------------------------------------------------------------------------
# OpenAI setup (fails gracefully so the whole app doesn't crash if the key is
# missing / expired / out of quota -- browsing and filtering still work,
# only the semantic-search box is affected).
# ---------------------------------------------------------------------------
try:
    openai.api_key = st.secrets["api_key"]
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

EMBEDDING_MODEL = "text-embedding-ada-002"
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1RviBVCNh5FaYaNjoMAgCncRBHBFtfyt6XXaKO4f4Wek/export?format=csv&gid=0"
)

names = [
    "Sense", "Remember", "Decide", "Create", "Learn",
    "Illuminate network", "Incentivize", "Feed", "Collaborate_",
    "Community", "Market", "Ecosystem", "Democracy", "Connect",
    "Curate", "Collaborate_", "Compute", "Consumer / retail",
    "Healthcare", "Public sector, NGO", "Manufacturing hardw., Infra",
    "High Tech (software)", "Financial services", "Professional services",
    "Media, telco, entertainment, hospitality", "Agriculture",
    "Energy, nat. resources", "Education and academia",
    "Supply chain, real estate",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_embedding(text):
    text = str(text).replace("\n", " ")
    resp = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
    return resp["data"][0]["embedding"]


def _normalize_cols(df):
    df.columns = [re.sub(r"Collaborate ", "Collaborate_", c).strip() for c in df.columns]
    return df


def _parse_plus(x):
    """Map the sheet's '+'/'++' scoring into 1 / 2 / NaN."""
    s = str(x)
    if re.search(r"\+{2}", s) or re.search(r"\*{2}", s):
        return 2
    if re.search(r"^[^\+]*\+[^\+]*$", s):
        return 1
    return np.nan


TEXT_COLS = ["Who / What", "Use case", "Description"]


def _load_live_sheet():
    """Read the live Google Sheet and parse the +/++ scoring columns.
    Raises on any error or unexpected layout so callers can fall back."""
    df = pd.read_csv(SHEET_CSV_URL, header=[1], on_bad_lines="skip")
    df = _normalize_cols(df)
    for req in TEXT_COLS:
        if req not in df.columns:
            raise ValueError(f"Live sheet missing expected column: {req}")
    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = df[col].apply(_parse_plus)
    df = df.dropna(subset=["Who / What"])
    df = df[df["Who / What"].astype(str).str.strip() != ""]
    if len(df) == 0:
        raise ValueError("Live sheet returned no rows")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Display data: read the LIVE sheet (cached ~10 min) so all current rows show.
# Falls back to the committed snapshot (data.csv) if the live read ever fails.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Loading database…")
def load_data():
    try:
        return _load_live_sheet(), "live Google Sheet"
    except Exception:
        db = pd.read_csv("data.csv", on_bad_lines="skip")
        db = _normalize_cols(db)
        db = db.drop(columns=["Case", "text_details", "embeddings"], errors="ignore")
        return db, "cached snapshot (data.csv)"


# ---------------------------------------------------------------------------
# Search index: pre-computed embeddings from the committed snapshot, keyed by
# row identity so we can score live rows. Only used by the search box.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_embedding_lookup():
    lut = {}
    try:
        snap = pd.read_csv("data.csv", on_bad_lines="skip")
        snap = _normalize_cols(snap)
        for _, r in snap.iterrows():
            key = (str(r.get("Who / What", "")).strip() + "|"
                   + str(r.get("Use case", "")).strip()).lower()
            try:
                lut[key] = np.asarray(literal_eval(r["embeddings"]), dtype=np.float32)
            except Exception:
                pass
    except Exception:
        pass
    return lut


def get_similar_docs(query, df, top_n=10):
    if not OPENAI_OK:
        st.warning("Search is unavailable: OpenAI API key is not configured in app secrets. "
                   "Browsing and filters still work.")
        return df
    try:
        q = np.asarray(_get_embedding(query), dtype=np.float32)
    except Exception as e:
        st.warning(f"Search is temporarily unavailable (OpenAI: {e}). "
                   "Browsing and filters below still work.")
        return df
    lut = load_embedding_lookup()
    if not lut:
        return df
    keys = (df["Who / What"].astype(str).str.strip() + "|"
            + df["Use case"].astype(str).str.strip()).str.lower()
    mask = keys.isin(lut.keys())
    sub = df[mask].copy()
    if len(sub) == 0:
        st.info("No pre-computed search index for these rows yet.")
        return df
    mat = np.vstack([lut[k] for k in keys[mask]])
    sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
    sub["similarity"] = sims
    return sub.sort_values(by="similarity", ascending=False).head(top_n)


db, data_source = load_data()
st.header("Supermind.design database output:")
st.caption(f"{len(db)} records · source: {data_source}")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.write("#")
button = st.sidebar.radio("Select a View:", ["Table", "Graph"], index=0)
query = st.sidebar.text_input("Search Related Ideas", value="", key=None, type="default")
process = st.sidebar.multiselect("Process", ["Sense", "Remember", "Decide", "Create", "Learn"])
module = st.sidebar.multiselect("Module", ["Illuminate network", "Incentivize", "Feed", "Collaborate"])
group = st.sidebar.multiselect("Group", ["Community", "Market", "Ecosystem", "Democracy"])
augmentation = st.sidebar.multiselect("Augmentation", ["Connect", "Curate", "Collaborate ", "Compute"])
sector = st.sidebar.multiselect(
    "Specific Sector",
    [
        "Consumer / retail", "Healthcare", "Public sector, NGO",
        "Manufacturing hardw., Infra", "High Tech (software)",
        "Financial services", "Professional services",
        "Media, telco, entertainment, hospitality", "Agriculture",
        "Energy, nat. resources", "Education and academia",
        "Supply chain, real estate",
    ],
)

cols = (
    process
    + [w if w != "Collaborate " else "Collaborate_" for w in augmentation]
    + module + group + sector
)
cols = [c for c in cols if c in db.columns]


def graph(db, cols, node_color, edge_color):
    nodes, edges = [], []
    dbnew = db[db[cols].notnull().any(axis=1)]
    for col in cols:
        dbnew = dbnew[dbnew[col].notnull()]
    dbnew = dbnew.copy()
    dbnew["score"] = dbnew[cols].sum(axis=1)
    dbnew = dbnew[["Who / What", "Use case", "Description", "score"]][dbnew["score"] > 0]
    dbnew = dbnew.sort_values(by=["score", "Who / What"], ascending=False)
    check = 50 if len(dbnew) // 8 > 50 else len(dbnew)
    dbnew = dbnew[:check]
    ls = []
    for i in range(len(dbnew)):
        sr = dbnew.iloc[i]["Who / What"]
        if sr not in ls:
            ls.append(sr)
            nodes.append(Node(id=sr, label=sr, size=15, color=node_color))
    mx = dbnew["score"].max() + 2 if len(dbnew) else 2
    for i in range(len(dbnew)):
        for j in range(i + 1, len(dbnew)):
            if abs(dbnew.iloc[i]["score"] - dbnew.iloc[j]["score"]) < 1:
                edges.append(
                    Edge(
                        source=dbnew.iloc[i]["Who / What"],
                        target=dbnew.iloc[j]["Who / What"],
                        length=(2 * mx - dbnew.iloc[i]["score"] - dbnew.iloc[j]["score"]) * 150,
                        color=edge_color,
                        type="CURVE_SMOOTH",
                    )
                )
    config = Config(
        width=1000, height=800, directed=False, collapsible=False,
        nodeHighlightBehavior=True, highlightColor="#F7A7A6",
        highlightFontSize=20, highlightFontWeight="bold",
        node={"labelProperty": "label", "color": node_color, "size": 30,
              "highlightStrokeColor": "SAME"},
        link={"highlightColor": edge_color},
    )
    return {"nodes": nodes, "edges": edges, "config": config}


if button == "Graph":
    node_color = "#fff333"
    edge_color = "#9999CC"
    gdb = db
    if query != "":
        gdb = get_similar_docs(query, gdb, top_n=10)
    if len(cols) == 0:
        st.info("Select one or more filters to build the graph.")
    else:
        dic = graph(gdb, cols, node_color, edge_color)
        nodes, edges, config = dic["nodes"], dic["edges"], dic["config"]
        return_value = agraph(nodes=nodes, edges=edges, config=config)
        if return_value:
            db_fltr = gdb[gdb["Who / What"] == return_value].iloc[0]
            with st.expander(db_fltr["Who / What"] + ": " + str(db_fltr["Use case"])):
                st.write(db_fltr["Description"])

else:
    dfhat = db.copy()
    if query != "":
        dfhat = get_similar_docs(query, dfhat, top_n=10)
    if query != "" or len(cols) != 0:
        for p in cols:
            dfhat[p] = pd.to_numeric(dfhat[p], errors="coerce")
            dfhat = dfhat[dfhat[p].isin([1, 2])]
        if query == "" and cols:
            dfhat = dfhat.sort_values(by=cols, ascending=False)
        if len(dfhat) == 0:
            st.write("No data found")
        for row in dfhat.iterrows():
            with st.expander(str(row[1]["Who / What"]) + ": " + str(row[1]["Use case"])):
                if cols:
                    st.markdown(
                        ", ".join(
                            key + ":" + {1: "+", 2: "++"}[int(value)]
                            for key, value in dict(row[1][cols]).items()
                            if pd.notna(value) and int(value) in (1, 2)
                        )
                    )
                st.write(row[1]["Description"])
        download_df = dfhat.drop(columns=["embeddings", "similarity"], errors="ignore")
        st.sidebar.download_button(
            "Download filtered Data",
            download_df.to_csv(index=False).encode("utf-8"),
            file_name="db_download.csv",
            mime="text/csv",
        )
        st.sidebar.write(
            "Copyright © Supermind.design Creative Commons (share, adapt, credit) license"
        )
