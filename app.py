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
# missing/expired -- browsing/filtering still works, only search is disabled)
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
# Helpers (local, so we avoid importing openai.embeddings_utils -- that module
# drags in matplotlib/plotly/scikit-learn and badly slows the cold start)
# ---------------------------------------------------------------------------
def _get_embedding(text):
    text = str(text).replace("\n", " ")
    resp = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
    return resp["data"][0]["embedding"]


def _normalize_cols(df):
    df.columns = [re.sub(r"Collaborate ", "Collaborate_", c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Load the pre-computed database from disk ONCE and cache it.
# The old code re-read this 34 MB file AND re-downloaded the whole Google
# Sheet on every single rerun (every keystroke/filter click) -- that was the
# main cause of the slowness.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading database…")
def load_data():
    db = pd.read_csv("data.csv", on_bad_lines="warn")
    db = _normalize_cols(db)
    if "embeddings" in db.columns:
        db["embeddings"] = db["embeddings"].apply(
            lambda s: np.asarray(literal_eval(s), dtype=np.float32)
            if isinstance(s, str) else np.zeros(1, dtype=np.float32)
        )
    return db


# Manual, opt-in rebuild from the Google Sheet (recomputes OpenAI embeddings).
# This used to run automatically on every load and would hang/cost money.
def rebuild_from_sheet():
    df = pd.read_csv(SHEET_CSV_URL, header=[1], on_bad_lines="warn")

    def apply_func(x):
        if re.search(r"\+{2}", str(x)) or re.search(r"\*{2}", str(x)):
            return 2
        elif re.search(r"^[^\+]*\+[^\+]*$", str(x)):
            return 1
        return np.nan

    for col in df.columns:
        if col not in ["Who / What", "Use case", "Description"]:
            df[col] = df[col].apply(apply_func)
    df = _normalize_cols(df)

    df["Case"] = ""
    for col in names:
        if col in df.columns:
            df["Case"] = df["Case"] + df[col].apply(
                lambda x: str(col) + " " if x in [1, 2] else ""
            )
    df["text_details"] = (
        df["Case"].fillna("") + " " + df["Description"].fillna("") + " "
        + df["Use case"].fillna("") + " " + df["Who / What"].fillna("")
    ).str.lower()
    df["embeddings"] = df["text_details"].apply(_get_embedding)
    df.to_csv("data.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Semantic search -- vectorized cosine similarity against the pre-computed
# embedding matrix (old code parsed every embedding string per row per query).
# ---------------------------------------------------------------------------
def get_similar_docs(query, df, top_n=10):
    if not OPENAI_OK:
        st.warning("Search is unavailable: OpenAI API key is not configured in app secrets.")
        return df
    try:
        q = np.asarray(_get_embedding(query), dtype=np.float32)
    except Exception as e:
        st.warning(f"Search failed (OpenAI): {e}")
        return df
    mat = np.vstack(df["embeddings"].to_numpy())
    sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
    out = df.copy()
    out["similarity"] = sims
    return out.sort_values(by="similarity", ascending=False).head(top_n)


db = load_data()
st.header("Supermind.design database output:")


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
    dic = graph(gdb, cols, node_color, edge_color)
    nodes, edges, config = dic["nodes"], dic["edges"], dic["config"]
    return_value = agraph(nodes=nodes, edges=edges, config=config)
    if return_value:
        db_fltr = gdb[gdb["Who / What"] == return_value].iloc[0]
        with st.expander(db_fltr["Who / What"] + ": " + db_fltr["Use case"]):
            st.write(db_fltr["Description"])

else:
    dfhat = db.copy()
    if query != "":
        dfhat = get_similar_docs(query, dfhat, top_n=10)
    if query != "" or len(cols) != 0:
        for p in cols:
            dfhat[p] = pd.to_numeric(dfhat[p], errors="coerce")
            dfhat = dfhat[dfhat[p].isin([1, 2])]
        if query == "":
            dfhat = dfhat.sort_values(by=cols, ascending=False)
        if len(dfhat) == 0:
            st.write("No data found")
        for row in dfhat.iterrows():
            with st.expander(row[1]["Who / What"] + ": " + row[1]["Use case"]):
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


# ---------------------------------------------------------------------------
# Optional: manual rebuild of the local database from the live Google Sheet.
# (Note: on Streamlit Community Cloud the filesystem is ephemeral, so a rebuild
#  lasts for the running session; commit a fresh data.csv for a permanent update.)
# ---------------------------------------------------------------------------
st.sidebar.write("---")
if st.sidebar.button("🔄 Rebuild data from Google Sheet"):
    if not OPENAI_OK:
        st.sidebar.error("Can't rebuild: OpenAI API key not configured in app secrets.")
    else:
        try:
            with st.spinner("Rebuilding embeddings from the Google Sheet… this can take a minute."):
                rebuild_from_sheet()
            load_data.clear()
            st.sidebar.success("Rebuilt from the sheet. Reloading…")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Rebuild failed: {e}")
