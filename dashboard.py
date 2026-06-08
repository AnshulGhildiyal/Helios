import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "helios_eval.db"

st.set_page_config(page_title="Helios Dashboard", layout="wide", page_icon="👓")

st.title("👓 Helios: LLM Evaluation Dashboard")
st.markdown("Automated RAG evaluation metrics and adversarial framework analytics.")
st.divider()

def load_evaluation_data():
    """Extracts combined run metrics and test case data from SQLite."""
    if not DB_PATH.exists():
        return pd.DataFrame()
        
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            r.run_id AS "Run ID", 
            r.timestamp AS "Timestamp", 
            t.difficulty AS "Difficulty", 
            t.question AS "Question",
            t.ground_truth AS "Ground Truth",
            m.system_answer AS "Evaluated Answer",
            m.score AS "Faithfulness Score", 
            m.hallucination_flag AS "Hallucinated", 
            m.reasoning AS "Judge Reasoning"
        FROM evaluation_runs r
        JOIN test_cases t ON r.run_id = t.run_id
        JOIN metrics_results m ON t.case_id = m.case_id
        ORDER BY r.timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

try:
    results_df = load_evaluation_data()
    
    if not results_df.empty:
        st.header("Evaluation Telemetry")
        col1, col2, col3 = st.columns(3)
        
        total_evals = len(results_df)
        avg_score = results_df["Faithfulness Score"].mean()

        hallucination_count = results_df["Hallucinated"].sum()
        hallucination_rate = (hallucination_count / total_evals) * 100
        
        col1.metric("Total Test Cases Evaluated", total_evals)
        col2.metric("Average Faithfulness Score", f"{avg_score:.2f} / 5.0")
        col3.metric("System Hallucination Rate", f"{hallucination_rate:.1f}%")
        
        st.divider()

        st.header("Granular Performance Breakdown")

        selected_run = st.selectbox("Filter by Run ID", ["All Runs"] + list(results_df["Run ID"].unique()))
        
        display_df = results_df if selected_run == "All Runs" else results_df[results_df["Run ID"] == selected_run]
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.info("The database file was located, but no complete evaluation records were found. Run `python main.py` to populate data.")
        
except Exception as e:
    st.error(f"Critical System Error reading database: {e}")