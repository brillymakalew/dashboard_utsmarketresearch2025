import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Market Research Exam Dashboard",
    layout="wide"
)

# Mapping question → description & topic (from exam)
QUESTION_INFO = {
    "1": {
        "short": "Definition & role of marketing research",
        "topic": "Fundamentals of marketing research",
        "lo": "LO1"
    },
    "2": {
        "short": "Steps in marketing research process & clear problem definition",
        "topic": "Marketing research process",
        "lo": "LO2"
    },
    "3A": {
        "short": "Formulating Management Decision Problem (MDP) – Kopi Rasa case",
        "topic": "Problem definition & MDP",
        "lo": "LO2"
    },
    "3B": {
        "short": "Information needs via expert interview & secondary data – Kopi Rasa",
        "topic": "Research design & information needs",
        "lo": "LO2/LO3"
    },
    "4A": {
        "short": "Choosing suitable design (Exploratory / Descriptive / Causal) – Aksara Fase I",
        "topic": "Research design selection",
        "lo": "LO2"
    },
    "4B": {
        "short": "Cross-sectional vs longitudinal (panel) – Aksara Fase II",
        "topic": "Descriptive research design",
        "lo": "LO2/LO3"
    },
    "5": {
        "short": "Mobile marketing research vs traditional; advantages & challenges",
        "topic": "Digital / mobile marketing research",
        "lo": "LO1/LO3"
    },
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
ID_COLS_KNOWN = [
    "No", "NO", "no",
    "NIM", "nim",
    "Nama", "NAMA", "nama",
    "Student ID", "Name"
]

SUMMARY_COLS_KNOWN = [
    "Total", "TOTAL", "total",
    "Bonus Quiz", "Bonus Komti", "Total + Bonus",
    "Excess", "FINAL", "Final", "Nilai Akhir"
]


def detect_columns(df: pd.DataFrame):
    """Detect ID, question, and summary columns based on names."""
    cols = df.columns.tolist()

    id_cols = [c for c in cols if c in ID_COLS_KNOWN]
    summary_cols = [c for c in cols if c in SUMMARY_COLS_KNOWN]

    question_cols = [c for c in cols if c not in id_cols + summary_cols]

    # Keep original order
    question_cols = [c for c in cols if c in question_cols]

    return id_cols, question_cols, summary_cols


@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_excel(file)


def basic_stats(series: pd.Series):
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def render_histogram(series: pd.Series, title: str, bins: int = 10):
    # shorter figure
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of students")
    st.pyplot(fig)


def create_student_pdf(
    label: str,
    row: pd.Series,
    question_cols,
    mean_scores: pd.Series,
    diff: pd.Series,
    main_grade_col: str,
    strengths,
    weaknesses,
):
    """
    Build a nicely formatted PDF report for a single student and return bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title = "Market Research Mid Exam – Individual Report"
    elements.append(Paragraph(title, styles["Title"]))
    elements.append(Spacer(1, 12))

    name = str(row.get("Nama", ""))
    nim = str(row.get("NIM", ""))
    elements.append(Paragraph(f"Student: {name or '-'}", styles["Normal"]))
    if nim and nim.lower() != "nan":
        elements.append(Paragraph(f"NIM: {nim}", styles["Normal"]))
    elements.append(
        Paragraph(f"{main_grade_col}: {row[main_grade_col]:.1f}", styles["Normal"])
    )
    elements.append(Spacer(1, 12))

    # Table of question scores
    elements.append(Paragraph("Scores by Question", styles["Heading2"]))
    data = [["Question", "Student", "Class Average", "Difference"]]
    for q in question_cols:
        s = row[q]
        s_val = f"{s:.1f}" if pd.notnull(s) else "-"
        avg_val = f"{mean_scores[q]:.1f}"
        diff_val = f"{diff[q]:+.1f}"
        data.append([q, s_val, avg_val, diff_val])

    table = Table(data, hAlign="LEFT")
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )
    table.setStyle(table_style)
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Interpretation – strengths & weaknesses
    elements.append(Paragraph("Interpretation", styles["Heading2"]))

    if strengths:
        text = "Strengths (≥ 5 points above class average):"
        elements.append(Paragraph(text, styles["Heading3"]))
        for q in strengths:
            topic = QUESTION_INFO.get(q, {}).get("topic", "")
            p_text = f"- Question {q} (topic: {topic})"
            elements.append(Paragraph(p_text, styles["Normal"]))
    else:
        elements.append(
            Paragraph(
                "No strong strengths (≥ 5 points above class average) detected.",
                styles["Normal"],
            )
        )

    elements.append(Spacer(1, 6))

    if weaknesses:
        text = "Improvement areas (≤ −5 points below class average):"
        elements.append(Paragraph(text, styles["Heading3"]))
        for q in weaknesses:
            topic = QUESTION_INFO.get(q, {}).get("topic", "")
            p_text = f"- Question {q} (topic: {topic})"
            elements.append(Paragraph(p_text, styles["Normal"]))
    else:
        elements.append(
            Paragraph(
                "No strong weaknesses (≤ −5 points below class average) detected.",
                styles["Normal"],
            )
        )

    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            "Generated automatically from the Market Research Exam Dashboard.",
            styles["Italic"],
        )
    )

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ---------------------------------------------------------
# SIDEBAR – FILE
# ---------------------------------------------------------
st.sidebar.title("Market Research Exam Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload exam scores (Excel)", type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload `Nilai Market Research.xlsx` to start.")
    st.stop()

df = load_data(uploaded_file)
id_cols, question_cols, summary_cols = detect_columns(df)

if not question_cols:
    st.error("Could not detect question columns. Please check your file.")
    st.stop()

# Pick which summary column to use as “main grade”
final_candidates = [c for c in summary_cols if c.upper().startswith("FINAL")]
if final_candidates:
    main_grade_col = final_candidates[0]
elif "Total" in df.columns:
    main_grade_col = "Total"
else:
    main_grade_col = summary_cols[0] if summary_cols else question_cols[0]

# Pre-compute stats
question_means = df[question_cols].mean().sort_values(ascending=False)
question_stats = df[question_cols].agg(["mean", "median", "std", "min", "max"]).T
question_stats["%>=70"] = (df[question_cols] >= 70).sum() / len(df) * 100

hardest_question = question_stats["mean"].idxmin()
easiest_question = question_stats["mean"].idxmax()

overall_stats = basic_stats(df[main_grade_col])

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------
st.title("Market Research Mid Exam – Analytics Dashboard")

st.caption(
    f"File: **{uploaded_file.name}** &nbsp; | &nbsp; "
    f"Students: **{len(df)}** &nbsp; | &nbsp; "
    f"Questions: **{len(question_cols)}**"
)

tab_overview, tab_question, tab_student = st.tabs(
    ["📊 Overall Analysis", "❓ Per Question", "👤 Per Student"]
)

# ---------------------------------------------------------
# TAB 1 – OVERALL
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Overall Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Number of students", len(df))
    col2.metric(f"Avg {main_grade_col}", f"{overall_stats['mean']:.1f}")
    col3.metric("Median", f"{overall_stats['median']:.1f}")
    col4.metric("Min", f"{overall_stats['min']:.1f}")
    col5.metric("Max", f"{overall_stats['max']:.1f}")

    st.markdown("### Grade Distribution")
    render_histogram(df[main_grade_col], f"Distribution of {main_grade_col}", bins=10)

    st.markdown("### Quick Insights")
    colL, colR = st.columns(2)

    with colL:
        st.write("**Hardest question (lowest average):**")
        info = QUESTION_INFO.get(hardest_question, {})
        st.markdown(
            f"- Question: **{hardest_question}**  \n"
            f"- Mean score: **{question_stats.loc[hardest_question, 'mean']:.1f}**  \n"
            f"- Topic: *{info.get('topic', 'N/A')}*"
        )

    with colR:
        st.write("**Easiest question (highest average):**")
        info_e = QUESTION_INFO.get(easiest_question, {})
        st.markdown(
            f"- Question: **{easiest_question}**  \n"
            f"- Mean score: **{question_stats.loc[easiest_question, 'mean']:.1f}**  \n"
            f"- Topic: *{info_e.get('topic', 'N/A')}*"
        )

    st.markdown("### Question Averages")
    st.bar_chart(question_means)

    st.markdown("### Question Statistics Table")
    stats_display = question_stats.copy()
    stats_display = stats_display.rename(
        columns={
            "mean": "Mean",
            "median": "Median",
            "std": "Std Dev",
            "min": "Min",
            "max": "Max",
            "%>=70": "% ≥ 70"
        }
    )
    st.dataframe(
        stats_display.style.format(
            {
                "Mean": "{:.1f}",
                "Median": "{:.1f}",
                "Std Dev": "{:.1f}",
                "Min": "{:.0f}",
                "Max": "{:.0f}",
                "% ≥ 70": "{:.1f}",
            }
        )
    )

# ---------------------------------------------------------
# TAB 2 – PER QUESTION
# ---------------------------------------------------------
with tab_question:
    st.subheader("Per Question Analysis")

    sort_by = st.selectbox(
        "Sort questions by",
        [
            "Question order",
            "Mean score (lowest → highest)",
            "Mean score (highest → lowest)",
            "Std deviation (highest → lowest)",
            "% ≥ 70 (lowest → highest)",
        ],
    )

    summary = question_stats.copy()
    summary["question"] = summary.index
    summary["%>=70"] = summary["%>=70"]

    if sort_by == "Mean score (lowest → highest)":
        summary = summary.sort_values("mean", ascending=True)
    elif sort_by == "Mean score (highest → lowest)":
        summary = summary.sort_values("mean", ascending=False)
    elif sort_by == "Std deviation (highest → lowest)":
        summary = summary.sort_values("std", ascending=False)
    elif sort_by == "% ≥ 70 (lowest → highest)":
        summary = summary.sort_values("%>=70", ascending=True)
    else:
        # Question order – as in the original dataframe
        summary["order"] = summary["question"].apply(
            lambda x: list(question_cols).index(x)
        )
        summary = summary.sort_values("order").drop(columns=["order"])

    summary_display = summary.rename(
        columns={
            "mean": "Mean",
            "median": "Median",
            "std": "Std Dev",
            "min": "Min",
            "max": "Max",
            "%>=70": "% ≥ 70",
        }
    )

    st.dataframe(
        summary_display[
            ["question", "Mean", "Median", "Std Dev", "Min", "Max", "% ≥ 70"]
        ].style.format(
            {
                "Mean": "{:.1f}",
                "Median": "{:.1f}",
                "Std Dev": "{:.1f}",
                "Min": "{:.0f}",
                "Max": "{:.0f}",
                "% ≥ 70": "{:.1f}",
            }
        )
    )

    st.markdown("---")
    selected_question = st.selectbox("Select a question", question_cols)

    q_series = df[selected_question]
    q_stats = basic_stats(q_series)
    q_info = QUESTION_INFO.get(selected_question, {})

    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("Mean", f"{q_stats['mean']:.1f}")
    colB.metric("Median", f"{q_stats['median']:.1f}")
    colC.metric("Std Dev", f"{q_stats['std']:.1f}")
    colD.metric("Min", f"{q_stats['min']:.0f}")
    colE.metric("Max", f"{q_stats['max']:.0f}")

    if q_info:
        st.markdown(
            f"**Question {selected_question} description**  \n"
            f"- Topic: *{q_info.get('topic')}*  \n"
            f"- Focus: {q_info.get('short')}  \n"
            f"- Learning Outcome: **{q_info.get('lo')}**"
        )

    st.markdown("#### Score Distribution")
    render_histogram(
        q_series, f"Distribution of Question {selected_question}", bins=10
    )

    st.markdown("#### Top Students for this Question")

    # Prepare ID columns for display
    display_ids = [c for c in ["NIM", "Nama", "No"] if c in df.columns]

    top_n = st.slider(
        "How many students to show?", min_value=3, max_value=15, value=5, step=1
    )

    top_scores = df.sort_values(selected_question, ascending=False).head(top_n)

    st.dataframe(
        top_scores[display_ids + [selected_question, main_grade_col]]
        .reset_index(drop=True)
    )

# ---------------------------------------------------------
# TAB 3 – PER STUDENT
# ---------------------------------------------------------
with tab_student:
    st.subheader("Individual Student Analysis")

    # Build label "Nama (NIM)" for selectbox
    if "Nama" in df.columns and "NIM" in df.columns:
        df["_label"] = df["Nama"].astype(str) + " (" + df["NIM"].astype(str) + ")"
    elif "Nama" in df.columns:
        df["_label"] = df["Nama"].astype(str)
    elif "NIM" in df.columns:
        df["_label"] = df["NIM"].astype(str)
    else:
        df["_label"] = df.index.astype(str)

    selected_label = st.selectbox("Select student", df["_label"].tolist())
    row = df[df["_label"] == selected_label].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    if "Nama" in df.columns:
        col1.metric("Name", str(row.get("Nama")))
    if "NIM" in df.columns:
        col2.metric("NIM", str(row.get("NIM")))
    col3.metric(f"{main_grade_col}", f"{row[main_grade_col]:.1f}")
    if "Total" in df.columns and main_grade_col != "Total":
        col4.metric("Total (raw)", f"{row['Total']:.1f}")

    st.markdown("#### Scores by Question (vs Class Average)")

    student_scores = row[question_cols].astype(float)
    mean_scores = df[question_cols].mean()
    diff = student_scores - mean_scores

    # Bar = Student - Class Average (green above, red below)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors_bar = ["green" if d >= 0 else "red" for d in diff]
    ax.bar(question_cols, diff, color=colors_bar)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Student score − Class average")
    ax.set_xlabel("Question")
    ax.set_title("Difference from Class Average (Green = above, Red = below)")
    st.pyplot(fig)

    # Detailed comparison table
    comp_df = pd.DataFrame(
        {
            "Question": question_cols,
            "Student": [student_scores[q] for q in question_cols],
            "Class Average": [mean_scores[q] for q in question_cols],
            "Difference": [diff[q] for q in question_cols],
        }
    ).set_index("Question")

    st.dataframe(
        comp_df.style.format(
            {"Student": "{:.1f}", "Class Average": "{:.1f}", "Difference": "{:+.1f}"}
        )
    )

    # Identify strengths & improvement areas
    strengths = [q for q in question_cols if diff[q] >= 5]
    weaknesses = [q for q in question_cols if diff[q] <= -5]

    st.markdown("#### Interpretation")

    if strengths:
        st.write("**Strengths (≥ 5 points above class average):**")
        for q in strengths:
            info = QUESTION_INFO.get(q, {})
            topic = info.get("topic", "")
            st.markdown(
                f"- Question **{q}** – score **{student_scores[q]:.0f}** "
                f"(class avg {mean_scores[q]:.1f})  "
                + (f"→ *{topic}*" if topic else "")
            )
    else:
        st.write("No strong strengths (≥ 5 points above class average) detected.")

    if weaknesses:
        st.write("**Improvement areas (≤ −5 points below class average):**")
        for q in weaknesses:
            info = QUESTION_INFO.get(q, {})
            topic = info.get("topic", "")
            st.markdown(
                f"- Question **{q}** – score **{student_scores[q]:.0f}** "
                f"(class avg {mean_scores[q]:.1f})  "
                + (f"→ *{topic}*" if topic else "")
            )
        st.info(
            "Use these topics for targeted feedback or remedial activities "
            "on the specific learning outcomes."
        )
    else:
        st.write("No strong weaknesses (≤ −5 points below class average) detected.")

    st.markdown("#### Raw Answer Scores")
    display_cols = (
        [c for c in ["No", "NIM", "Nama"] if c in df.columns]
        + question_cols
        + [main_grade_col]
    )
    st.dataframe(df[df["_label"] == selected_label][display_cols])

    # ---------- PDF DOWNLOAD BUTTON ----------
    pdf_bytes = create_student_pdf(
        label=selected_label,
        row=row,
        question_cols=question_cols,
        mean_scores=mean_scores,
        diff=diff,
        main_grade_col=main_grade_col,
        strengths=strengths,
        weaknesses=weaknesses,
    )

    # Make a safe filename
    base_name = str(row.get("Nama", selected_label)).strip().replace(" ", "_")
    file_name = f"Report_{base_name}.pdf"

    st.download_button(
        label="📄 Download PDF for this student",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("Dashboard generated for Market Research Mid Exam – built in Streamlit.")
