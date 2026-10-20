"""Streamlit dashboard for the ISV compatibility assessment CSV.

Run with ``py -m streamlit run isv_compatibility_dashboard.py``.
The supplied file contains vendor report links and public-access labels, but
no product-level compatibility labels. z/OS compatibility is therefore shown
as unverified rather than inferred.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = Path(__file__).resolve().parent / "data" / "ISV Compatibility assessment.csv"
RELEASES = ["2.4", "2.5", "3.1", "3.2"]
VENDOR_COLUMN = "vendor"
AVAILABILITY_COLUMN = "Information Available without login"
REPORT_COLUMN = "Link to verify compatibility report"
MFA_COLUMN = "If No -Need MFA to Login ?"


@st.cache_data
def load_assessment(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    required = {VENDOR_COLUMN, AVAILABILITY_COLUMN, REPORT_COLUMN}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing))

    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
    df = df[df[VENDOR_COLUMN] != ""].copy()
    df[VENDOR_COLUMN] = df[VENDOR_COLUMN].replace({"Preciesly": "Precisely"})
    df[AVAILABILITY_COLUMN] = df[AVAILABILITY_COLUMN].replace("", "Not provided")
    if MFA_COLUMN in df.columns:
        df[MFA_COLUMN] = df[MFA_COLUMN].replace("", "Not provided")
    else:
        df[MFA_COLUMN] = "Not provided"
    return df.reset_index(drop=True)


def availability_evaluation(df: pd.DataFrame):
    labels = df[AVAILABILITY_COLUMN].str.casefold()
    valid = labels.isin(["yes", "no"])
    model_df = df.loc[valid, [VENDOR_COLUMN]].copy()
    y = labels.loc[valid].map({"yes": "Yes", "no": "No"})
    if y.nunique() < 2:
        return None, (
            "A model cannot be trained and tested: the CSV needs both Yes and No "
            "availability examples. Blank labels are treated as missing, not No."
        )

    counts = y.value_counts()
    test_count = math.ceil(len(y) * 0.3)
    train_count = len(y) - test_count
    if (
        counts.min() < 2
        or test_count < len(counts)
        or train_count < len(counts)
    ):
        return None, (
            "There are both Yes and No labels, but not enough examples of each "
            "class to create a reliable stratified train/test split. Add at least "
            "two labeled examples for each class, with enough rows for both splits."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        model_df,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    model = Pipeline(
        steps=[
            (
                "preprocessing",
                ColumnTransformer(
                    [
                        (
                            "vendor",
                            OneHotEncoder(handle_unknown="ignore"),
                            [VENDOR_COLUMN],
                        )
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=["Yes", "No"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, predictions),
        "report": report,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }, None


st.set_page_config(
    page_title="z/OS ISV Compatibility Dashboard",
    page_icon="🖥️",
    layout="wide",
)

st.title("z/OS ISV Compatibility Dashboard")
st.caption(
    "Vendor report access and compatibility evidence from the supplied "
    "assessment CSV. Access to a report does not establish product compatibility."
)

if not DATA_FILE.is_file():
    st.error(f"Assessment CSV not found: {DATA_FILE}")
    st.stop()

try:
    assessment = load_assessment(str(DATA_FILE))
except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
    st.error(f"Could not load the assessment CSV: {exc}")
    st.stop()

vendor_count = len(assessment)
available_count = int(
    assessment[AVAILABILITY_COLUMN].str.casefold().eq("yes").sum()
)
report_count = int(assessment[REPORT_COLUMN].str.startswith("http").sum())

metric_columns = st.columns(4)
metric_columns[0].metric("Vendors listed", vendor_count)
metric_columns[1].metric("Reports marked public", available_count)
metric_columns[2].metric("Vendor report links", report_count)
metric_columns[3].metric("z/OS releases shown", len(RELEASES))

st.warning(
    "The CSV has no product names, product versions, or compatibility labels. "
    "Compatibility for every release below is therefore unverified; no "
    "compatibility prediction is generated."
)

overview_tab, vendor_tab, model_tab = st.tabs(
    ["Release overview", "Vendor reports", "Availability model"]
)

with overview_tab:
    st.subheader("Compatibility evidence by z/OS release")
    release_status = pd.DataFrame(
        {
            "z/OS release": RELEASES,
            "Status": ["Unverified — no compatibility data in input"] * len(RELEASES),
            "Evidence": ["No product-level records"] * len(RELEASES),
        }
    )
    st.dataframe(release_status, hide_index=True, width="stretch")
    st.info(
        "The selected releases are 2.4, 2.5, 3.1, and 3.2. The input file "
        "contains vendor-level links only, so it cannot establish historical "
        "or current product support for any of them."
    )

with vendor_tab:
    st.subheader("Vendor report access")
    display_columns = [
        VENDOR_COLUMN,
        "Product",
        "Messsage ID to identify Product version",
        AVAILABILITY_COLUMN,
        MFA_COLUMN,
        REPORT_COLUMN,
    ]
    display_columns = [column for column in display_columns if column in assessment]
    st.dataframe(
        assessment[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            REPORT_COLUMN: st.column_config.LinkColumn(
                "Compatibility report", display_text="Open report"
            )
        },
    )
    selected_vendor = st.selectbox(
        "Open a vendor report",
        assessment[VENDOR_COLUMN].tolist(),
    )
    vendor_row = assessment.loc[
        assessment[VENDOR_COLUMN] == selected_vendor
    ].iloc[0]
    report_url = vendor_row[REPORT_COLUMN]
    if report_url.startswith("https://") or report_url.startswith("http://"):
        st.link_button("Open vendor report", report_url)
    else:
        st.info("No report URL was supplied for this vendor.")

with model_tab:
    st.subheader("Information-availability model")
    st.caption(
        "This model would predict only whether a vendor report is marked "
        "available without login. It is not a compatibility model."
    )
    labels = assessment[AVAILABILITY_COLUMN].str.casefold()
    valid_labels = labels[labels.isin(["yes", "no"])].map(
        {"yes": "Yes", "no": "No"}
    )
    label_counts = valid_labels.value_counts().reindex(["Yes", "No"], fill_value=0)
    st.dataframe(
        label_counts.rename_axis("Availability label").rename("Rows").reset_index(),
        hide_index=True,
        width="content",
    )
    evaluation, reason = availability_evaluation(assessment)
    if reason:
        st.warning(reason)
        st.write(
            f"Usable labels: {len(valid_labels)}; "
            f"Yes: {int(label_counts['Yes'])}; No: {int(label_counts['No'])}; "
            f"missing/other: {len(assessment) - len(valid_labels)}."
        )
    else:
        st.metric("Held-out accuracy", f"{evaluation['accuracy']:.1%}")
        st.caption(
            f"Train rows: {evaluation['train_rows']} · "
            f"Test rows: {evaluation['test_rows']}. "
            "This small-data score is not evidence of compatibility."
        )
        report_rows = []
        for label in ("Yes", "No"):
            scores = evaluation["report"][label]
            report_rows.append(
                {
                    "Label": label,
                    "Precision": scores["precision"],
                    "Recall": scores["recall"],
                    "F1": scores["f1-score"],
                    "Support": int(scores["support"]),
                }
            )
        st.dataframe(
            pd.DataFrame(report_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.2f"),
                "Recall": st.column_config.NumberColumn(format="%.2f"),
                "F1": st.column_config.NumberColumn(format="%.2f"),
            },
        )
