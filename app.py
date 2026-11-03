from pathlib import Path


import numpy as np
import pandas as pd
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent
DATA_FOLDER = PROJECT_ROOT / "data"
PLOTS_FOLDER = PROJECT_ROOT / "plots"
HORIZON = 56
WEEK_LENGTH = 7

MODEL_RESULTS = pd.DataFrame(
    {
        "MASE": [1.064, 0.876, 0.868, 1.091, 0.841, 0.870],
        "Median MASE": [0.968, 0.825, 0.830, 0.852, 0.820, 0.802],
        "sMAPE (%)": [26.135, 21.191, 20.622, 21.401, 20.441, 19.777],
        "ATMs won": [7, 10, 16, 19, 29, 31],
    },
    index=[
        "Seasonal Naive",
        "Holt Winters",
        "SARIMA",
        "Light GBM",
        "Chronos 2",
        "Ensemble",
    ],
)


@st.cache_data
def load_data():
    transformed = pd.read_csv(DATA_FOLDER / "transformed.csv")
    transformed["day"] = pd.to_datetime(transformed["day"], utc=True).dt.tz_localize(None)
    transformed = transformed.set_index("day").sort_index()
    transformed = transformed.apply(pd.to_numeric, errors="coerce")

    cleaned = pd.read_csv(DATA_FOLDER / "train_cleaned.csv")
    cleaned["day"] = pd.to_datetime(cleaned["day"], utc=True).dt.tz_localize(None)
    cleaned = cleaned.set_index("day").sort_index()
    cleaned = cleaned.drop(columns=["Unnamed: 0"], errors="ignore")
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")

    test = pd.read_csv(DATA_FOLDER / "test.csv")
    test["day"] = pd.to_datetime(test["day"], utc=True).dt.tz_localize(None)
    test = test.set_index("day").sort_index()
    test = test.drop(columns=["Unnamed: 0"], errors="ignore")
    test = test.apply(pd.to_numeric, errors="coerce")

    return transformed, cleaned, test


def image_path(name):
    return str(PLOTS_FOLDER / name)


def seasonal_naive_forecast(series, horizon=HORIZON):
    recent_week = series.dropna().iloc[-WEEK_LENGTH:].to_numpy()
    forecast_values = np.tile(
        recent_week, int(np.ceil(horizon / WEEK_LENGTH))
    )[:horizon]
    forecast_index = pd.date_range(
        series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return pd.Series(forecast_values, index=forecast_index, name="Forecast")


st.set_page_config(
    page_title="ATM Demand Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

transformed_data, cleaned_data, test_data = load_data()
atm_columns = [column for column in transformed_data.columns if column.startswith("ATM_")]
missing_count = int(transformed_data[atm_columns].isna().sum().sum())
systemic_days = int(
    (transformed_data[atm_columns].isna().mean(axis=1) > 0.5).sum()
)

st.title("ATM Demand Forecasting")
st.markdown(
    "An end-to-end view of data transformation, data quality, exploratory analysis, and forecasting model evaluation."
)

with st.sidebar:
    st.header("Project navigation")
    st.write("Use the tabs to move through the analysis.")
    st.metric("ATM series", len(atm_columns))
    st.metric("Observed days", len(transformed_data))
    st.metric("Missing values", f"{missing_count:,}")
    st.metric("Forecast horizon", f"{len(test_data)} days")

overview_tab, cleaning_tab, eda_tab, modeling_tab, explorer_tab = st.tabs(
    ["Overview", "Cleaning", "EDA", "Modeling", "ATM Explorer"]
)

with overview_tab:
    st.header("Project overview")
    overview_columns = st.columns(4)
    overview_columns[0].metric("ATM series", len(atm_columns))
    overview_columns[1].metric("Data points", f"{transformed_data.size:,}")
    overview_columns[2].metric("Missing values", f"{missing_count:,}")
    overview_columns[3].metric("Test window", f"{len(test_data)} days")

    st.subheader("Workflow")
    workflow = pd.DataFrame(
        {
            "Stage": [
                "Transformation",
                "Data quality",
                "Imputation",
                "Forecasting",
                "Evaluation",
            ],
            "Description": [
                "Parse the NN5 TSF file into a dated ATM-by-day table.",
                "Measure missing values, gap lengths, and systemic missing days.",
                "Fill training gaps using weekday-based rolling median values.",
                "Generate 56-day forecasts with statistical and machine-learning models.",
                "Compare forecasts using MASE, median MASE, sMAPE, and ATM wins.",
            ],
        }
    )
    st.dataframe(workflow, hide_index=True, use_container_width=True)

    st.subheader("Key conclusions")
    st.write(
        "The dataset contains strong weekly behavior. Eight days contain missing values for more than half of the ATM series, while the remaining gaps are mostly short and isolated."
    )
    st.write(
        "The ensemble provides the lowest average sMAPE, while Chronos 2 provides the lowest average MASE. Both outperform the seasonal-naive baseline."
    )

with cleaning_tab:
    st.header("Data transformation and cleaning")
    st.write(
        "The original TSF file was transformed into a table with one date column and one numeric column for each ATM series."
    )

    cleaning_columns = st.columns(3)
    cleaning_columns[0].metric("Raw missing values", f"{missing_count:,}")
    cleaning_columns[1].metric("Systemic missing days", systemic_days)
    cleaning_columns[2].metric("Filled training cells", f"{missing_count - int(test_data.isna().sum().sum()):,}")

    missing_by_atm = transformed_data[atm_columns].isna().sum().sort_values(
        ascending=False
    )
    st.subheader("Missing values by ATM")
    st.bar_chart(missing_by_atm.head(15), height=350)

    missing_share = transformed_data[atm_columns].isna().mean(axis=1)
    missing_share = missing_share.rename("Missing share")
    st.subheader("Missing share by day")
    st.line_chart(missing_share, height=300)

    st.subheader("Cleaning method comparison")
    cleaning_results = pd.DataFrame(
        {"Mean absolute error": [0.180, 0.198, 0.199, 0.310]},
        index=[
            "Weekday rolling median",
            "Weekday interpolation",
            "Weekday median",
            "Linear interpolation",
        ],
    )
    st.dataframe(cleaning_results, use_container_width=True)
    st.write(
        "Weekday rolling median was selected because it had the lowest validation error while preserving the weekly pattern."
    )

with eda_tab:
    st.header("Exploratory data analysis")
    st.write(
        "The plots below show the distribution and temporal behavior of ATM demand across the cleaned training data."
    )

    eda_choice = st.selectbox(
        "Choose an EDA visualization",
        [
            "ATM time-series distributions",
            "Daily mean and median",
            "ATM-day heatmap",
            "Weekday-week heatmap",
        ],
    )

    eda_images = {
        "ATM time-series distributions": [
            "atm_lines_dist_1.png",
            "atm_lines_dist_2.png",
            "atm_lines_dist_3.png",
            "atm_lines_dist_4.png",
            "atm_lines_dist_5.png",
            "atm_lines_dist_6.png",
        ],
        "Daily mean and median": ["daily_mean_median.png"],
        "ATM-day heatmap": ["heatmap_atm_day.png"],
        "Weekday-week heatmap": ["heatmap_weekday_week.png"],
    }

    for selected_image in eda_images[eda_choice]:
        st.image(image_path(selected_image), use_column_width=True)

    st.subheader("Interactive ATM summary")
    selected_atms = st.multiselect(
        "Select ATM series",
        atm_columns,
        default=atm_columns[:3],
        max_selections=8,
    )
    if selected_atms:
        selected_data = cleaned_data[selected_atms].tail(180)
        st.line_chart(selected_data, height=400)
        summary_data = pd.DataFrame(
            {
                "Mean": cleaned_data[selected_atms].mean(),
                "Median": cleaned_data[selected_atms].median(),
                "Minimum": cleaned_data[selected_atms].min(),
                "Maximum": cleaned_data[selected_atms].max(),
            }
        )
        st.dataframe(summary_data.round(2), use_container_width=True)

with modeling_tab:
    st.header("Forecasting model comparison")
    st.write(
        "The evaluation uses a 56-day holdout period. MASE compares each model with the seasonal-naive baseline, while sMAPE measures relative forecast error."
    )

    best_mase_model = MODEL_RESULTS["MASE"].idxmin()
    best_smape_model = MODEL_RESULTS["sMAPE (%)"].idxmin()
    model_columns = st.columns(3)
    model_columns[0].metric("Best average MASE", best_mase_model)
    model_columns[1].metric("Best average sMAPE", best_smape_model)
    model_columns[2].metric("Best sMAPE", f"{MODEL_RESULTS['sMAPE (%)'].min():.3f}%")

    st.dataframe(MODEL_RESULTS, use_container_width=True)
    st.bar_chart(MODEL_RESULTS[["MASE", "sMAPE (%)"]], height=400)

    model_plot = st.selectbox(
        "View model forecast examples",
        [
            "Ensemble",
            "Chronos 2",
            "SARIMA",
            "Holt Winters",
            "Light GBM",
            "Seasonal Naive",
        ],
    )
    model_images = {
        "Ensemble": ["ensemble_first.png", "ensemble_second.png", "ensemble_third.png"],
        "Chronos 2": ["chronos_first.png", "chronos_second.png"],
        "SARIMA": ["SARIMA_first.png", "SARIMA_second.png"],
        "Holt Winters": ["holt_winters_first.png", "holt_winters_second.png"],
        "Light GBM": ["light_gbm_first.png", "light_gbm_second.png"],
        "Seasonal Naive": ["s_naive_first.png", "s_naive_second.png"],
    }
    for selected_image in model_images[model_plot]:
        st.image(image_path(selected_image), use_column_width=True)

with explorer_tab:
    st.header("Interactive ATM forecast explorer")
    st.write(
        "Select an ATM to inspect its recent cleaned history and a 56-day seasonal-naive forecast. This follows the baseline method used in the modeling notebook: repeat the latest observed week."
    )

    selected_atm = st.selectbox("ATM series", atm_columns)
    history_days = st.slider("History shown", min_value=28, max_value=365, value=120, step=7)
    history = cleaned_data[selected_atm].tail(history_days)
    forecast = seasonal_naive_forecast(cleaned_data[selected_atm])
    chart_data = pd.concat(
        [
            history.rename("Observed"),
            forecast.rename("Seasonal naive forecast"),
        ],
        axis=1,
    )
    st.line_chart(chart_data, height=450)

    explorer_columns = st.columns(3)
    explorer_columns[0].metric("Historical mean", f"{history.mean():.2f}")
    explorer_columns[1].metric("Forecast mean", f"{forecast.mean():.2f}")
    explorer_columns[2].metric("Forecast horizon", f"{len(forecast)} days")

    st.subheader("Forecast values")
    st.dataframe(forecast.to_frame().round(2), use_container_width=True)
