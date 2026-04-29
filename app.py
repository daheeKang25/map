import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from shapely.geometry import shape, MultiLineString, LineString
from shapely.ops import unary_union


# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="서울시 노인 일자리 인프라 사각지대 시각화",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.2rem;
    }
    .block-container {
        max-width: 1500px;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        color: #4b5563;
        margin-bottom: 1.1rem;
        line-height: 1.7;
    }
    .caption-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0.7rem;
        line-height: 1.65;
    }
    .viz-card {
        border: 1px solid #e5e7eb;
        background: white;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
    }
    .viz-card-title {
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .metric-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    .summary-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        line-height: 1.7;
    }
    .footer-note {
        color: #6b7280;
        font-size: 0.88rem;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 경로 설정
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "institutions": DATA_DIR / "노인일자리수행기관_지오코딩결과.csv",
    "basic_recipients": DATA_DIR / "2024_서울시_국민기초생활수급자_일반+생계+의료+구별_65세이상.csv",
    "elderly_population": DATA_DIR / "고령자현황_내국인_구별_2024.csv",
    "single_basic": DATA_DIR / "독거노인_기초수급.csv",
    "single_low_income": DATA_DIR / "독거노인_저소득.csv",
    "single_total": DATA_DIR / "독거노인_총.csv",
    "geojson": DATA_DIR / "seoul_municipalities_geo_simple.json",
}

DISTRICT_ORDER = [
    "종로구", "중구", "용산구", "성동구", "광진구",
    "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구",
    "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구",
]

DEMAND_METRICS = {
    "고령자 경제취약 비중 (%)": "economic_vulnerability_rate",
    "경제취약 독거노인 비중 (%)": "single_recipient_rate",
    "저소득 독거노인 비중 (%)": "single_low_income_rate",
}

SUPPLY_METRICS = {
    "기관 수": "institution_count",
    "고령자 1만 명당 기관 수": "institutions_per_10000_elderly",
}

COLOR_DEMAND = "#c2410c"
COLOR_SUPPLY = "#1d4ed8"
COLOR_BLINDSPOT = "#7c3aed"


# -----------------------------
# 유틸 함수
# -----------------------------
def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def minmax_100(series: pd.Series) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return (series - s_min) / (s_max - s_min) * 100


def mean_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("자치구", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    )
    return agg


def find_geojson_feature_key(geojson_data: dict, district_names: list[str]) -> str | None:
    if "features" not in geojson_data or not geojson_data["features"]:
        return None

    candidate_scores = {}
    district_set = set(district_names)

    for key in geojson_data["features"][0]["properties"].keys():
        values = {
            str(feature["properties"].get(key, "")).strip()
            for feature in geojson_data["features"]
        }
        score = len(district_set.intersection(values))
        candidate_scores[key] = score

    best_key, best_score = max(candidate_scores.items(), key=lambda x: x[1])
    if best_score == 0:
        return None
    return best_key


@st.cache_data
def load_all_data():
    required_files = [
        FILES["institutions"],
        FILES["basic_recipients"],
        FILES["elderly_population"],
        FILES["single_basic"],
        FILES["single_low_income"],
        FILES["single_total"],
    ]

    missing_files = [str(path.name) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "다음 파일이 data 폴더에 없습니다: " + ", ".join(missing_files)
        )

    institutions = read_csv_auto(FILES["institutions"])
    basic = read_csv_auto(FILES["basic_recipients"])
    elderly = read_csv_auto(FILES["elderly_population"])
    single_basic = read_csv_auto(FILES["single_basic"])
    single_low = read_csv_auto(FILES["single_low_income"])
    single_total = read_csv_auto(FILES["single_total"])

    institutions["latitude"] = to_numeric(institutions["latitude"])
    institutions["longitude"] = to_numeric(institutions["longitude"])
    institutions = institutions.dropna(subset=["latitude", "longitude"]).copy()

    basic = basic.rename(columns={"총 수급자수": "basic_recipient_65plus"})
    basic["basic_recipient_65plus"] = to_numeric(basic["basic_recipient_65plus"])

    elderly = elderly.rename(columns={"고령자(내국인)수": "elderly_population"})
    elderly["elderly_population"] = to_numeric(elderly["elderly_population"])

    single_basic = single_basic.rename(columns={"시군구": "자치구", "전체수": "single_basic_count"})
    single_basic["single_basic_count"] = to_numeric(single_basic["single_basic_count"])
    single_basic = single_basic[["자치구", "single_basic_count"]]

    single_low = single_low.rename(columns={"시군구": "자치구", "전체수": "single_low_income_count"})
    single_low["single_low_income_count"] = to_numeric(single_low["single_low_income_count"])
    single_low = single_low[["자치구", "single_low_income_count"]]

    single_total = single_total.rename(columns={"시군구": "자치구", "전체수": "single_total_count"})
    single_total["single_total_count"] = to_numeric(single_total["single_total_count"])
    single_total = single_total[["자치구", "single_total_count"]]

    supply_count = (
        institutions.groupby("자치구", as_index=False)
        .agg(institution_count=("기관명", "count"))
    )
    supply_centers = mean_lat_lon(institutions)

    merged = (
        elderly[["자치구", "elderly_population"]]
        .merge(basic[["자치구", "basic_recipient_65plus"]], on="자치구", how="left")
        .merge(single_basic, on="자치구", how="left")
        .merge(single_low, on="자치구", how="left")
        .merge(single_total, on="자치구", how="left")
        .merge(supply_count, on="자치구", how="left")
        .merge(supply_centers, on="자치구", how="left")
    )

    merged = merged.fillna(
        {
            "basic_recipient_65plus": 0,
            "single_basic_count": 0,
            "single_low_income_count": 0,
            "single_total_count": 0,
            "institution_count": 0,
        }
    )

    merged["economic_vulnerability_rate"] = (
        merged["basic_recipient_65plus"] / merged["elderly_population"] * 100
    )
    merged["single_recipient_rate"] = (
        merged["single_basic_count"] / merged["elderly_population"] * 100
    )
    merged["single_low_income_rate"] = (
        merged["single_low_income_count"] / merged["elderly_population"] * 100
    )
    merged["single_total_rate"] = (
        merged["single_total_count"] / merged["elderly_population"] * 100
    )
    merged["institutions_per_10000_elderly"] = (
        merged["institution_count"] / merged["elderly_population"] * 10000
    )
    merged["vulnerable_elderly_per_institution"] = np.where(
        merged["institution_count"] > 0,
        merged["basic_recipient_65plus"] / merged["institution_count"],
        np.nan,
    )

    merged["district_order"] = pd.Categorical(
        merged["자치구"], categories=DISTRICT_ORDER, ordered=True
    )
    merged = merged.sort_values("district_order").drop(columns="district_order")

    geojson_data = None
    geojson_feature_key = None
    if FILES["geojson"].exists():
        with open(FILES["geojson"], "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        feature_key = find_geojson_feature_key(geojson_data, merged["자치구"].tolist())
        if feature_key:
            geojson_feature_key = f"properties.{feature_key}"

    return merged, institutions, geojson_data, geojson_feature_key


def build_dynamic_metrics(df: pd.DataFrame, demand_col: str, supply_col: str):
    chart_df = df.copy()

    chart_df["demand_value"] = chart_df[demand_col]
    chart_df["supply_value"] = chart_df[supply_col]
    chart_df["demand_z"] = zscore(chart_df["demand_value"])
    chart_df["supply_z"] = zscore(chart_df["supply_value"])
    chart_df["blind_spot_index"] = chart_df["demand_z"] - chart_df["supply_z"]
    chart_df["demand_scaled"] = minmax_100(chart_df["demand_value"])
    chart_df["supply_scaled"] = minmax_100(chart_df["supply_value"])

    x_mean = chart_df["demand_value"].mean()
    y_mean = chart_df["supply_value"].mean()

    def assign_quadrant(row):
        if row["demand_value"] >= x_mean and row["supply_value"] < y_mean:
            return "고수요·저공급"
        elif row["demand_value"] >= x_mean and row["supply_value"] >= y_mean:
            return "고수요·고공급"
        elif row["demand_value"] < x_mean and row["supply_value"] < y_mean:
            return "저수요·저공급"
        return "저수요·고공급"

    chart_df["quadrant"] = chart_df.apply(assign_quadrant, axis=1)
    return chart_df, x_mean, y_mean

def add_outer_boundary_trace(fig, geojson_data: dict, color="#111827", width=4):
    geometries = [shape(feature["geometry"]) for feature in geojson_data["features"]]
    merged = unary_union(geometries)
    boundary = merged.boundary

    lines = []
    if isinstance(boundary, LineString):
        lines = [boundary]
    elif isinstance(boundary, MultiLineString):
        lines = list(boundary.geoms)

    for line in lines:
        coords = list(line.coords)
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]

        fig.add_trace(
            go.Scattermap(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

# -----------------------------
# 지도 함수
# -----------------------------
def make_choropleth(df: pd.DataFrame, geojson_data: dict, feature_key: str, value_col: str, title: str):
    fig = px.choropleth_map(
        df,
        geojson=geojson_data,
        locations="자치구",
        featureidkey=feature_key,
        color=value_col,
        color_continuous_scale="Oranges",
        map_style="carto-positron",
        zoom=9.5,
        center={"lat": 37.5665, "lon": 126.9780},
        opacity=0.65,
        hover_data={
            "자치구": True,
            value_col: ":.2f",
            "basic_recipient_65plus": True,
            "elderly_population": True,
            "institution_count": True,
        },
    )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=45, b=0),
        coloraxis_colorbar=dict(title="취약 수요"),
    )
    return fig


def make_bubble_map(df: pd.DataFrame, geojson_data: dict, feature_key: str, title: str):
    plot_df = df.dropna(subset=["latitude", "longitude"]).copy()

    # 자치구 경계선용 투명 지도
    boundary_df = df.copy()
    boundary_df["_boundary"] = 1

    boundary_fig = px.choropleth_map(
        boundary_df,
        geojson=geojson_data,
        locations="자치구",
        featureidkey=feature_key,
        color="_boundary",
        color_continuous_scale=[
            [0.0, "rgba(0,0,0,0)"],
            [1.0, "rgba(0,0,0,0)"],
        ],
        map_style="carto-positron",
        zoom=9.5,
        center={"lat": 37.5665, "lon": 126.9780},
        opacity=0.0,
        hover_data={"자치구": True},
    )

    # 자치구 내부 경계선
    boundary_fig.update_traces(
        marker_line_color="#000000",
        marker_line_width=1.2,
        hovertemplate=None,
        hoverinfo="skip",
    )
    boundary_fig.update_layout(coloraxis_showscale=False)

    # 서울시 외곽선 추가
    add_outer_boundary_trace(boundary_fig, geojson_data, color="#111827", width=4)

    # 버블 레이어
    bubble_fig = px.scatter_map(
        plot_df,
        lat="latitude",
        lon="longitude",
        size="institution_count",
        color="institution_count",
        size_max=42,
        zoom=9.5,
        center={"lat": 37.5665, "lon": 126.9780},
        map_style="carto-positron",
        color_continuous_scale="Blues",
        hover_name="자치구",
        hover_data={
            "institution_count": True,
            "institutions_per_10000_elderly": ":.2f",
            "latitude": False,
            "longitude": False,
        },
    )

    for tr in bubble_fig.data:
        if hasattr(tr, "marker") and tr.marker is not None:
            tr.marker.coloraxis = "coloraxis2"
        boundary_fig.add_trace(tr)

    boundary_fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=45, b=0),
        coloraxis2=dict(
            colorscale="Blues",
            cmin=plot_df["institution_count"].min(),
            cmax=plot_df["institution_count"].max(),
            colorbar=dict(title="기관 수", x=1.02),
        ),
    )

    return boundary_fig


def make_integrated_map(df: pd.DataFrame, geojson_data: dict, feature_key: str, demand_col: str, title: str):
    # 취약 수요 지도 재사용
    base_fig = make_choropleth(
        df=df,
        geojson_data=geojson_data,
        feature_key=feature_key,
        value_col=demand_col,
        title=title,
    )

    # 버블맵 재사용
    bubble_overlay_fig = make_bubble_map(
        df=df,
        geojson_data=geojson_data,
        feature_key=feature_key,
        title="",
    )

    # bubble_overlay_fig의 첫 trace는 경계선용 투명 지도이므로 제외하고 버블만 추가
    for tr in bubble_overlay_fig.data[1:]:
        if hasattr(tr, "marker") and tr.marker is not None:
            tr.marker.coloraxis = "coloraxis2"
        base_fig.add_trace(tr)

    base_fig.update_layout(
        margin=dict(l=0, r=0, t=45, b=0),
        coloraxis=dict(
            colorscale="Oranges",
            colorbar=dict(title="취약 수요", x=1.00),
        ),
        coloraxis2=dict(
            colorscale="Blues",
            cmin=df["institution_count"].min(),
            cmax=df["institution_count"].max(),
            colorbar=dict(title="기관 수", x=1.08),
        ),
    )

    return base_fig


# -----------------------------
# 일반 차트 함수
# -----------------------------
def make_grouped_comparison_bar(df: pd.DataFrame, demand_label: str, supply_label: str):
    long_df = pd.DataFrame({
        "자치구": np.repeat(df["자치구"].values, 2),
        "구분": [demand_label, supply_label] * len(df),
        "값": np.ravel(np.column_stack([df["demand_scaled"], df["supply_scaled"]])),
    })
    fig = px.bar(
        long_df,
        x="자치구",
        y="값",
        color="구분",
        barmode="group",
        color_discrete_map={demand_label: COLOR_DEMAND, supply_label: COLOR_SUPPLY},
        title="수요 vs 공급 이중 비교 바 차트 (0~100 정규화)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_title="자치구",
        yaxis_title="정규화 값 (0~100)",
        legend_title="지표",
        xaxis_tickangle=-45,
    )
    return fig


def make_district_detail_bar(df: pd.DataFrame, selected_district: str, demand_label: str):
    row = df[df["자치구"] == selected_district].iloc[0]
    avg_demand = df["demand_value"].mean()
    avg_supply = df["supply_value"].mean()

    detail_df = pd.DataFrame({
        "항목": [demand_label, "고령자 1만 명당 기관 수"],
        "서울시 평균": [avg_demand, avg_supply],
        selected_district: [row["demand_value"], row["supply_value"]],
    })
    detail_long = detail_df.melt(id_vars="항목", var_name="비교대상", value_name="값")

    fig = px.bar(
        detail_long,
        x="항목",
        y="값",
        color="비교대상",
        barmode="group",
        title=f"지역 상세 바 차트: 서울시 평균 vs {selected_district}",
        color_discrete_map={"서울시 평균": "#9ca3af", selected_district: COLOR_BLINDSPOT},
        text="값",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_title="",
        yaxis_title="값",
        legend_title="",
    )
    return fig


def make_quadrant_scatter(df: pd.DataFrame, demand_label: str, supply_label: str, x_mean: float, y_mean: float):
    color_map = {
        "고수요·저공급": "#dc2626",
        "고수요·고공급": "#ea580c",
        "저수요·저공급": "#2563eb",
        "저수요·고공급": "#16a34a",
    }
    fig = px.scatter(
        df,
        x="demand_value",
        y="supply_value",
        color="quadrant",
        color_discrete_map=color_map,
        hover_name="자치구",
        hover_data={
            "blind_spot_index": ":.2f",
            "institution_count": True,
            "demand_value": ":.2f",
            "supply_value": ":.2f",
        },
        title="사분면 산점도",
        text="자치구",
    )
    fig.add_vline(x=x_mean, line_dash="dash", line_color="gray")
    fig.add_hline(y=y_mean, line_dash="dash", line_color="gray")
    fig.update_traces(
        textposition="top center",
        marker=dict(size=12, line=dict(width=1, color="white"))
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_title=demand_label,
        yaxis_title=supply_label,
        legend_title="사분면",
    )
    return fig


def make_blindspot_bar(df: pd.DataFrame):
    sorted_df = df.sort_values("blind_spot_index", ascending=False)
    fig = px.bar(
        sorted_df,
        x="blind_spot_index",
        y="자치구",
        orientation="h",
        color="blind_spot_index",
        color_continuous_scale="Purples",
        title="사각지대 지수 바 차트",
        hover_data={"demand_value": ":.2f", "supply_value": ":.2f"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_title="사각지대 지수 (수요 Z - 공급 Z)",
        yaxis_title="",
        yaxis=dict(categoryorder="array", categoryarray=sorted_df["자치구"].tolist()[::-1]),
        coloraxis_showscale=False,
    )
    return fig


# -----------------------------
# UI 보조 함수
# -----------------------------
def draw_viz_cards():
    cards = [
        ("자치구별 취약 수요 지도", "서울시 자치구별 고령자 경제취약 수요의 공간적 분포를 색으로 보여줍니다."),
        ("수행기관 버블맵", "자치구 경계선 위에 자치구별 노인일자리 수행기관 규모를 버블 색과 크기로 보여줍니다."),
        ("수요-공급 불균형 통합 지도", "취약 수요 지도와 수행기관 버블맵을 겹쳐 지역별 불균형을 한 화면에서 보여줍니다."),
        ("수요 vs 공급 이중 비교 바 차트", "자치구별 취약 수요와 공급 수준을 나란히 비교합니다."),
        ("지역 상세 바 차트", "서울시 평균과 선택 지역을 직접 비교하여 특정 지역의 특징을 확인합니다."),
        ("사분면 산점도", "자치구를 고수요·저공급 등 유형으로 분류해 사각지대 후보를 드러냅니다."),
        ("사각지대 지수 바 차트", "수요 대비 공급이 상대적으로 부족한 자치구를 순위 형태로 제시합니다."),
    ]
    for title, desc in cards:
        st.markdown(
            f"""
            <div class="viz-card">
                <div class="viz-card-title">{title}</div>
                <div class="metric-note">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_geojson_warning():
    st.warning(
        "지도형 시각화를 보려면 `data/seoul_municipalities_geo_simple.json` 파일이 필요합니다."
    )
    st.code(
        "프로젝트 폴더\n"
        "├─ app.py\n"
        "└─ data/\n"
        "   ├─ 노인일자리수행기관_지오코딩결과.csv\n"
        "   ├─ 2024_서울시_국민기초생활수급자_일반+생계+의료+구별_65세이상.csv\n"
        "   ├─ 고령자현황_내국인_구별_2024.csv\n"
        "   ├─ 독거노인_기초수급.csv\n"
        "   ├─ 독거노인_저소득.csv\n"
        "   ├─ 독거노인_총.csv\n"
        "   └─ seoul_municipalities_geo_simple.json",
        language="text",
    )


# -----------------------------
# 데이터 로드
# -----------------------------
try:
    df_base, institutions_raw, geojson_data, geojson_feature_key = load_all_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.header("분석 설정")

selected_demand_label = st.sidebar.selectbox(
    "취약 수요 지표 선택",
    list(DEMAND_METRICS.keys()),
    index=0,
)

selected_supply_label = st.sidebar.selectbox(
    "공급 지표 선택",
    list(SUPPLY_METRICS.keys()),
    index=1,
)

selected_district = st.sidebar.selectbox(
    "상세 비교 지역 선택",
    df_base["자치구"].tolist(),
    index=df_base["자치구"].tolist().index("중랑구") if "중랑구" in df_base["자치구"].tolist() else 0,
)

sort_option = st.sidebar.selectbox(
    "자치구 정렬 기준",
    ["사각지대 지수 순", "취약 수요 순", "공급 지표 순", "기본 행정구 순"],
    index=0,
)

st.sidebar.markdown("---")

demand_col = DEMAND_METRICS[selected_demand_label]
supply_col = SUPPLY_METRICS[selected_supply_label]

df_chart, x_mean, y_mean = build_dynamic_metrics(df_base, demand_col, supply_col)

if sort_option == "사각지대 지수 순":
    df_sorted = df_chart.sort_values("blind_spot_index", ascending=False)
elif sort_option == "취약 수요 순":
    df_sorted = df_chart.sort_values("demand_value", ascending=False)
elif sort_option == "공급 지표 순":
    df_sorted = df_chart.sort_values("supply_value", ascending=False)
else:
    df_sorted = df_chart.copy()


# -----------------------------
# 상단 헤더
# -----------------------------
st.markdown(
    '<div class="app-title">노인 빈곤 문제 해결을 위한 서울시 노인 일자리 인프라 사각지대 시각화</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
        2026 서울시 빅데이터 활용 경진대회 -시각화 부문 출품작
    </div>
    """,
    unsafe_allow_html=True,
)

summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.metric("분석 대상 자치구", f"{df_chart['자치구'].nunique()}개")
with summary_col2:
    st.metric("전체 수행기관 수", f"{int(df_chart['institution_count'].sum())}개")


# -----------------------------
# 탭 구성
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["작품 소개", "수요와 공급의 공간적 분포", "자치구별 비교 분석", "사각지대 도출"]
)


# -----------------------------
# 탭 1. 작품 소개
# -----------------------------
with tab1:
    st.subheader("작품 소개")
    st.markdown(
        """
        이 작품은 서울시 자치구별 노인 경제 취약 수요와 노인 일자리 지원 인프라를 함께 시각화하여,
        지역별 수요-공급 불균형과 정책적 사각지대를 파악하는 것을 목표로 합니다.
        """
    )

    st.markdown("### 무엇을 볼 수 있는가")
    draw_viz_cards()

    st.markdown("### 팀 정보")
    st.info("팀명: on-jeon(온전)")

    st.markdown("### 데이터 출처")
    with st.expander("서울 데이터허브 데이터 출처 보기", expanded=True):
        st.markdown(
            """
            **모든 공공 데이터는 2024년 기준으로 시각화했습니다.**

            - 서울시 사회복지시설(노인일자리지원기관) 목록  
              https://data.seoul.go.kr/dataList/OA-20414/S/1/datasetView.do
            - 서울시 국민기초생활보장 연령별 일반수급자(구별) 통계  
              https://data.seoul.go.kr/dataList/10769/S/2/datasetView.do
            - 서울시 고령자현황 통계  
              https://data.seoul.go.kr/dataList/10020/S/2/datasetView.do
            - 서울시 국민기초생활 수급자 동별 현황  
              https://data.seoul.go.kr/dataList/OA-22227/F/1/datasetView.do
            - 서울시 독거노인 현황 (성별/구별) 통계  
              https://data.seoul.go.kr/bsp/wgs/dataView/data300View/10075.do
            """
        )

    with st.expander("기타 자료 출처 보기", expanded=True):
        st.markdown(
            """
            노인일자리 및 사회활동 지원사업 수행기관 정보는 아래 웹사이트에서 직접 수집했습니다.  
            **2026년 4월 현재 게재된 정보를 기준으로 시각화했습니다.**

            - 수행기관 정보 수집 출처  
              https://www.seniorro.or.kr/noin/organization.do
            """
        )


# -----------------------------
# 탭 2. 수요와 공급의 공간적 분포
# -----------------------------
with tab2:
    st.subheader("수요와 공급의 공간적 분포")
    st.markdown(
        """
        서울시 전체 공간 구조를 먼저 직관적으로 이해하기 위한 탐색용 화면입니다.
        취약 수요와 수행기관 분포를 따로 본 뒤, 통합 지도로 겹쳐 보면서 지역별 불균형을 확인할 수 있습니다.
        """
    )

    if not geojson_data or not geojson_feature_key:
        show_geojson_warning()
    else:
        col_map1, col_map2 = st.columns(2)

        with col_map1:
            choropleth_fig = make_choropleth(
                df_chart,
                geojson_data,
                geojson_feature_key,
                demand_col,
                f"자치구별 취약 수요 지도: {selected_demand_label}",
            )
            st.plotly_chart(choropleth_fig, width="stretch")
            st.markdown(
                """
                <div class="caption-box">
                서울 내 취약 수요는 균등하지 않습니다. 자치구별 색의 차이를 통해 취약 수요가 집중된 지역을 확인할 수 있습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_map2:
            bubble_fig = make_bubble_map(
                df_chart,
                geojson_data,
                geojson_feature_key,
                "수행기관 버블맵"
            )
            st.plotly_chart(bubble_fig, width="stretch")
            st.markdown(
                """
                <div class="caption-box">
                수행기관 분포도 균등하지 않습니다. 버블의 색과 크기는 자치구별 수행기관 수를 함께 보여줍니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        integrated_fig = make_integrated_map(
            df_chart,
            geojson_data,
            geojson_feature_key,
            demand_col,
            "수요-공급 불균형 통합 지도",
        )
        st.plotly_chart(integrated_fig, width="stretch")

        st.markdown(
            """
            <div class="summary-box">
            취약 수요와 수행기관 분포를 함께 보면 지역별 불균형이 더 분명하게 드러납니다.
            이 통합 지도는 각 자치구의 취약 수요 수준과 기관 기반이 어떻게 겹치는지를 한 화면에서 보여줍니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_bubble_district = st.selectbox(
            "기관 목록을 확인할 자치구 선택",
            df_chart["자치구"].tolist(),
            index=df_chart["자치구"].tolist().index(selected_district),
            key="bubble_fallback_select",
        )

        st.markdown(f"### 선택 자치구 기관 목록: {selected_bubble_district}")
        selected_institutions = (
            institutions_raw[institutions_raw["자치구"] == selected_bubble_district]
            [["기관명", "주소정제"]]
            .rename(columns={"주소정제": "상세주소"})
            .sort_values("기관명")
            .reset_index(drop=True)
        )
        st.dataframe(selected_institutions, width="stretch", hide_index=True)


# -----------------------------
# 탭 3. 자치구별 비교 분석
# -----------------------------
with tab3:
    st.subheader("자치구별 비교 분석")
    st.markdown(
        """
        공간적 분포를 수치로 확인하는 정량 비교 화면입니다.
        전체 자치구를 한 번에 비교한 뒤, 특정 자치구를 서울시 평균과 직접 비교할 수 있습니다.
        """
    )

    grouped_fig = make_grouped_comparison_bar(df_sorted, selected_demand_label, selected_supply_label)
    st.plotly_chart(grouped_fig, width="stretch")

    st.markdown(
        """
        <div class="caption-box">
        자치구마다 고령 인구 규모가 다르기 때문에 단순 기관 수만으로는 공급 수준을 판단하기 어렵습니다. 
        <br>따라서 고령 인구 1만명 대비 기관 비율을  공급 수준지표로 설정하였습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    compare_col1, compare_col2 = st.columns([2.2, 1])

    with compare_col1:
        detail_fig = make_district_detail_bar(df_chart, selected_district, selected_demand_label)
        st.plotly_chart(detail_fig, width="stretch")

    with compare_col2:
        row = df_chart[df_chart["자치구"] == selected_district].iloc[0]
        city_avg_demand = df_chart["demand_value"].mean()
        city_avg_supply = df_chart["supply_value"].mean()

        st.markdown("### 선택 지역 요약")
        st.metric(
            label=selected_demand_label,
            value=f"{row['demand_value']:.2f}",
            delta=f"{row['demand_value'] - city_avg_demand:+.2f} (서울 평균 대비)",
        )
        st.metric(
            label=selected_supply_label,
            value=f"{row['supply_value']:.2f}",
            delta=f"{row['supply_value'] - city_avg_supply:+.2f} (서울 평균 대비)",
        )
        st.metric(
            label="기관 1개당 경제취약 고령자 수",
            value="N/A" if pd.isna(row["vulnerable_elderly_per_institution"]) else f"{row['vulnerable_elderly_per_institution']:.1f}",
        )

        st.markdown(
            f"""
            <div class="summary-box">
            <b>{selected_district}</b>와 서울시 평균을 비교했을 때
            취약 수요와 공급 기반이 어느 방향으로 차이를 보이는지 확인할 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# 탭 4. 사각지대 도출
# -----------------------------
with tab4:
    st.subheader("사각지대 도출")
    st.markdown(
        """
        사각지대 분석 결과를 보여주는 화면입니다.
        사분면 산점도는 자치구를 유형으로 나누고, 사각지대 지수 바 차트는 우선순위를 순위 형태로 제시합니다.
        """
    )

    final_col1, final_col2 = st.columns(2)

    with final_col1:
        scatter_fig = make_quadrant_scatter(
            df_chart,
            selected_demand_label,
            selected_supply_label,
            x_mean,
            y_mean,
        )
        st.plotly_chart(scatter_fig, width="stretch")

    with final_col2:
        blind_fig = make_blindspot_bar(df_chart)
        st.plotly_chart(blind_fig, width="stretch")

    top5 = df_chart.sort_values("blind_spot_index", ascending=False).head(5)
    top5_text = " · ".join(top5["자치구"].tolist())

    st.markdown(
        f"""
        <div class="summary-box">
        <b>사각지대 결과 요약</b><br>
        현재 선택한 지표 기준에서 취약 수요가 높고 기관 기반이 낮은 자치구는
        <b>{top5_text}</b> 등으로 나타납니다.<br><br>
        세 지표 모두에서 고수요·저공급으로 나타나는 공통 자치구는 <b>강북구·강서구·금천구·은평구</b>입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note">
        사각지대 지수는 선택한 취약 수요 지표의 Z-점수에서 공급 지표의 Z-점수를 뺀 값입니다.
        값이 클수록 수요 대비 공급이 상대적으로 부족하다는 의미입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
