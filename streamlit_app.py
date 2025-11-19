#######################
# Import libraries
#######################
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# XGBoost는 설치되어 있으면 사용, 없으면 건너뜀
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

#######################
# Page configuration
#######################
st.set_page_config(
    page_title="도로 재비산먼지 분석 대시보드",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# altair 테마 (경고는 뜰 수 있지만 동작에는 문제 없음)
alt.themes.enable("default")

#######################
# CSS styling
#######################
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

/* metric 카드 배경을 투명으로 변경 */
[data-testid="stMetric"] {
    background-color: transparent;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

#######################
# Load data
#######################
df_reshaped = pd.read_csv("dataset.csv", encoding="cp949")
cols = df_reshaped.columns

#######################
# 타깃 컬럼 자동 탐색 (이름에 '재비산먼지' 포함된 숫자형 컬럼)
#######################
target_col = None
candidate_targets = [c for c in cols if "재비산먼지" in str(c)]
if candidate_targets:
    for c in candidate_targets:
        if pd.api.types.is_numeric_dtype(df_reshaped[c]):
            target_col = c
            break
    if target_col is None:
        target_col = candidate_targets[0]

#######################
# 위도/경도 컬럼 자동 탐색
#######################
lat_candidates = ["위도", "lat", "LAT", "Latitude"]
lon_candidates = ["경도", "lon", "LON", "Longitude"]

lat_col = next((c for c in lat_candidates if c in cols), None)
lon_col = next((c for c in lon_candidates if c in cols), None)

#######################
# Sidebar
#######################
with st.sidebar:
    st.header("⚙️ 설정 및 필터")

    # 측정일자 필터 (있을 때만)
    selected_date = None
    if "측정일자" in cols:
        dates = df_reshaped["측정일자"].dropna().unique()
        selected_date = st.selectbox("측정일자", ["전체"] + sorted(dates.tolist()))

    # 측정시간 필터
    selected_time = None
    if "측정시간" in cols:
        times = df_reshaped["측정시간"].dropna().unique()
        selected_time = st.selectbox("측정시간", ["전체"] + sorted(times.tolist()))

    # 지역명 필터
    selected_region = None
    if "지역명" in cols:
        regions = df_reshaped["지역명"].dropna().unique()
        selected_region = st.selectbox("지역명", ["전체"] + sorted(regions.tolist()))

    # 도로명 필터
    selected_road = None
    if "도로명" in cols:
        roads = df_reshaped["도로명"].dropna().unique()
        selected_road = st.selectbox("도로명", ["전체"] + sorted(roads.tolist()))

    # 기온 범위 슬라이더
    selected_temp = None
    if "기온" in cols and pd.api.types.is_numeric_dtype(df_reshaped["기온"]):
        tmin, tmax = df_reshaped["기온"].min(), df_reshaped["기온"].max()
        selected_temp = st.slider("기온 범위", float(tmin), float(tmax), (float(tmin), float(tmax)))

    # 습도 범위 슬라이더
    selected_hum = None
    if "습도" in cols and pd.api.types.is_numeric_dtype(df_reshaped["습도"]):
        hmin, hmax = df_reshaped["습도"].min(), df_reshaped["습도"].max()
        selected_hum = st.slider("습도 범위", float(hmin), float(hmax), (float(hmin), float(hmax)))

    st.markdown("---")

    # 군집분석 옵션
    use_clustering = st.checkbox("군집 분석(K-Means) 사용", value=True)
    if use_clustering:
        k_clusters = st.slider("클러스터 수 (K)", 2, 10, 4)

#######################
# Filtered DataFrame
#######################
filtered_df = df_reshaped.copy()

if selected_date and selected_date != "전체" and "측정일자" in cols:
    filtered_df = filtered_df[filtered_df["측정일자"] == selected_date]

if selected_time and selected_time != "전체" and "측정시간" in cols:
    filtered_df = filtered_df[filtered_df["측정시간"] == selected_time]

if selected_region and selected_region != "전체" and "지역명" in cols:
    filtered_df = filtered_df[filtered_df["지역명"] == selected_region]

if selected_road and selected_road != "전체" and "도로명" in cols:
    filtered_df = filtered_df[filtered_df["도로명"] == selected_road]

if selected_temp is not None and "기온" in cols:
    filtered_df = filtered_df[
        (filtered_df["기온"] >= selected_temp[0]) &
        (filtered_df["기온"] <= selected_temp[1])
    ]

if selected_hum is not None and "습도" in cols:
    filtered_df = filtered_df[
        (filtered_df["습도"] >= selected_hum[0]) &
        (filtered_df["습도"] <= selected_hum[1])
    ]

#######################
# Top row layout (3 columns)
#######################
col0, col1, col2 = st.columns((1.5, 4.5, 3), gap="medium")

############################################
# Column 0 — 요약 지표
############################################
with col0:
    st.subheader("📊 요약 지표")

    if filtered_df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다.")
    else:
        if target_col and target_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[target_col]):
            st.metric(f"평균 {target_col}", f"{filtered_df[target_col].mean():.2f}")
            st.metric("최고", f"{filtered_df[target_col].max():.2f}")
            st.metric("최저", f"{filtered_df[target_col].min():.2f}")

        if "기온" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["기온"]):
            st.metric("평균 기온", f"{filtered_df['기온'].mean():.1f} °C")

        if "습도" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["습도"]):
            st.metric("평균 습도", f"{filtered_df['습도'].mean():.1f} %")

############################################
# Column 1 — 메인 시각화 (지역별 평균)
############################################
with col1:
    st.subheader("📈 지역별 평균")

    if filtered_df.empty:
        st.warning("데이터가 없습니다.")
    else:
        if "지역명" in filtered_df.columns and target_col and target_col in filtered_df.columns:
            region_mean = (
                filtered_df.groupby("지역명")[target_col]
                .mean()
                .reset_index()
                .sort_values(target_col, ascending=False)
            )

            fig_region = px.bar(
                region_mean,
                x="지역명",
                y=target_col,
                title=f"지역별 평균 {target_col}"
            )
            st.plotly_chart(fig_region, use_container_width=True)

############################################
# Column 2 — Top 10 지역 지도 시각화
############################################
with col2:
    st.subheader("🗺️ Top 10 지역 지도")

    if filtered_df.empty:
        st.info("데이터가 없습니다.")
    elif not (lat_col and lon_col):
        st.info("위도/경도 컬럼을 찾지 못해서 지도 시각화를 할 수 없습니다.")
    elif not ("지역명" in filtered_df.columns and target_col and target_col in filtered_df.columns):
        st.info("지역명 또는 타깃 컬럼이 없어 Top 10 지역을 계산할 수 없습니다.")
    else:
        # 지역별 평균 타깃 값 계산 후 상위 10개
        region_mean = (
            filtered_df.groupby("지역명")[target_col]
            .mean()
            .reset_index()
            .sort_values(target_col, ascending=False)
            .head(10)
        )

        # 각 지역의 위도/경도 평균값 계산
        coord_group = (
            filtered_df.groupby("지역명")[[lat_col, lon_col]]
            .mean()
            .reset_index()
        )

        top_map = pd.merge(region_mean, coord_group, on="지역명", how="left").dropna(subset=[lat_col, lon_col])

        if top_map.empty:
            st.info("Top 10 지역에 대한 위도/경도 정보가 부족합니다.")
        else:
            # 순위 컬럼 추가 (1위가 가장 높은 값)
            top_map = top_map.sort_values(target_col, ascending=False).reset_index(drop=True)
            top_map["rank"] = top_map.index + 1
            top_map["rank_label"] = top_map["rank"].astype(str) + "위"

            # 중심 좌표
            center_lat = top_map[lat_col].mean()
            center_lon = top_map[lon_col].mean()

            fig_map = px.scatter_mapbox(
                top_map,
                lat=lat_col,
                lon=lon_col,
                color="rank_label",  # 범례에 순위 표시
                size=target_col,
                size_max=25,
                hover_name="지역명",
                hover_data={target_col: True, "rank": True, lat_col: False, lon_col: False},
                zoom=6,
                center={"lat": center_lat, "lon": center_lon},
                title=f"Top 10 지역 지도 ({target_col} 기준)"
            )

            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                legend_title_text="순위"
            )

            st.plotly_chart(fig_map, use_container_width=True)

##############################
# Bottom full-width: 3D + ML + Clustering
##############################
st.markdown("---")

with st.container():
    st.subheader("🔍 3D 상관분석 & 예측 모델 + 군집 분석")

    if target_col is None:
        st.warning("컬럼 이름에 '재비산먼지'가 포함된 타깃 컬럼을 찾지 못했습니다.")
    elif filtered_df.empty:
        st.info("필터 조건에 맞는 데이터가 없습니다.")
    else:
        df_ml = filtered_df.copy()

        ############################
        # 1) 3D Scatter Plot (기온·습도·타깃)
        ############################
        if all(c in df_ml.columns for c in ["기온", "습도", target_col]):
            st.markdown(f"### 🌐 기온·습도·{target_col} 3D 산점도")

            df_3d = df_ml.dropna(subset=["기온", "습도", target_col])

            if not df_3d.empty:
                fig_3d = px.scatter_3d(
                    df_3d,
                    x="기온",
                    y="습도",
                    z=target_col,
                    color=target_col,
                    opacity=0.7,
                    title=f"기온·습도·{target_col} 3D 시각화"
                )
                fig_3d.update_traces(marker=dict(size=4))
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("3D 시각화를 위한 기온·습도·타깃 값이 충분하지 않습니다.")

        ############################
        # 2) 예측 모델 (RandomForest + XGBoost)
        ############################
        st.markdown("### 🤖 예측 모델 (Random Forest / XGBoost)")

        df_model = df_ml.dropna(subset=[target_col]).copy()

        if df_model.shape[0] < 5:
            st.info("예측 모델 학습을 위한 데이터가 너무 적습니다.")
        else:
            X_raw = df_model.drop(columns=[target_col])
            X = pd.get_dummies(X_raw, drop_first=True)

            num_in_X = X.select_dtypes(include=[np.number]).columns
            for c in num_in_X:
                if X[c].isna().any():
                    X[c] = X[c].fillna(X[c].median())

            y = df_model[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            # Random Forest 회귀
            rf = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_mse = mean_squared_error(y_test, rf_pred)
            rf_rmse = np.sqrt(rf_mse)

            st.markdown("#### 🌲 Random Forest 성능")
            st.write(f"**MAE:** {rf_mae:.3f}")
            st.write(f"**RMSE:** {rf_rmse:.3f}")

            result_rf = pd.DataFrame({
                "실제 PM": y_test,
                "예측 PM": rf_pred
            })

            fig_rf = px.scatter(
                result_rf,
                x="실제 PM",
                y="예측 PM",
                title="실제값 vs 예측값 (Random Forest)"
            )
            st.plotly_chart(fig_rf, use_container_width=True)

            # XGBoost (가능한 경우)
            if XGB_AVAILABLE:
                st.markdown("#### ⚡ XGBoost 성능")

                xgb = XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                xgb.fit(X_train, y_train)
                xgb_pred = xgb.predict(X_test)

                xgb_mae = mean_absolute_error(y_test, xgb_pred)
                xgb_mse = mean_squared_error(y_test, xgb_pred)
                xgb_rmse = np.sqrt(xgb_mse)

                st.write(f"**MAE:** {xgb_mae:.3f}")
                st.write(f"**RMSE:** {xgb_rmse:.3f}")

                result_xgb = pd.DataFrame({
                    "실제 PM": y_test,
                    "예측 PM": xgb_pred
                })

                fig_xgb = px.scatter(
                    result_xgb,
                    x="실제 PM",
                    y="예측 PM",
                    title="실제값 vs 예측값 (XGBoost)"
                )
                st.plotly_chart(fig_xgb, use_container_width=True)
            else:
                st.warning("XGBoost 패키지가 설치되어 있지 않아 Random Forest만 사용 중입니다.")

        ############################
        # 3) 군집 분석(K-Means)
        ############################
        if use_clustering:
            st.markdown("### 🧩 군집 분석 (K-Means)")

            num_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in num_cols if c != target_col]

            if len(feature_cols) < 2:
                st.info("군집 분석을 위해 최소 2개 이상의 숫자형 컬럼이 필요합니다.")
            else:
                df_cluster = df_ml.copy()
                for c in feature_cols:
                    if df_cluster[c].isna().any():
                        df_cluster[c] = df_cluster[c].fillna(df_cluster[c].median())

                if df_cluster.shape[0] < k_clusters:
                    st.info("데이터 수가 클러스터 수(K)보다 적어 군집 분석이 어렵습니다.")
                else:
                    scaler = StandardScaler()
                    X_clust = scaler.fit_transform(df_cluster[feature_cols])

                    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_clust)

                    df_cluster["cluster"] = clusters

                    if all(c in df_cluster.columns for c in ["기온", "습도"]):
                        x_col, y_col = "기온", "습도"
                    else:
                        x_col, y_col = feature_cols[0], feature_cols[1]

                    fig_cluster = px.scatter(
                        df_cluster,
                        x=x_col,
                        y=y_col,
                        color="cluster",
                        title=f"K-Means 군집 결과 (K={k_clusters})",
                        hover_data=[target_col] if target_col in df_cluster.columns else None
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)

                    st.markdown("#### 군집별 평균 프로파일")
                    cluster_profile = df_cluster.groupby("cluster")[feature_cols + ([target_col] if target_col in df_cluster.columns else [])].mean()
                    st.dataframe(cluster_profile)
