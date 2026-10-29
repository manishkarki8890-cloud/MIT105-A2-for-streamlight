"""BaadhiSuchak - Streamlit version (for Streamlit Community Cloud).
Run locally:  pip install -r requirements.txt  &&  streamlit run streamlit_app.py
"""
import datetime as dt
import pandas as pd
import streamlit as st
import flood_core as fc

st.set_page_config(page_title='BaadhiSuchak · Flood risk', page_icon='🌊', layout='wide')


@st.cache_resource(show_spinner='Loading flood model…')
def assets():
    return fc.load_assets()


A = assets()
H, THR = A['hist'], A['thr']
DMIN, DMAX = H.date.min().date(), H.date.max().date()
COLOR = {'Normal': '#0ca30c', 'Watch': '#c98500', 'Warning': '#ec835a', 'Severe': '#d03b3b'}

st.sidebar.title('🌊 BaadhiSuchak')
st.sidebar.caption('Next-day flood risk for 10 Nepal river stations')
page = st.sidebar.radio('Page', ['Dashboard', 'Station history', 'What-if forecast', 'Model'])
st.sidebar.markdown(f'**Model:** {A["name"]}  \n**Alert threshold:** p ≥ {THR:.3f}  \n**Loaded from:** {A["source"]}')
st.sidebar.caption('Prototype for research/education. Not an official DHM warning.')


def sample_note(d):
    if d < dt.date(2025, 1, 1):
        st.warning('Dates before 2025 were used to train the replay model, so these results are in-sample.')
    else:
        st.success('Out-of-sample replay: the model never saw data from 2025 onward.')


if page == 'Dashboard':
    st.title('National flood-risk dashboard')
    st.write("Probability that each station's discharge exceeds its flood threshold **tomorrow**.")
    d = st.date_input('Forecast issued on', dt.date(2025, 10, 4), min_value=DMIN, max_value=DMAX)
    st.caption('Try 2025-10-04 (Koshi Province floods, unseen) or 2024-09-27 (Kathmandu Valley floods, in training data).')
    sample_note(d)
    snap = H[H.date == pd.Timestamp(d)].copy()
    if snap.empty:
        st.info('No data for this date.')
    else:
        snap['p'] = fc.predict(A, snap, replay=True)
        snap['Alert'] = snap.p.apply(lambda p: fc.level(p, THR))
        snap = snap.sort_values('p', ascending=False)
        k1, k2, k3 = st.columns(3)
        k1.metric('Stations on Warning or Severe', int(snap.Alert.isin(['Warning', 'Severe']).sum()))
        k2.metric('Highest risk', f'{snap.p.iloc[0]:.0%}', snap.location.iloc[0])
        k3.metric('Stations that actually flooded next day', int(snap.flood_next_day.sum()))
        c1, c2 = st.columns([1, 1.2])
        with c1:
            m = snap.assign(lat=snap.latitude, lon=snap.longitude, color=snap.Alert.map(COLOR),
                            size=2500 + snap.p * 12000)
            st.map(m[['lat', 'lon', 'color', 'size']], latitude='lat', longitude='lon', color='color', size='size')
            st.caption('🟢 Normal  🟠 Watch  🟧 Warning  🔴 Severe · size = probability')
        with c2:
            tbl = pd.DataFrame({'Station': snap.location, 'River': snap.river,
                                'P(flood tomorrow)': snap.p, 'Alert': snap.Alert,
                                'Discharge m³/s': snap.river_discharge_m3s.round(1),
                                'Threshold m³/s': snap.threshold_m3s.round(1),
                                'Actual next day': snap.flood_next_day.map({1: 'Flood', 0: 'No flood'})})
            st.dataframe(tbl, hide_index=True,
                         column_config={'P(flood tomorrow)': st.column_config.ProgressColumn(
                             format='%.2f', min_value=0.0, max_value=1.0)})

elif page == 'Station history':
    st.title('Station history')
    st.write('Replay the model day by day and compare its alerts with what the river actually did next day.')
    c1, c2, c3 = st.columns([1.4, 1, 1])
    locs = sorted(H.location.unique())
    loc = c1.selectbox('Station', locs, index=locs.index('Chameliya/Nayalbadi'))
    start = c2.date_input('From', dt.date(2025, 6, 1), min_value=DMIN, max_value=DMAX)
    end = c3.date_input('To', dt.date(2025, 10, 31), min_value=DMIN, max_value=DMAX)
    sample_note(start)
    s = H[(H.location == loc) & (H.date >= pd.Timestamp(start)) & (H.date <= pd.Timestamp(end))].copy()
    if s.empty:
        st.info('No data in this range.')
    else:
        s['p'] = fc.predict(A, s, replay=True)
        k1, k2, k3 = st.columns(3)
        k1.metric('Flood days (actual)', int(s.flood_next_day.sum()))
        k2.metric('Flood days caught by an alert', int(((s.p >= THR) & (s.flood_next_day == 1)).sum()))
        k3.metric('Alert days issued', int((s.p >= THR).sum()))
        thr_q = A['meta']['threshold_m3s'][loc]
        st.subheader('Next-day discharge vs flood threshold (m³/s)')
        st.line_chart(s.set_index('date')[['q_next']].rename(columns={'q_next': 'Discharge'})
                      .assign(**{'Flood threshold': thr_q}), color=['#2a78d6', '#d03b3b'])
        st.subheader('Predicted probability of flood tomorrow')
        st.area_chart(s.set_index('date')[['p']].rename(columns={'p': 'P(flood)'})
                      .assign(**{'Alert threshold': THR}), color=['#1baf7a', '#555555'])

elif page == 'What-if forecast':
    st.title('What-if forecast')
    st.write("Enter today's observations for a station. Leave optional fields at 0 to use the station's typical value.")
    meta = A['meta']
    with st.form('whatif'):
        c1, c2 = st.columns(2)
        loc = c1.selectbox('Station', sorted(meta['threshold_m3s']), index=sorted(meta['threshold_m3s']).index('Kusum'),
                           format_func=lambda l: f"{l} (flood threshold {meta['threshold_m3s'][l]:.1f} m³/s)")
        month = c2.selectbox('Month', list(range(1, 13)), index=7,
                             format_func=lambda m: dt.date(2025, m, 1).strftime('%B'))
        q_today = c1.number_input('River discharge today (m³/s)', min_value=0.0, value=600.0)
        q_lag1 = c2.number_input('Discharge yesterday (m³/s)', min_value=0.0, value=420.0)
        rain_today = c1.number_input('Rain today (mm)', min_value=0.0, value=70.0)
        rain_lag1 = c2.number_input('Rain yesterday (mm)', min_value=0.0, value=40.0)
        rain_lag2 = c1.number_input('Rain 2 days ago (mm)', min_value=0.0, value=20.0)
        rain_30d = c2.number_input('Rain, last 30 days (mm) — 0 = typical', min_value=0.0, value=0.0)
        soil = c1.number_input('Soil moisture (m³/m³, 0–0.6) — 0 = typical', min_value=0.0, max_value=0.6, value=0.0)
        rh = c2.number_input('Relative humidity (%) — 0 = typical', min_value=0.0, max_value=100.0, value=0.0)
        go = st.form_submit_button('Predict flood risk', type='primary')
    if go:
        f = {'location': loc, 'month': month, 'q_today': q_today, 'q_lag1': q_lag1, 'rain_today': rain_today,
             'rain_lag1': rain_lag1, 'rain_lag2': rain_lag2,
             'rain_30d': rain_30d or None, 'soil': soil or None, 'rh': rh or None}
        X, thr_q = fc.build_manual(A, f)
        p = float(fc.predict(A, X)[0]); lv = fc.level(p, THR)
        c1, c2 = st.columns([1, 1.4])
        c1.metric(f'Probability of exceeding {thr_q:.1f} m³/s tomorrow', f'{p:.0%}')
        c1.markdown(f"<span style='background:{COLOR[lv]};color:white;padding:6px 14px;border-radius:99px;"
                    f"font-weight:700'>{lv}</span>", unsafe_allow_html=True)
        c1.progress(min(max(p, 0.0), 1.0))
        c2.markdown('**What is driving this?**')
        c2.dataframe(fc.drivers(A, X, loc), hide_index=True)
        st.caption(f'Alert levels: Watch ≥ {THR/2:.0%}, Warning ≥ {THR:.0%}, Severe ≥ 60%. '
                   'The threshold favours catching floods (recall) over avoiding false alarms.')

else:
    R = A['results']
    st.title('Model card')
    st.write(f"Trained on 2023–2024; tested on unseen data from 1 Jan 2025 to 31 Aug 2026 "
             f"({R['n_test']} station-days, {R['pos_test']} flood days).")
    cols = ['precision', 'recall', 'f1', 'f2', 'mcc', 'roc_auc', 'pr_auc', 'brier', 'onset_recall']
    df = pd.DataFrame(R['results']).T[cols].astype(float).round(3)
    df.index = [n + (' ★' if n == R['best'] else '') for n in df.index]
    st.dataframe(df)
    st.caption(f"Onset recall = share of the {R['onset_events']} days where a flood began with no flood the day "
               'before that the model flagged. The persistence baseline cannot catch any of them.')
    st.markdown('**Intended use:** decision support for municipal disaster-management officers, NGOs and '
                "community early-warning focal points. It complements, and does not replace, DHM's official warnings.")
