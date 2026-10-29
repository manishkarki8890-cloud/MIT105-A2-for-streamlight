"""Shared model logic for the BaadhiSuchak apps (Streamlit and Flask). No web framework imports here."""
import json, os, warnings
import numpy as np, pandas as pd, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, 'model')


def _retrain(hist, feats, num):
    """Fallback if the saved model can't be loaded (e.g. a different scikit-learn version on the host).
    Rebuilds the selected Random Forest with its tuned settings (about 20-40 s)."""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier

    def make():
        prep = ColumnTransformer([('num', 'passthrough', num),
                                  ('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])])
        rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
        return Pipeline([('prep', prep), ('clf', rf)])
    train = hist[hist.date <= '2024-12-31']
    m_eval = make().fit(train[feats], train['flood_next_day'])
    m_final = make().fit(hist[feats], hist['flood_next_day'])
    return m_final, m_eval


def load_assets():
    hist = pd.read_csv(os.path.join(MD, 'nepal_features.csv'), parse_dates=['date'])
    meta = json.load(open(os.path.join(MD, 'thresholds.json')))
    results = json.load(open(os.path.join(MD, 'results.json')))
    cfg = json.load(open(os.path.join(MD, 'features.json')))
    feats, thr, name = cfg['features'], cfg['threshold'], cfg['name']
    num = [f for f in feats if f != 'location']
    source = 'saved model'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')          # treat version-mismatch warnings as failure
            b = joblib.load(os.path.join(MD, 'flood_model.joblib'))
        model, model_eval = b['model'], b.get('model_eval', b['model'])
        model_eval.predict_proba(hist[feats].head(2))  # smoke test
    except Exception as e:
        model, model_eval = _retrain(hist, feats, num)
        source = f'retrained on start-up ({type(e).__name__})'
    stations = hist.groupby('location').agg(lat=('latitude', 'first'), lon=('longitude', 'first'),
                                            river=('river', 'first'), basin=('basin', 'first')).reset_index()
    return dict(hist=hist, meta=meta, results=results, feats=feats, num=num, thr=thr, name=name,
                model=model, model_eval=model_eval, stations=stations, source=source)


def level(p, thr):
    if p >= max(0.6, thr * 2.5): return 'Severe'
    if p >= thr: return 'Warning'
    if p >= thr / 2: return 'Watch'
    return 'Normal'


def predict(A, X, replay=False):
    m = A['model_eval'] if replay else A['model']
    return m.predict_proba(X[A['feats']])[:, 1]


def build_manual(A, f):
    """Short form (what a field officer knows) -> full feature vector.
    Unknown features fall back to the station's median for that calendar month."""
    hist, meta, num = A['hist'], A['meta'], A['num']
    loc = f['location']; month = int(f.get('month', 7))
    row = hist[(hist.location == loc) & (hist.date.dt.month == month)][num].median().copy()
    thr, med = meta['threshold_m3s'][loc], max(meta['median_m3s'][loc], 0.05)
    g = lambda k, d=None: float(f[k]) if f.get(k) not in (None, '') else d
    r0, r1, r2, r3 = g('rain_today', 0), g('rain_lag1', 0), g('rain_lag2', 0), g('rain_lag3', 0)
    q0, q1, q2 = g('q_today'), g('q_lag1'), g('q_lag2')
    q1 = q0 if q1 is None else q1; q2 = q1 if q2 is None else q2
    row['precipitation_mm'], row['rain_lag1'], row['rain_lag2'], row['rain_lag3'] = r0, r1, r2, r3
    row['rain_sum_3d'] = r0 + r1 + r2
    row['rain_sum_7d'] = g('rain_7d', max(row['rain_sum_3d'] + r3, row['rain_sum_7d']))
    row['rain_sum_14d'] = max(row['rain_sum_14d'], row['rain_sum_7d'])
    row['rain_sum_30d'] = g('rain_30d', max(row['rain_sum_30d'], row['rain_sum_14d']))
    row['precipitation_hours'] = g('rain_hours', row['precipitation_hours'])
    row['soil_moisture_0_100cm_m3m3'] = g('soil', row['soil_moisture_0_100cm_m3m3'])
    row['relative_humidity_mean_pct'] = g('rh', row['relative_humidity_mean_pct'])
    row['q_norm'], row['q_norm_lag1'], row['q_norm_lag2'] = q0 / med, q1 / med, q2 / med
    row['q_change_1d'] = row['q_norm'] - row['q_norm_lag1']
    row['q_norm_max_7d'] = max(row['q_norm'], row['q_norm_lag1'], row['q_norm_lag2'])
    row['q_to_thr'], row['q_to_thr_lag1'] = q0 / thr, q1 / thr
    row['q_to_thr_max_3d'] = max(q0, q1, q2) / thr
    row['flood_today'] = int(q0 > thr)
    doy = pd.Timestamp(2025, month, 15).dayofyear
    row['doy_sin'], row['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    row['is_monsoon'] = int(6 <= month <= 9)
    X = pd.DataFrame([row]); X['location'] = loc
    return X, thr


def drivers(A, X, loc):
    typ = A['hist'][A['hist'].location == loc][A['num']].median()
    rows = []
    for k, label, unit in [('q_to_thr', 'Discharge vs flood threshold', '×'), ('rain_sum_3d', '3-day rainfall', 'mm'),
                           ('rain_sum_30d', '30-day rainfall', 'mm'), ('soil_moisture_0_100cm_m3m3', 'Soil moisture', 'm³/m³'),
                           ('q_change_1d', 'Discharge change (1 day)', '× median')]:
        v, t = float(X[k].iloc[0]), float(typ[k])
        rows.append({'Input': label, 'Now': round(v, 3), 'Station typical': round(t, 3), 'Unit': unit,
                     'Above normal': '▲' if (v > t * 1.5 if t > 0 else v > 0.05) else ''})
    return pd.DataFrame(rows)
