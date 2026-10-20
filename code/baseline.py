"""Reproduce the proposal's corrected static baselines and exploration.

Only Tests 1 and 2 are scored. Test 3 remains outside model selection.
Data and feature definitions: Lee et al. (IMS) and fatemeh-studio repository.
Forecast pairs use explicit timestamp and calibration-window checks.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POS = ['rms', 'std', 'max', 'crest_factor', 'shape_factor', 'impulse_factor',
       'clearance_factor', 'psd_mean', 'psd_std', 'bpfo_energy', 'bpfi_energy',
       'bsf_energy', 'env_bpfo_energy', 'env_bpfi_energy', 'env_bsf_energy']
SIGNED = ['kurt', 'skew', 'env_kurtosis']
UNIT = ['spectral_entropy', 'high_freq_ratio']
FEATURES = POS + SIGNED + UNIT
FAILED = {(1, 3): 'inner race', (1, 4): 'rolling element', (2, 1): 'outer race'}
CALIB, HORIZON, ROLL = 200, 6, 6


def load_run(data_dir, run):
    r = pd.read_csv(data_dir / f'test{run}_raw.csv')
    s = pd.read_csv(data_dir / f'test{run}_spectral.csv')
    assert r.file_index.is_unique and s.file_index.is_unique
    assert np.array_equal(r.file_index, s.file_index)
    d = r.merge(s, on='file_index', validate='one_to_one')
    d['timestamp'] = pd.to_datetime(d['timestamp'])
    assert d.timestamp.is_monotonic_increasing and d.timestamp.is_unique
    assert not d.isna().any().any()
    raw_n = len(d)
    cols = [f'Bearing{b}_ch1_rms' for b in range(1, 5)]
    # A shutdown requires all four channels to fall below 20% of their
    # own calibration median. Only a contiguous trailing shutdown is removed.
    quiet = (d[cols] < 0.2 * d[cols].iloc[:CALIB].median()).all(axis=1)
    cut = len(d)
    while cut and quiet.iloc[cut - 1]:
        cut -= 1
    d = d.iloc[:cut].copy().reset_index(drop=True)
    dt = d.timestamp.diff().dt.total_seconds()
    seg = (~dt.between(540, 660)).cumsum()
    audit = dict(run=run, original_snapshots=raw_n, live_snapshots=len(d),
                 shutdown_removed=raw_n-len(d), missing_values=0,
                 gaps_over_15min=int((dt > 900).sum()),
                 longest_gap_hours=float(dt.max()/3600),
                 elapsed_days=float((d.timestamp.iloc[-1]-d.timestamp.iloc[0]).total_seconds()/86400))
    return d, seg, audit


def make_examples(d, seg, run, bearing):
    f = pd.DataFrame({c: d[f'Bearing{bearing}_ch1_{c}'] for c in FEATURES})
    z = f.copy()
    z[POS] = np.log10(z[POS].clip(lower=1e-12))
    z[SIGNED] = np.sign(z[SIGNED]) * np.log10(1+np.abs(z[SIGNED]))
    rolling = z.groupby(seg).transform(lambda a: a.rolling(ROLL, min_periods=1).mean())
    z = pd.concat([z, rolling.add_suffix('_roll')], axis=1)
    mu, sd = z.iloc[:CALIB].mean(), z.iloc[:CALIB].std()
    sd = sd.mask(sd < 1e-12, 1.0)
    x = np.arcsinh(((z-mu)/sd).to_numpy())
    lr = np.log10(f.rms.to_numpy())
    idx = np.arange(len(d)-HORIZON)
    target_idx = idx+HORIZON
    ts = d.timestamp.to_numpy()
    elapsed = (ts[target_idx]-ts[idx])/np.timedelta64(1, 's')
    history = d.groupby(seg).cumcount().to_numpy()+1
    keep = ((idx >= CALIB) & (seg.to_numpy()[idx] == seg.to_numpy()[target_idx])
            & (np.abs(elapsed-3600) <= 60) & (history[idx] >= ROLL))
    idx = idx[keep]
    target_idx = idx+HORIZON
    roll_lr = pd.Series(lr).groupby(seg).transform(
        lambda a: a.rolling(ROLL, min_periods=ROLL).mean()).to_numpy()
    late_start = ts[0]+(ts[-1]-ts[0])*0.9
    out = pd.DataFrame(dict(run=run, bearing=bearing, input_index=idx,
                           target_index=target_idx, time=ts[idx], target_time=ts[target_idx],
                           y=lr[target_idx], persistence=lr[idx], moving_average=roll_lr[idx],
                           late=(ts[target_idx] >= late_start) & ((run, bearing) in FAILED)))
    for j, name in enumerate(z.columns):
        out[f'x_{name}'] = x[idx, j]
    assert len(out) and np.isfinite(out.select_dtypes('number')).all().all()
    assert (out.input_index >= CALIB).all()
    assert out.target_time.gt(out.time).all()
    return out


def scores(y, p, late):
    e = np.asarray(p)-np.asarray(y)
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.mean(e**2))),
                late_mae=float(np.abs(e[np.asarray(late)]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    start = time.perf_counter()
    root = args.out
    (root/'results').mkdir(parents=True, exist_ok=True)
    (root/'paper/figures').mkdir(parents=True, exist_ok=True)
    examples, raw, audits = {}, {}, []
    for run in (1, 2):
        d, seg, audit = load_run(args.data_dir, run)
        ex = pd.concat([make_examples(d, seg, run, b) for b in range(1,5)], ignore_index=True)
        examples[run], raw[run] = ex, (d, seg)
        audit['eligible_forecasts'] = len(ex)
        audit['per_bearing_forecasts'] = len(ex)//4
        audit['late_failure_forecasts'] = int(ex.late.sum())
        audit['target_mean'] = float(ex.y.mean())
        audit['target_sd'] = float(ex.y.std(ddof=0))
        audit['target_min'] = float(ex.y.min())
        audit['target_max'] = float(ex.y.max())
        audits.append(audit)
    tr, va = examples[1], examples[2]
    xcols = [c for c in tr if c.startswith('x_')]
    X = np.c_[tr[xcols].to_numpy(), np.ones(len(tr))]
    V = np.c_[va[xcols].to_numpy(), np.ones(len(va))]
    residual = (tr.y-tr.moving_average).to_numpy()
    grid = []
    for lam in (1., 10., 100., 1000., 10000., 100000., 1000000.):
        reg = lam*np.eye(X.shape[1]); reg[-1,-1] = 0
        w = np.linalg.solve(X.T@X+reg, X.T@residual)
        p = V@w+va.moving_average.to_numpy()
        grid.append((float(np.abs(p-va.y).mean()), lam, w))
    val_mae, lam, w = min(grid, key=lambda a:a[0])
    rows = []
    for run, ex in examples.items():
        x = np.c_[ex[xcols].to_numpy(), np.ones(len(ex))]
        predictions = {'Last-value persistence': ex.persistence,
                       'Moving-average persistence': ex.moving_average,
                       'Ridge residual': x@w+ex.moving_average.to_numpy()}
        for model, pred in predictions.items():
            rows.append(dict(model=model, split='Train (Test 1)' if run==1 else 'Validation (Test 2)',
                             n=len(ex), **scores(ex.y,pred,ex.late)))
        ex[['run','bearing','input_index','target_index','time','target_time','y',
            'persistence','moving_average','late']].assign(ridge=predictions['Ridge residual']).to_csv(
                root/f'results/test{run}_predictions.csv', index=False)
    results = pd.DataFrame(rows)
    results.to_csv(root/'results/baseline_metrics.csv', index=False)
    pd.DataFrame(audits).to_csv(root/'results/data_audit.csv', index=False)
    pd.DataFrame([{'lambda':a[1],'validation_mae':a[0]} for a in grid]).to_csv(
        root/'results/ridge_selection.csv',index=False)
    np.savez(root/'results/ridge_model.npz', weights=w, columns=np.array(xcols), ridge_lambda=lam)
    elapsed = time.perf_counter()-start
    metadata = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    matplotlib=matplotlib.__version__, selected_lambda=lam,
                    computation_seconds=elapsed, inputs=len(xcols),
                    calibration_snapshots=CALIB, horizon_minutes=60, horizon_tolerance_seconds=60,
                    test3_scored=False, late_definition='Target in last 10% of elapsed live-run time, documented failed bearings only',
                    provenance='Supplied CSVs hash-matched to upstream; raw waveform extraction not independently verified')
    (root/'results/run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(audits).to_string(index=False))
    print(results.round(6).to_string(index=False))
    print('Selected ridge lambda:',lam,'; seconds:',round(elapsed,3))
    plt.rcParams.update({'font.size':8, 'axes.spines.top':False,'axes.spines.right':False,
                         'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(10.0,2.25),gridspec_kw={'width_ratios':[1,1,1.1]})
    colors=['#245a81','#bb6b27','#49836c','#8b5d95']
    for ax,run in zip(axes[:2],(1,2)):
        d,seg = raw[run]
        days = (d.timestamp-d.timestamp.iloc[0]).dt.total_seconds()/86400
        for b,col in zip(range(1,5),colors):
            y=d[f'Bearing{b}_ch1_rms'].copy()
            # Acquisition gaps appear as breaks in the plotted curves.
            y.loc[seg.ne(seg.shift())]=np.nan
            ax.plot(days,y,color=col,lw=.85,label=f'B{b}'+('*' if (run,b) in FAILED else ''))
        ax.set_yscale('log');ax.set_title(f'Test {run}: vibration history',loc='left',fontweight='bold')
        ax.set_xlabel('Elapsed days');ax.set_ylabel('RMS (g)')
        ax.axvspan(0,days.iloc[CALIB-1],color='#dce2e8',alpha=.65,zorder=-2)
        ax.legend(ncol=2,fontsize=6.8,frameon=False,loc='upper left')
        ax.grid(alpha=.17)
    bins=np.linspace(min(tr.y.min(),va.y.min()),max(tr.y.max(),va.y.max()),38)
    for ex,label,color in [(tr,'Train','#245a81'),(va,'Validation','#bb6b27')]:
        axes[2].hist(ex.y,bins=bins,density=True,histtype='step',lw=1.3,label=label,color=color)
    axes[2].set_title('Eligible target distribution',loc='left',fontweight='bold')
    axes[2].set_xlabel(r'$\log_{10}(\mathrm{RMS}/1\,g)$, 1 h ahead')
    axes[2].set_ylabel('Density');axes[2].legend(frameon=False,fontsize=7)
    axes[2].grid(alpha=.17)
    fig.tight_layout(pad=.6,w_pad=1.0)
    fig.savefig(root/'paper/figures/data_overview.pdf',bbox_inches='tight')
    fig.savefig(root/'paper/figures/data_overview.png',dpi=220,bbox_inches='tight')


if __name__=='__main__':
    main()
