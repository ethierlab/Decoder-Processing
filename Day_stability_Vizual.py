import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import shapiro
# ---------- fixed colors per decoder ----------
DECODER_COLORS = {"GRU":"red","LSTM":"tab:blue","Linear":"tab:green","LiGRU":"tab:orange"}

# ---------- IO ----------
def load_results(results_dir, recalc_day_from_date=False):
    pattern = os.path.join(results_dir, "crossday_results_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PKL files match {pattern}")
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
            print(f"Loaded {f}  shape={df.shape}")
        except Exception as e:
            print(f"Could not read {f}: {e}")
    if not dfs:
        raise RuntimeError("No results could be loaded.")
    all_df = pd.concat(dfs, ignore_index=True)

    # Optional: force global timeline from real recording dates
    if recalc_day_from_date:
        date_col = "day" if "day" in all_df.columns else ("date" if "date" in all_df.columns else None)
        if date_col:
            all_df[date_col] = pd.to_datetime(all_df[date_col], errors="coerce")
            base = all_df[date_col].min()
            all_df["day_int"] = (all_df[date_col] - base).dt.days
            print(f"[INFO] Recomputed day_int from {date_col}. Baseline = {base.date()}")
        else:
            print("[WARN] No 'day' or 'date' column found; using stored day_int.")

    if "day_int" in all_df:
        all_df["day_int"] = pd.to_numeric(all_df["day_int"], errors="coerce")

    # Build mean_vaf if missing (mean across EMG channels within each fold/day/condition)
    if "mean_vaf" not in all_df.columns:
        all_df["mean_vaf"] = all_df.groupby(
            ["decoder","dim_red","align","fold","day"], dropna=False
        )["vaf"].transform("mean")

    # Normalize columns we may rely on
    if "emg_channel" not in all_df.columns:
        all_df["emg_channel"] = -1
    if "fold" not in all_df.columns:
        all_df["fold"] = 0
    return all_df

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ---------- helpers ----------
def build_avg_muscles_points(df, exclude_channels=(0,5,6),
                             decoder=None, dim_red=None, align=None,
                             per_day=True):
    """
    Each row = one point averaged across muscles (channels not excluded).
    Keys preserved: decoder, dim_red, align, day_int, fold (and seed if present).
    If per_day=True (default) you get one point per (day, fold[,seed]).
    """
    d = df.copy()
    if decoder: d = d[d["decoder"]==decoder]
    if dim_red: d = d[d["dim_red"]==dim_red]
    if align:   d = d[d["align"]==align]

    d = d[~d["emg_channel"].isin(exclude_channels)]

    keys = ["decoder","dim_red","align","day_int","fold"]
    if "seed" in d.columns:
        keys.append("seed")

    g = (
        d.groupby(keys, dropna=False)["vaf"]
         .mean()
         .reset_index()
         .rename(columns={"vaf":"vaf_avg_muscles"})
    )
    return g


def violin_avg_muscles_by_day_grouped(df_points, title="", save=None, ylim=(-0.5,1.05)):
    """
    Grouped violin: per day on x-axis; within each day, one violin per decoder.
    df_points must have columns: decoder, day_int, vaf_avg_muscles.
    """
    if df_points.empty:
        print("[violin-grouped] nothing to plot"); return

    # order + spacing
    days   = sorted(df_points["day_int"].dropna().unique())
    decs   = sorted(df_points["decoder"].dropna().unique())
    day_gap = 1.4
    base_pos = np.arange(len(days), dtype=float) * day_gap
    G = len(decs)
    max_span = 0.9
    offsets = np.linspace(-max_span/2, max_span/2, G)
    widths  = min(0.28, (max_span/G)*0.85)

    # build data & positions in parallel
    data, positions, colors, med_x, med_y = [], [], [], [], []
    for gi, dec in enumerate(decs):
        gdec = df_points[df_points["decoder"]==dec]
        for i, d in enumerate(days):
            vals = gdec[gdec["day_int"]==d]["vaf_avg_muscles"].values
            data.append(vals if len(vals) else np.array([]))
            x = base_pos[i] + offsets[gi]
            positions.append(x)
            colors.append(DECODER_COLORS.get(dec, None))
            if len(vals):
                med_x.append(x)
                med_y.append(np.median(vals))

    # draw violins
    plt.figure(figsize=(12,6))
    parts = plt.violinplot(data, positions=positions, widths=widths, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        if c: body.set_facecolor(c)
        body.set_alpha(0.45)

    # overlay medians
    if med_x:
        plt.scatter(med_x, med_y, s=18, color="k", zorder=3, label="Median")

    # axis / legend
    plt.xticks(base_pos, [str(int(d)) for d in days])
    plt.ylim(0, ylim)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xlabel("Day"); plt.ylabel("VAF (avg across muscles)")
    plt.title(title)

    handles = [mpatches.Patch(facecolor=DECODER_COLORS.get(dec,"gray"), alpha=0.45, label=dec)
               for dec in decs]
    handles.append(mpatches.Patch(color="black", label="Median"))  # legend dot marker substitute
    plt.legend(handles=handles, title="Decoder", loc="center left", bbox_to_anchor=(1,0.5), frameon=False)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()





# inverse error function without importing scipy.special explicitly
# (uses numpy's polynomial approx if available; fallback simple newton)
def erfinv(x):
    # good-enough approximation for small n visualization
    a = 0.147  # Hastings
    sgn = np.sign(x)
    ln = np.log(1 - x**2)
    first = 2/(np.pi*a) + ln/2
    second = ln/a
    return sgn * np.sqrt( np.sqrt(first**2 - second) - first )




def build_err_stats_from_raw(df_raw, group_mode, series):
    """
    Calcule mean±std par jour et par 'series' (fold/emg_channel/decoder) SUR LE DF BRUT (vaf),
    en respectant la dimension moyennée par group_mode.
    Retourne un DataFrame avec colonnes: day_int, series, y_mean, y_std.
    """
    base_col = "vaf"
    if base_col not in df_raw.columns:
        raise ValueError("Expected 'vaf' in raw dataframe for errorbar stats.")

    if group_mode == "avg_channels" and series == "fold":
        # moyenne/dispersion à travers les canaux → une stat par (jour, fold)
        g = df_raw.groupby(["day_int","fold"])[base_col].agg(["mean","std"]).reset_index()
        g = g.rename(columns={"fold":"series","mean":"y_mean","std":"y_std"})
    elif group_mode == "avg_folds" and series == "emg_channel":
        # moyenne/dispersion à travers les folds → une stat par (jour, canal)
        g = df_raw.groupby(["day_int","emg_channel"])[base_col].agg(["mean","std"]).reset_index()
        g = g.rename(columns={"emg_channel":"series","mean":"y_mean","std":"y_std"})
    elif group_mode in ("avg_channels_folds","avg_all"):
        # tout écrasé → une stat par jour
        g = df_raw.groupby(["day_int"])[base_col].agg(["mean","std"]).reset_index()
        g["series"] = "all"
        g = g.rename(columns={"mean":"y_mean","std":"y_std"})
    else:
        # fallback générique: dispersion sur les duplicatas de (jour, series)
        g = df_raw.groupby(["day_int", series])[base_col].agg(["mean","std"]).reset_index()
        g = g.rename(columns={series:"series","mean":"y_mean","std":"y_std"})

    g["y_std"] = np.nan_to_num(g["y_std"].values, nan=0.0)
    return g

def scatter_with_errorbars_by_day(df_points, df_raw, group_mode, series="decoder",
                                  title="", save=None, ylim=None, offset_span=0.6):
    """
    Jittered scatter (df_points, already aggregated per group_mode) + mean±std (from df_raw).
    Each 'series' gets a small horizontal offset within each day so curves don't overlap.
    offset_span controls total width of the cluster (e.g., 0.6 spreads series over +/-0.3).
    """
    if df_points.empty:
        print("[scatter+err] nothing to plot"); return

    plt.figure(figsize=(10,6))
    rng = np.random.default_rng(0)

    # series & colors
    series_vals = sorted(df_points[series].dropna().unique())
    # offsets per series (centered around day)
    if len(series_vals) <= 1:
        offsets = {series_vals[0]: 0.0} if series_vals else {}
    else:
        offs = np.linspace(-offset_span/2, offset_span/2, len(series_vals))
        offsets = {s:o for s, o in zip(series_vals, offs)}

    # color map
    color_map = {}
    if series == "decoder":
        for s in series_vals:
            color_map[s] = DECODER_COLORS.get(s, None)
    else:
        cycl = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        for i, s in enumerate(series_vals):
            color_map[s] = cycl[i % max(1, len(cycl))]

    days = sorted(df_points["day_int"].dropna().unique())

    # scatter cloud per series with offset
    jitter_width = 0.08  # a bit tighter since we’re offsetting series
    for s in series_vals:
        g = df_points[df_points[series]==s]
        if g.empty: continue
        off = offsets.get(s, 0.0)
        for d in days:
            vals = g[g["day_int"]==d]["y"].values
            if len(vals)==0: continue
            center = np.full(len(vals), float(d) + off)
            xx = jitter(center, jitter_width, rng)
            plt.scatter(xx, vals, s=10, alpha=0.35, c=color_map[s])

    # mean±std computed from RAW (pre-aggregation), same offsets
    stats = build_err_stats_from_raw(df_raw, group_mode=group_mode, series=series)
    for s in series_vals:
        gs = stats[stats["series"]==s]
        if gs.empty: continue
        xs = gs["day_int"].values.astype(float) + offsets.get(s, 0.0)
        ys = gs["y_mean"].values
        ye = np.nan_to_num(gs["y_std"].values, nan=0.0)
        plt.errorbar(xs, ys, yerr=ye, fmt="o", capsize=3, alpha=0.95,
                     label=str(s), color=color_map[s], zorder=5)

    # axes

    plt.ylim(0, 1)
    if len(days):
        xmin, xmax = float(min(days)), float(max(days))
        pad = 0.5 + (offset_span/2)
        plt.xlim(xmin - pad, xmax + pad)

    plt.grid(True, alpha=0.3)
    plt.xlabel("Days from day0"); plt.ylabel("VAF")
    plt.title(title)
    _legend_outside(series)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()



def jitter(vals, width=0.12, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return vals + rng.uniform(-width, width, size=len(vals))

# ---------- grouping (channels/folds only) ----------
def apply_group_mode(df, group_mode, use_mean):
    """
    Produces a DataFrame with column 'y' after averaging per group_mode.
    Base metric = 'mean_vaf' if use_mean else 'vaf'.

    group_mode:
      raw | avg_folds | avg_channels | avg_channels_folds | avg_all (alias of channels_folds)
    """
    df = df.copy()
    base_metric = "mean_vaf" if use_mean else "vaf"
    if base_metric not in df.columns:
        raise ValueError(f"Column '{base_metric}' not found.")

    mode_map = {
        "raw": [],
        "avg_folds": ["fold"],
        "avg_channels": ["emg_channel"],
        "avg_channels_folds": ["emg_channel", "fold"],
        "avg_all": ["emg_channel", "fold"],
    }
    avg_over = mode_map[group_mode]

    entity_keys = ["decoder", "dim_red", "align", "day_int"]
    detail_keys = [k for k in ["emg_channel", "fold"] if k not in avg_over]
    group_keys = entity_keys + detail_keys

    if not avg_over:
        return df[group_keys + [base_metric]].rename(columns={base_metric: "y"})

    grouped = df.groupby(group_keys, dropna=False)[base_metric].mean().reset_index()
    grouped = grouped.rename(columns={base_metric: "y"})
    return grouped

# ---------- day0 injection from crossval ----------
def inject_day0_from_crossval(df_grouped, df_source, dim_red, align):
    """
    If day 0 is missing for (dim_red, align != 'crossval'), copy crossval rows
    of the same dim_red and set day_int=0 so you get a baseline box at day 0.

    df_grouped: the already-grouped frame (has column 'y')
    df_source:  the same grouped frame for the whole condition set (has 'y' too)
    """
    # Nothing to do if we're already on crossval or day 0 exists
    if align == "crossval":
        return df_grouped
    days_present = set(pd.to_numeric(df_grouped["day_int"], errors="coerce").dropna().unique())
    if 0 in days_present:
        return df_grouped

    # Ensure unique columns in both inputs
    df_grouped = df_grouped.loc[:, ~df_grouped.columns.duplicated()].copy()
    df_source  = df_source.loc[:,  ~df_source.columns.duplicated()].copy()

    # Take crossval rows from the grouped source (so it already has 'y')
    base = df_source[(df_source["dim_red"] == dim_red) & (df_source["align"] == "crossval")].copy()
    if base.empty:
        return df_grouped

    # Stamp them to day 0
    base["day_int"] = 0

    # Force identical columns (order + presence)
    cols = list(df_grouped.columns)
    for c in cols:
        if c not in base.columns:
            base[c] = np.nan
    base = base[cols]

    # Be paranoid about dtype of 'y'
    if "y" in base.columns:
        base["y"] = pd.to_numeric(base["y"], errors="coerce")

    # Finally concatenate
    out = pd.concat([df_grouped, base], axis=0, ignore_index=True, sort=False)
    return out


# ---------- plotting ----------
def _legend_outside(title=None):
    plt.legend(title=title, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

def scatter_by_day(df, x="day_int", y="y", hue=None, title="", save=None, ylim=None):
    if df.empty:
        print("[scatter] nothing to plot"); return
    days = sorted(df[x].dropna().unique())
    plt.figure(figsize=(10,6))
    rng = np.random.default_rng(0)

    if hue is None:
        for d in days:
            yy = df.loc[df[x]==d, y].values
            xx = jitter(np.full_like(yy, d, dtype=float), 0.12, rng)
            plt.scatter(xx, yy, s=12, alpha=0.6)
    else:
        # one series per hue value so the legend isn't 400 items long
        for val, g in df.groupby(hue):
            color = DECODER_COLORS.get(val, None) if hue=="decoder" else None
            Xs, Ys = [], []
            for d in days:
                gg = g[g[x]==d]
                if len(gg)==0: continue
                Xs.append(jitter(np.full(len(gg), d, float), 0.12, rng))
                Ys.append(gg[y].values)
            if Xs:
                X = np.concatenate(Xs); Y = np.concatenate(Ys)
                plt.scatter(X, Y, s=12, alpha=0.6, label=str(val), c=color)

    # sane x-lims even for single-day clouds
    if len(days) == 1:
        d = days[0]; plt.xlim(d - 0.5, d + 0.5)
    else:
        plt.xlim(min(days) - 0.5, max(days) + 0.5)

    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True, alpha=0.3)
    plt.xlabel("Days from day0")
    plt.ylabel("VAF")
    plt.title(title)
    if hue is not None:
        _legend_outside(hue)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()

def box_by_day_simple(df, x="day_int", y="y", title="", save=None, ylim=None):
    if df.empty:
        print("[box] nothing to plot"); return
    groups = [(d, g[y].values) for d, g in sorted(df.groupby(x), key=lambda kv: kv[0])]
    labels = [str(int(d)) for d, _ in groups]
    data   = [vals for _, vals in groups]
    plt.figure(figsize=(10,6))
    plt.boxplot(data, labels=labels, showfliers=False)
    if ylim is not None: plt.ylim(ylim)
    plt.xlabel("Day"); plt.ylabel("VAF"); plt.title(title)
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()

def box_by_day_grouped(df, x="day_int", y="y", group="decoder", title="", save=None, ylim=None):
    """
    Grouped boxplot per day with extra spacing to avoid overlap between days.
    """
    if df.empty:
        print("[box] nothing to plot"); return

    days = sorted(df[x].dropna().unique())
    groups = sorted(df[group].dropna().unique())

    # widen spacing between days to prevent overlap across adjacent days
    day_gap = 1.4  # >1 adds whitespace between day blocks
    base_positions = np.arange(len(days), dtype=float) * day_gap

    G = len(groups)
    max_group_span = 0.8  # keep group cluster narrower than day_gap
    offsets = np.linspace(-max_group_span/2, max_group_span/2, G)
    boxw = min(0.18, (max_group_span / G) * 0.7)

    data, positions, colors = [], [], []
    for gi, gv in enumerate(groups):
        gdf = df[df[group] == gv]
        for i, d in enumerate(days):
            vals = gdf[gdf[x] == d][y].values
            data.append(vals if len(vals) else np.array([]))
            positions.append(base_positions[i] + offsets[gi])
            colors.append(DECODER_COLORS.get(gv, None) if group == "decoder" else None)

    plt.figure(figsize=(12,6))
    bp = plt.boxplot(data, positions=positions, widths=boxw,
                     showfliers=False, patch_artist=True)

    for patch, c in zip(bp["boxes"], colors):
        if c is not None:
            patch.set_facecolor(c); patch.set_alpha(0.6)
    for elem in ["medians", "whiskers", "caps"]:
        for artist in bp[elem]:
            artist.set_linewidth(1.0)

    plt.xticks(base_positions, [str(int(d)) for d in days])
    if ylim is not None: plt.ylim(ylim)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xlabel("Day"); plt.ylabel("VAF"); plt.title(title)

    if group == "decoder":
        handles = [mpatches.Patch(facecolor=DECODER_COLORS.get(g, "gray"), alpha=0.6, label=str(g))
                   for g in groups]
        plt.legend(handles=handles, title="Decoder", loc="center left",
                   bbox_to_anchor=(1, 0.5), frameon=False)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()

def violin_by_day_one_decoder(df, decoder, x="day_int", y="y", title="", save=None, ylim=None):
    sub = df[df["decoder"]==decoder]
    if sub.empty:
        print("[violin] nothing to plot"); return
    days = sorted(sub[x].dropna().unique())
    data = [sub[sub[x]==d][y].values for d in days]
    plt.figure(figsize=(12,6))
    parts = plt.violinplot(data, positions=np.arange(len(days)), widths=0.8, showextrema=False)
    for b in parts["bodies"]:
        b.set_alpha(0.5); b.set_facecolor(DECODER_COLORS.get(decoder, "gray"))
    meds = [np.median(d) if len(d)>0 else np.nan for d in data]
    plt.scatter(np.arange(len(days)), meds, s=20, color="black", zorder=3, label="Median")
    plt.xticks(np.arange(len(days)), [str(int(d)) for d in days])
    if ylim is not None: plt.ylim(ylim)
    plt.grid(True, axis="y", alpha=0.3); plt.xlabel("Day"); plt.ylabel("VAF")
    plt.title(title or f"{decoder} • VAF distribution per day")
    _legend_outside(None)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()

# ---------- figure generator ----------
def make_everything(df, outdir, one_decoder=None, one_align_for_cumsum="aligned",
                    per_channel=False, emg_channels=None, use_mean=False, group_mode="raw",
                    do_violin=False, add_day0_from_crossval=False, split_by_channel=False,
                    series_by="decoder", errorbars=False):
    # Split-by-channel: produce a full figure set for each channel independently
    if split_by_channel:
        chan_list = (emg_channels if (per_channel and emg_channels is not None)
                     else sorted(df["emg_channel"].dropna().unique()))
        if group_mode in ("avg_channels", "avg_channels_folds", "avg_all"):
            print("[WARN] group_mode averages channels; overriding to 'raw' for split_by_channel.")
            group_mode_local = "raw"
        else:
            group_mode_local = group_mode

        for ch in chan_list:
            df_ch = df[df["emg_channel"] == ch].copy()
            if df_ch.empty:
                print(f"[info] Channel {ch}: no data, skipping.")
                continue
            make_everything(
                df=df_ch,
                outdir=os.path.join(outdir, "by_channel", f"ch{ch}"),
                one_decoder=one_decoder,
                one_align_for_cumsum=one_align_for_cumsum,
                per_channel=False,
                emg_channels=None,
                use_mean=use_mean,
                group_mode=group_mode_local,
                do_violin=do_violin,
                add_day0_from_crossval=add_day0_from_crossval,
                split_by_channel=False,
                series_by=series_by,          
                errorbars=errorbars
            )
        return

    # Optional channel filter BEFORE grouping
    if per_channel and emg_channels is not None:
        df = df[df["emg_channel"].isin(emg_channels)]

    # Apply grouping -> produce column 'y'
    df_g = apply_group_mode(df, group_mode=group_mode, use_mean=use_mean)

    decoders = sorted(df_g["decoder"].dropna().unique())
    dimreds  = [d for d in ["PCA","UMAP"] if d in set(df_g["dim_red"])]
    aligns   = [a for a in ["crossval","direct","aligned"] if a in set(df_g["align"])]

    suffix = f"__{group_mode}"
    if use_mean: suffix += "__meanvaf"
    if per_channel and emg_channels is not None:
        suffix += f"__perch_{'-'.join(map(str,emg_channels))}"

    # 1) All decoders, per align, per dimred (series=hue=decoder) + grouped boxes
    for dim in dimreds:
        for al in aligns:
            sub = df_g[(df_g["dim_red"]==dim) & (df_g["align"]==al)]
            if sub.empty: continue
            if add_day0_from_crossval:
                sub = inject_day0_from_crossval(sub, df_g, dim_red=dim, align=al)
            base = ensure_dir(os.path.join(outdir, "1_all_decoders", f"{dim}", f"{al}{suffix}"))
            title = f"All decoders • align={al} • {dim} • {group_mode}" + (" • mean_vaf" if use_mean else "")
            if errorbars:
                df_raw = df[(df["dim_red"]==dim) & (df["align"]==al)].copy()
                scatter_with_errorbars_by_day(sub, df_raw, group_mode=group_mode, series=series_by,
                                  title=title, save=os.path.join(base, "scatter.png"),
                                  ylim=(-0.5, 1.05))
            else:
                # l’ancien scatter
                scatter_by_day(sub, y="y", hue=series_by,
                            title=title, save=os.path.join(base, "scatter.png"),
                            ylim=(-0.5, 1.05))
            box_by_day_grouped(sub, y="y", group="decoder", title=title+" (boxplot)",
                               save=os.path.join(base, "box.png"), ylim=(-0.5, 1.05))

    # default focus decoder
    if one_decoder is None:
        one_decoder = decoders[0] if decoders else None
    if one_decoder is None:
        print("No decoders found, skipping decoder-specific sections.")
        return

    # 2) One decoder, align overlay (PCA-only & UMAP-only)
    decoders_to_plot = [one_decoder] if one_decoder else decoders
    if not decoders_to_plot:
        print("No decoders found, skipping decoder-specific sections.")
        return

    for dec in decoders_to_plot:
        for dim in dimreds:
            for al in aligns:
                sub  = df_g[(df_g["decoder"] == dec) & (df_g["dim_red"] == dim) & (df_g["align"] == al)]
                if sub.empty: 
                    continue
                base  = ensure_dir(os.path.join(outdir, "2_one_decoder", dec, dim, f"{al}{suffix}"))
                title = f"{dec} • {dim} • align={al} • {group_mode}" + (" • mean_vaf" if use_mean else "")
                if errorbars:
                    # stats des barres = std ENTRE folds, calculées sur le DF BRUT filtré au même scope
                    df_raw = df[(df["decoder"] == dec) & (df["dim_red"] == dim) & (df["align"] == al)].copy()
                    scatter_with_errorbars_by_day(
                        sub, df_raw, group_mode=group_mode, series=series_by,
                        title=title, save=os.path.join(base, "scatter.png"),
                        ylim=(-0.5, 1.05)
                    )
                else:
                    scatter_by_day(
                        sub, y="y", hue=series_by,
                        title=title, save=os.path.join(base, "scatter.png"),
                        ylim=(-0.5, 1.05)
                    )
                box_by_day_simple(
                    sub, y="y", title=title + " (boxplot)",
                    save=os.path.join(base, "box.png"), ylim=(-0.5, 1.05)
                )

    # 3) Optional violins per day for the chosen decoder
    if do_violin:
        for dec in decoders_to_plot:
            base_vi = ensure_dir(os.path.join(outdir, "2_one_decoder", dec, "VIOLINS"+suffix))
            for dim in dimreds:
                sub = df_g[(df_g["decoder"] == dec) & (df_g["dim_red"] == dim)]
                if sub.empty: 
                    continue
                title = f"{dec} • {dim} • per-day distribution (violin) • {group_mode}"
                violin_by_day_one_decoder(
                    sub, dec, title=title,
                    save=os.path.join(base_vi, f"violin_{dim}.png"),
                    ylim=(-0.5, 1.05)
                )

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate cross-day decoding figures.")
    ap.add_argument("--results_dir", type=str, default=".", help="Folder with crossday_results_*.pkl")
    ap.add_argument("--out_dir", type=str, default="figs_all", help="Where to save figures")
    ap.add_argument("--decoder", type=str, default="", help="Focus decoder for section 2 & violins")
    ap.add_argument("--violin_avg_muscles_grouped", action="store_true",
                help="Grouped per-day violins of avg VAF across muscles, split by decoder.")
    ap.add_argument("--violin_dimred", type=str, default="",
                    help="Filter dim red (e.g., PCA).")
    ap.add_argument("--violin_align", type=str, default="",
                    help="Filter align (e.g., aligned).")
    ap.add_argument("--violin_exclude_channels", nargs="+", type=int, default=[0,5,6],
                    help="Channels to exclude when averaging across muscles.")
    ap.add_argument("--only_violin", action="store_true",
                    help="If set, skip the big 'make_everything' suite and only draw the grouped violin.")
    ap.add_argument("--recalc_day_from_date", action="store_true",
                    help="Recompute day_int from real recording dates across all files")
    ap.add_argument("--per_channel", action="store_true",
                    help="Filter by EMG channels before grouping")
    ap.add_argument("--emg_channels", nargs="+", type=int, default=None,
                    help="Use with --per_channel (e.g., 0 2)")
    ap.add_argument("--use_mean", action="store_true",
                    help="Use mean_vaf instead of raw vaf as base metric")
    ap.add_argument("--group_mode", type=str, default="raw",
        choices=["raw","avg_folds","avg_channels","avg_channels_folds","avg_all"],
        help="What to average over before plotting")
    ap.add_argument("--do_violin", action="store_true",
                    help="Add per-day violin plots for the chosen decoder")
    ap.add_argument("--add_day0_from_crossval", action="store_true",
                    help="Inject crossval distribution at day 0 when missing (for align!=crossval)")
    ap.add_argument("--split_by_channel", action="store_true",
                    help="Generate a full figure set separately for each EMG channel")
    ap.add_argument("--series_by", type=str, default="decoder",
                choices=["decoder", "fold", "emg_channel"],
                help="Variable utilisée comme 'série' (couleur) pour les scatters/points.")
    ap.add_argument("--errorbars", action="store_true",
                help="Affiche des points 'moyenne ± écart-type' par jour et par série (au lieu des nuages bruts).")
    args = ap.parse_args()
    df = load_results(args.results_dir, recalc_day_from_date=args.recalc_day_from_date)
    one_decoder = args.decoder if args.decoder else None
    if args.violin_avg_muscles_grouped:
        pts = build_avg_muscles_points(
            df,
            exclude_channels=tuple(args.violin_exclude_channels),
            decoder=None,  # keep all decoders on same figure
            dim_red=(args.violin_dimred or None),
            align=(args.violin_align or None),
            per_day=True
        )
        title = "Avg across muscles"
        if args.violin_dimred: title += f" • {args.violin_dimred}"
        if args.violin_align:  title += f" • align={args.violin_align}"
        outdir = ensure_dir(os.path.join(args.out_dir, "violin_avg_muscles_grouped"))
        save = os.path.join(outdir, "violin_grouped.png")
        violin_avg_muscles_by_day_grouped(pts, title=title, save=save)

        if args.only_violin:
            return
    make_everything(
        df=df,
        outdir=args.out_dir,
        one_decoder=one_decoder,
        per_channel=args.per_channel,
        emg_channels=args.emg_channels,
        use_mean=args.use_mean,
        group_mode=args.group_mode,
        do_violin=args.do_violin,
        add_day0_from_crossval=args.add_day0_from_crossval,
        split_by_channel=args.split_by_channel,
        series_by=args.series_by,
        errorbars=args.errorbars
)

if __name__ == "__main__":
    main()
