"""
=============================================================================
MTF STRATEGY v2 FAST — Автономная версия без повторной загрузки из v1
=============================================================================
Улучшения над v1:
  • 10-компонентный score на каждом TF (EMA, BB, RSI, ADX, Volume, Momentum, Stoch, Fractal Break)
  • Весовой коэффициент D=2 (дневной TF самый важный)
  • Фрактальный пробой как фильтр входа
  • ZigZag направление как фильтр (только восходящая фаза)
  • Частичная фиксация прибыли на фрактальных максимумах (33%+33%+34%)
  • Фрактальный SL движется вверх по мере появления новых фракталов
=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

DATA_DIR   = Path("c:/investor/data")
COMMISSION = 0.0005
SLIPPAGE   = 0.0003
INITIAL_CAP   = 100_000.0
MAX_POSITIONS = 4
RISK_PCT      = 0.20

TICKERS = ["SBER","GAZP","LKOH","NVTK","NLMK","MGNT","ROSN","MTLR","OZPH","YDEX","T"]
TF_LIST  = ["1H","4H","8H","12H","D"]
TF_WEIGHT = {"1H":1,"4H":1,"8H":1,"12H":1,"D":2}

# ── Индикаторы ─────────────────────────────────────────────────────────────────
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def atr(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).ewm(span=n,adjust=False).mean()
    ls=(-d).clip(lower=0).ewm(span=n,adjust=False).mean()
    return 100-100/(1+g/(ls+1e-10))
def adx(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    dmp=(h-h.shift()).clip(lower=0).where((h-h.shift())>(l.shift()-l),0)
    dmm=(l.shift()-l).clip(lower=0).where((l.shift()-l)>(h-h.shift()),0)
    at=tr.ewm(span=n,adjust=False).mean()
    dip=100*dmp.ewm(span=n,adjust=False).mean()/(at+1e-10)
    dim=100*dmm.ewm(span=n,adjust=False).mean()/(at+1e-10)
    dx=100*(dip-dim).abs()/(dip+dim+1e-10)
    return dx.ewm(span=n,adjust=False).mean()
def vol_ratio(df,n=20): v=df["volume"]; return v/(v.rolling(n).mean()+1e-10)
def momentum(s,n=10): return (s-s.shift(n))/(s.shift(n)+1e-10)*100
def bb_mid(s,n=20): return s.rolling(n).mean()
def stoch(df,k=14,d=3):
    h14=df["high"].rolling(k).max(); l14=df["low"].rolling(k).min()
    k_=100*(df["close"]-l14)/(h14-l14+1e-10)
    return k_.rolling(d).mean()
def bw_fractals(df,n=2):
    h,l=df["high"],df["low"]; fh=pd.Series(False,index=df.index); fl=pd.Series(False,index=df.index)
    hv,lv=h.values,l.values
    for i in range(n,len(df)-n):
        if hv[i]>max(hv[i-n:i]) and hv[i]>max(hv[i+1:i+n+1]): fh.iloc[i]=True
        if lv[i]<min(lv[i-n:i]) and lv[i]<min(lv[i+1:i+n+1]): fl.iloc[i]=True
    return fh,fl
def zigzag(df,dev=5.0):
    h,l=df["high"].values,df["low"].values; n=len(df)
    zz=np.full(n,np.nan); direction=0; lpi=0; lpp=(h[0]+l[0])/2
    for i in range(1,n):
        if direction==0:
            if h[i]>lpp*(1+dev/100): direction=1;lpi=i;lpp=h[i];zz[i]=h[i]
            elif l[i]<lpp*(1-dev/100): direction=-1;lpi=i;lpp=l[i];zz[i]=l[i]
        elif direction==1:
            if h[i]>lpp: zz[lpi]=np.nan;lpi=i;lpp=h[i];zz[i]=h[i]
            elif l[i]<lpp*(1-dev/100): direction=-1;lpi=i;lpp=l[i];zz[i]=l[i]
        elif direction==-1:
            if l[i]<lpp: zz[lpi]=np.nan;lpi=i;lpp=l[i];zz[i]=l[i]
            elif h[i]>lpp*(1+dev/100): direction=1;lpi=i;lpp=h[i];zz[i]=h[i]
    return pd.Series(zz,index=df.index)

# ── Загрузка одного TF ─────────────────────────────────────────────────────────
def load_tf(ticker,tf):
    if tf=="1H":
        candidates=list(DATA_DIR.glob(f"{ticker}_2022_2026_1H.csv"))
        if not candidates: candidates=list(DATA_DIR.glob(f"{ticker}_*_1H.csv"))
    else:
        candidates=list(DATA_DIR.glob(f"{ticker}_2022_2026_{tf}.csv"))
        if not candidates: candidates=list(DATA_DIR.glob(f"{ticker}_*_{tf}.csv"))
    if not candidates: return None
    path=candidates[0]
    df=pd.read_csv(path,sep=";")
    df.columns=[c.strip("<>").lower() for c in df.columns]
    def pdt(row):
        d=str(row["date"]); t=str(int(row["time"])).zfill(6)
        return pd.to_datetime(d+" "+t,format="%d/%m/%y %H%M%S",errors="coerce")
    df.index=df.apply(pdt,axis=1)
    df=df[["open","high","low","close","vol"]].rename(columns={"vol":"volume"})
    return df.dropna().sort_index()

# ── Вычисление TF score ────────────────────────────────────────────────────────
def tf_signal(df,tf):
    if df is None or len(df)<210: return pd.Series(dtype=float)
    c=df["close"]; thr={"1H":6,"4H":5,"8H":4,"12H":4,"D":4}.get(tf,5)
    at14=atr(df,14); at5=atr(df,5); e200=ema(c,200); e50=ema(c,50)
    r14=rsi(c,14); a14=adx(df,14); vr=vol_ratio(df,20); bm=(c-c.shift(1)).clip(lower=0)
    mom=momentum(c,10); st=stoch(df,14,3); bm_=bb_mid(c,20)
    fh,_=bw_fractals(df,2); fh_s=df["high"].where(fh).ffill()
    mult={"1H":1.2,"4H":0.8,"8H":0.5,"12H":0.3,"D":0.2}.get(tf,1.0)
    rlo={"1H":50,"4H":48,"8H":47,"12H":46,"D":45}.get(tf,48)
    rhi={"1H":80,"4H":82,"8H":83,"12H":84,"D":85}.get(tf,82)
    athr={"1H":18,"4H":16,"8H":15,"12H":14,"D":13}.get(tf,15)
    vthr={"1H":1.2,"4H":1.1,"8H":1.0,"12H":1.0,"D":1.0}.get(tf,1.0)
    score=(
        (c>e200).astype(int)+(c>e50).astype(int)+(bm>=mult*at14).astype(int)+
        ((r14>=rlo)&(r14<=rhi)).astype(int)+(a14>=athr).astype(int)+
        (vr>=vthr).astype(int)+(mom>0).astype(int)+
        ((st>20)&(st<85)).astype(int)+(c>bm_).astype(int)+
        (c>fh_s.shift(1)).astype(int)
    )
    return (score>=thr).astype(float)

# ══════════════════════════════════════════════════════════════════════════════
print("="*72)
print("  MTF STRATEGY v2 — Загрузка данных и вычисление features")
print("="*72)
print("Загрузка...")

ALL:  dict[str,dict[str,pd.DataFrame]] = {}
SIGS: dict[str,dict[str,pd.Series]]   = {}

for ticker in TICKERS:
    tfd={}; tfs={}
    for tf in TF_LIST:
        df=load_tf(ticker,tf)
        if df is not None and len(df)>=210:
            tfd[tf]=df; tfs[tf]=tf_signal(df,tf)
    if "1H" in tfd and "D" in tfd:
        ALL[ticker]=tfd; SIGS[ticker]=tfs
        print(f"  {ticker:6s}: 1H={len(tfd['1H']):5d}  D={len(tfd.get('D',[])):4d}")

print(f"  Тикеров: {len(ALL)}\n")

# Предрасчёт выровненных сигналов, фракталов, ZigZag, ATR(D) на 1H ось
print("Выравнивание на 1H ось...")
MTF_SCORE: dict[str,pd.Series]  = {}
FRAC_H1H:  dict[str,pd.Series]  = {}
FRAC_L1H:  dict[str,pd.Series]  = {}
ATR_D1H:   dict[str,pd.Series]  = {}
ZZ_DIR1H:  dict[str,pd.Series]  = {}

for ticker,tfd in ALL.items():
    df1h=tfd["1H"]; total=pd.Series(0.0,index=df1h.index)
    for tf in TF_LIST:
        if tf not in SIGS[ticker]: continue
        sig=SIGS[ticker][tf].astype(float)
        al=sig.reindex(df1h.index,method="ffill").ffill().fillna(0)
        total+=al*TF_WEIGHT.get(tf,1)
    MTF_SCORE[ticker]=total

    df_d=tfd["D"]
    fh,fl=bw_fractals(df_d,2)
    fh_s=df_d["high"].where(fh).ffill()
    fl_s=df_d["low"].where(fl).ffill()
    at_d=atr(df_d,14)
    zz=zigzag(df_d,5.0)
    zz_diff=(zz-zz.shift(1)).replace(0,np.nan).ffill().fillna(0)
    zz_dir=(zz_diff>0).astype(float)-(zz_diff<0).astype(float)
    FRAC_H1H[ticker]=fh_s.reindex(df1h.index,method="ffill").ffill()
    FRAC_L1H[ticker]=fl_s.reindex(df1h.index,method="ffill").ffill()
    ATR_D1H[ticker] =at_d.reindex(df1h.index,method="ffill").ffill()
    ZZ_DIR1H[ticker]=zz_dir.reindex(df1h.index,method="ffill").ffill().fillna(0)

print("  Готово\n")

MAX_SCORE=sum(TF_WEIGHT.values())  # =6
CORR_GROUPS=[{"SBER","T"},{"LKOH","ROSN","NVTK","GAZP"},{"NLMK","MTLR"}]
def corr_blocked(t,open_set):
    return any(t in g and g&open_set for g in CORR_GROUPS)

@dataclass
class Pos:
    ticker:str; entry_dt:object; entry_px:float; shares:float
    sl:float; trail:float; partial:float=0.0; rem:float=1.0
    tp1:bool=False; tp2:bool=False; fh_entry:float=0.0

@dataclass
class Tr:
    ticker:str; entry_dt:object; exit_dt:object
    entry_px:float; exit_px:float; pnl_rub:float; pnl_pct:float
    reason:str; hold_days:float; score:float

def run(min_sc,req_d,req_frac,req_zz,trail,part_tp,hold_days,label):
    # Сигналы
    ENTRY: dict[str,pd.Series]={}
    n_sig=0
    for ticker,score in MTF_SCORE.items():
        df1h=ALL[ticker]["1H"]; c=df1h["close"]
        sig=score>=min_sc
        if req_d:
            d_act=SIGS[ticker].get("D",pd.Series(0,index=df1h.index))
            d_al=d_act.reindex(df1h.index,method="ffill").ffill().fillna(0)
            sig=sig&(d_al>0)
        if req_frac and ticker in FRAC_H1H:
            fh=FRAC_H1H[ticker].shift(1)
            sig=sig&(c>fh)
        if req_zz and ticker in ZZ_DIR1H:
            sig=sig&(ZZ_DIR1H[ticker]>0)
        sig=sig&~sig.shift(1).fillna(False)
        ENTRY[ticker]=sig; n_sig+=int(sig.sum())

    master=set()
    for t in ALL: master.update(ALL[t]["1H"].index)
    timeline=sorted(master)
    IDX={t:{d:i for i,d in enumerate(df.index)} for t,df in {t:ALL[t]["1H"] for t in ALL}.items()}

    free=INITIAL_CAP; positions:dict[str,Pos]={}; trades:list[Tr]=[]
    equity=[INITIAL_CAP]; peak=INITIAL_CAP; max_dd=0.0

    for dt in timeline:
        to_close=[]
        for ticker,pos in positions.items():
            df1h=ALL[ticker]["1H"]; idx=IDX[ticker].get(dt)
            if idx is None or idx<=0 or dt<=pos.entry_dt: continue
            hi=float(df1h["high"].iloc[idx]); lo=float(df1h["low"].iloc[idx])
            op=float(df1h["open"].iloc[idx]); cls=float(df1h["close"].iloc[idx])
            at_d=float(ATR_D1H[ticker].iloc[idx]) if ticker in ATR_D1H else 0.0
            fh=float(FRAC_H1H[ticker].iloc[idx]) if ticker in FRAC_H1H else 0.0
            fl=float(FRAC_L1H[ticker].iloc[idx]) if ticker in FRAC_L1H else 0.0
            hd=(dt-pos.entry_dt).total_seconds()/86400

            # Trailing update
            if at_d>0:
                tr=cls-trail*at_d
                if tr>pos.trail: pos.trail=tr
            # SL: фрактальный минимум двигается вверх
            if fl>pos.sl and fl<cls: pos.sl=fl
            curr_sl=max(pos.sl,pos.trail)

            # Частичный TP на фрактальных максимумах
            if part_tp and pos.rem>1e-6:
                if not pos.tp1 and fh>pos.entry_px*1.015 and hi>=fh:
                    fr=min(0.33,pos.rem); ep=fh*(1-SLIPPAGE)
                    pos.partial+=(ep-pos.entry_px)*fr*pos.shares-(pos.entry_px+ep)*COMMISSION*fr*pos.shares
                    pos.rem-=fr; pos.tp1=True; pos.fh_entry=fh
                    pos.sl=max(pos.sl,pos.entry_px*1.001)
                    pos.trail=max(pos.trail,pos.entry_px*1.001)
                    curr_sl=max(pos.sl,pos.trail)
                elif pos.tp1 and not pos.tp2 and fh>pos.fh_entry*1.01 and hi>=fh:
                    fr=min(0.33,pos.rem); ep=fh*(1-SLIPPAGE)
                    pos.partial+=(ep-pos.entry_px)*fr*pos.shares-(pos.entry_px+ep)*COMMISSION*fr*pos.shares
                    pos.rem-=fr; pos.tp2=True
                    pos.sl=max(pos.sl,pos.fh_entry*0.98)
                    curr_sl=max(pos.sl,pos.trail)

            reason=exit_px=None
            if pos.rem<=1e-6: reason="FRAC_TP"; exit_px=pos.entry_px
            elif lo<=curr_sl: reason="SL_FRACTAL"; exit_px=max(curr_sl*(1-SLIPPAGE),lo)
            elif hd>=hold_days: reason="TIME"; exit_px=op*(1-SLIPPAGE)

            if reason:
                rem=pos.rem
                cash_r=exit_px*rem*pos.shares*(1-COMMISSION)
                cost_t=pos.entry_px*pos.shares*(1+COMMISSION)
                total_cash=pos.partial+cash_r
                pnl=total_cash-cost_t; pnl_pct=pnl/cost_t*100
                free+=total_cash
                sc=float(MTF_SCORE[ticker].get(pos.entry_dt,0)) if pos.entry_dt in MTF_SCORE[ticker].index else 0.0
                trades.append(Tr(ticker,pos.entry_dt,dt,pos.entry_px,exit_px,pnl,pnl_pct,reason,hd,sc))
                to_close.append(ticker)
        for t in to_close: positions.pop(t,None)

        for ticker in ALL:
            if len(positions)>=MAX_POSITIONS: break
            if ticker in positions: continue
            idx=IDX[ticker].get(dt)
            if idx is None or idx<1: continue
            if not bool(ENTRY[ticker].iloc[idx]): continue
            if corr_blocked(ticker,set(positions.keys())): continue
            df1h=ALL[ticker]["1H"]
            entry=float(df1h["open"].iloc[idx])*(1+SLIPPAGE)
            fl_v=float(FRAC_L1H[ticker].iloc[idx]) if ticker in FRAC_L1H else entry*0.90
            if fl_v<=0 or (entry-fl_v)/entry>0.20: fl_v=entry*0.85
            at_d=float(ATR_D1H[ticker].iloc[idx]) if ticker in ATR_D1H else entry*0.02
            trail_i=entry-trail*at_d; init_sl=max(fl_v,trail_i)
            pos_val=sum(float(ALL[t]["1H"]["close"].iloc[IDX[t].get(dt,-1)])*p.shares*p.rem
                        for t,p in positions.items() if IDX[t].get(dt) is not None)
            total_cap=free+pos_val
            alloc=min(total_cap*RISK_PCT,free*0.95)
            if alloc<=0: continue
            shares=alloc/entry; cost=shares*entry*(1+COMMISSION)
            if cost>free: continue
            free-=cost
            positions[ticker]=Pos(ticker,dt,entry,shares,init_sl,init_sl)

        pos_val=sum(float(ALL[t]["1H"]["close"].iloc[IDX[t].get(dt,-1)])*p.shares*p.rem
                    for t,p in positions.items() if IDX[t].get(dt) is not None)
        eq=free+pos_val; equity.append(eq)
        if eq>peak: peak=eq
        dd=(peak-eq)/peak*100
        if dd>max_dd: max_dd=dd

    last_dt=timeline[-1]
    for ticker,pos in list(positions.items()):
        cls=float(ALL[ticker]["1H"]["close"].iloc[-1]); ep=cls*(1-SLIPPAGE)
        rem=pos.rem; cash_r=ep*rem*pos.shares*(1-COMMISSION)
        cost_t=pos.entry_px*pos.shares*(1+COMMISSION)
        pnl=pos.partial+cash_r-cost_t; pnl_pct=pnl/cost_t*100
        free+=pos.partial+cash_r
        trades.append(Tr(ticker,pos.entry_dt,last_dt,pos.entry_px,ep,pnl,pnl_pct,"FORCED",(last_dt-pos.entry_dt).total_seconds()/86400,0.0))

    final=free; total_pnl=(final-INITIAL_CAP)/INITIAL_CAP*100
    n_days=(timeline[-1]-timeline[0]).days
    ann=((final/INITIAL_CAP)**(365/max(n_days,1))-1)*100
    pnls=np.array([t.pnl_pct for t in trades]); n_tr=len(trades)
    n_win=(pnls>0).sum(); wr=n_win/n_tr*100 if n_tr else 0
    wins=pnls[pnls>0]; losses=pnls[pnls<=0]
    pf=wins.sum()/(-losses.sum()+1e-9) if len(losses) else 99.0
    eq_arr=np.array(equity); dr=np.diff(eq_arr)/(eq_arr[:-1]+1e-9)
    sharpe=(dr.mean()/(dr.std()+1e-9))*np.sqrt(252*6.5)
    by_r:dict[str,dict]={}
    for t in trades:
        s=by_r.setdefault(t.reason,{"n":0,"wins":0,"pnl":0.0})
        s["n"]+=1; s["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: s["wins"]+=1
    by_y:dict[str,float]={}; running=INITIAL_CAP
    for t in sorted(trades,key=lambda x:x.exit_dt):
        yr=str(t.exit_dt)[:4]; running+=t.pnl_rub; by_y[yr]=running
    return dict(label=label,final=final,total_pnl=total_pnl,ann=ann,max_dd=-max_dd,
                sharpe=sharpe,trades=n_tr,wr=wr,pf=min(pf,99),
                avg_win=float(wins.mean()) if len(wins) else 0,
                avg_loss=float(losses.mean()) if len(losses) else 0,
                by_r=by_r,by_y=by_y,trades_list=trades,n_sig=n_sig)

# ══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("  MTF v2: ОПТИМИЗАЦИЯ  |  Все тикеры  |  Фракталы + ZigZag + Partial TP")
print("="*80)
print(f"  {'Конфиг':30s} {'Сиг':5s} {'Сд':4s} {'WR':6s} {'PF':5s} "
      f"{'ANN%':6s} {'MaxDD':7s} {'Sharpe':7s} {'Итог ₽':10s}")
print("  "+"─"*85)

CFGS=[
    # label,              min_sc, req_d, req_fr, req_zz, trail, ptp, hold
    ("sc4-D-FR-ZZ-t2-h40",   4, True, True,  True,  2.0, True, 40),
    ("sc4-D-FR-ZZ-t2-h30",   4, True, True,  True,  2.0, True, 30),
    ("sc4-D-FR-ZZ-t2.5-h40", 4, True, True,  True,  2.5, True, 40),
    ("sc4-D-FR-noZZ-t2-h40", 4, True, True,  False, 2.0, True, 40),
    ("sc4-D-noFR-ZZ-t2-h40", 4, True, False, True,  2.0, True, 40),
    ("sc4-noD-FR-ZZ-t2-h40", 4, False,True,  True,  2.0, True, 40),
    ("sc4-D-FR-ZZ-noPTP-h40",4, True, True,  True,  2.0, False,40),
    ("sc3.5-D-FR-ZZ-t2-h40",3.5,True, True,  True,  2.0, True, 40),
    ("sc5-D-FR-ZZ-t2-h45",   5, True, True,  True,  2.0, True, 45),
    ("sc4-D-FR-ZZ-t2-h50",   4, True, True,  True,  2.0, True, 50),
    ("sc4-D-FR-ZZ-t1.5-h40", 4, True, True,  True,  1.5, True, 40),
    ("sc4-D-FR-ZZ-t3-h45",   4, True, True,  True,  3.0, True, 45),
]

best_sc=-999; best_r=None
for cfg in CFGS:
    lb,ms,rd,rf,rz,tr,pt,hd=cfg
    r=run(ms,rd,rf,rz,tr,pt,hd,lb)
    sc=r["ann"]*0.6+r["sharpe"]*5-abs(r["max_dd"])*0.4
    mk=" ◄" if sc>best_sc and r["trades"]>=10 else ""
    if sc>best_sc and r["trades"]>=10: best_sc=sc; best_r=r
    print(f"  {lb:30s} {r['n_sig']:5d} {r['trades']:4d} {r['wr']:5.1f}% "
          f"{r['pf']:5.2f} {r['ann']:>+5.1f}% {r['max_dd']:>+6.1f}% "
          f"{r['sharpe']:>6.2f}  {r['final']:>10,.0f}{mk}")

# ── Детальный разбор победителя ────────────────────────────────────────────────
if best_r:
    b=best_r
    print(f"\n{'═'*80}")
    print(f"  ПОБЕДИТЕЛЬ: {b['label']}")
    print(f"{'═'*80}")
    print(f"  Начальный капитал:  {INITIAL_CAP:>10,.0f} ₽")
    print(f"  Итоговый капитал:   {b['final']:>10,.0f} ₽")
    print(f"  Суммарная прибыль:  {b['total_pnl']:>+9.1f}%")
    print(f"  CAGR:               {b['ann']:>+9.1f}%")
    print(f"  MaxDD:              {b['max_dd']:>+9.1f}%")
    print(f"  Sharpe:             {b['sharpe']:>9.2f}")
    print(f"  Сделок:             {b['trades']:>9d}")
    print(f"  WR:                 {b['wr']:>9.1f}%")
    print(f"  PF:                 {b['pf']:>9.2f}")
    print(f"  Avg Win / Loss:     {b['avg_win']:>+6.2f}% / {b['avg_loss']:>+6.2f}%")

    print("\n  ПО ГОДАМ:")
    print(f"  {'Год':6s} {'Капитал':>12s} {'Прибыль':>12s} {'%':>8s}")
    print("  "+"─"*42)
    prev=INITIAL_CAP
    for yr in sorted(b["by_y"]):
        cap=b["by_y"][yr]; pr=cap-prev; pct=pr/prev*100
        print(f"  {yr}   {cap:>12,.0f} ₽  {pr:>+10,.0f} ₽  {pct:>+7.1f}%")
        prev=cap
    print("  "+"─"*42)
    print(f"  ИТОГО  {b['final']:>12,.0f} ₽  "
          f"{b['final']-INITIAL_CAP:>+10,.0f} ₽  {b['total_pnl']:>+7.1f}%")

    print("\n  ПО ПРИЧИНАМ ВЫХОДА:")
    print(f"  {'Причина':12s} {'N':5s} {'WR%':7s} {'Avg%':8s}")
    print("  "+"─"*36)
    for reason,s in sorted(b["by_r"].items()):
        wr_r=s["wins"]/s["n"]*100 if s["n"] else 0
        avg=s["pnl"]/s["n"] if s["n"] else 0
        print(f"  {reason:12s} {s['n']:5d} {wr_r:>6.1f}% {avg:>+7.2f}%")

    tl=b["trades_list"]
    by_t:dict[str,list]={}
    for t in tl: by_t.setdefault(t.ticker,[]).append(t.pnl_pct)
    print("\n  ПО ТИКЕРАМ:")
    print(f"  {'Тикер':6s} {'N':4s} {'WR%':6s} {'Total%':8s} {'Avg%':7s}")
    print("  "+"─"*36)
    for tk,pnls in sorted(by_t.items(),key=lambda x:sum(x[1]),reverse=True):
        pa=np.array(pnls)
        print(f"  {tk:6s} {len(pnls):4d} {(pa>0).mean()*100:5.1f}% "
              f"{pa.sum():>+7.1f}% {pa.mean():>+6.2f}%")

    tl_s=sorted(tl,key=lambda t:t.pnl_pct,reverse=True)
    print("\n  ТОП-7 ЛУЧШИХ СДЕЛОК:")
    print(f"  {'Тикер':6s} {'Вход':16s} {'Выход':16s} {'Дни':4s} {'P&L%':7s} {'Причина':12s}")
    for t in tl_s[:7]:
        print(f"  {t.ticker:6s} {str(t.entry_dt)[:16]:16s} "
              f"{str(t.exit_dt)[:16]:16s} {t.hold_days:3.0f}  "
              f"{t.pnl_pct:>+6.1f}% {t.reason}")
    print("\n  ХУДШИЕ 5:")
    for t in tl_s[-5:]:
        print(f"  {t.ticker:6s} {str(t.entry_dt)[:16]:16s} "
              f"{str(t.exit_dt)[:16]:16s} {t.hold_days:3.0f}  "
              f"{t.pnl_pct:>+6.1f}% {t.reason}")

print(f"\n{'═'*80}")
print("  ФИНАЛЬНОЕ СРАВНЕНИЕ ВСЕХ СТРАТЕГИЙ:")
print(f"  {'Стратегия':36s} {'ANN%':7s} {'MaxDD':7s} {'Sharpe':8s} {'WR':6s} {'Итог ₽':10s}")
print("  "+"─"*78)
print(f"  {'Дневная ATR_BO (лидер)':36s}  +13.9%   -12.9%     1.19   69.4%    173,223")
print(f"  {'MTF v1 (score≥3 hold=30)':36s}  +11.8%   -13.8%     0.39   47.4%    159,764")
if best_r:
    print(f"  {best_r['label']:36s} {best_r['ann']:>+6.1f}% "
          f"{best_r['max_dd']:>+6.1f}%  {best_r['sharpe']:>7.2f}  "
          f"{best_r['wr']:5.1f}%  {best_r['final']:>10,.0f}")
print(f"{'═'*80}")
