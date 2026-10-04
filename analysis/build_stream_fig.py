"""Redraw the streaming schematic (analysis/figures/fig_stream.pdf); no measurement data are read.

    python3 analysis/build_stream_fig.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42})
fig,ax=plt.subplots(figsize=(3.7,5.05))
fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,3.7),ylim=(0,5.05));ax.axis("off")
edge="#354454"; control="#286b96"
def box(x,y,w,h,text,fill,fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.035",linewidth=.8,edgecolor=edge,facecolor=fill,zorder=3))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,linespacing=1.35,zorder=4)
def arrow(points,dash=False):
    codes=[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)
    ax.add_patch(FancyArrowPatch(path=MPath(points,codes),arrowstyle="-|>",mutation_scale=9,linewidth=1.05,linestyle=(0,(3,2)) if dash else "-",color=control if dash else edge,zorder=2))
x=.49; w=2.23; cx=x+w/2
box(x,4.32,w,.53,"GH200 generator GPU\nSynthesize tiles; batch sends","#e4ecf7")
box(x,3.44,w,.48,"400 GbE link\n4032 payload bytes / frame","#fff0d6")
box(x,2.37,w,.67,"RTX PRO 6000 GPU\nRX buffer and receive warp\nWorkers: XOR / popcount","#e1f1e9")
box(x,1.22,w,.72,"Row accumulators\nEvaluate energy; update state*\nRefresh resident signs","#e1f1e9",7.65)
box(x,.34,w,.48,"GPU sends PASS_ACK\nRelease next matrix pass","#eef3f7",7.8)
for a,b in [(4.32,3.92),(3.44,3.04),(2.37,1.94),(1.22,.82)]:arrow([(cx,a),(cx,b)])
box(2.99,1.56,.66,1.23,"Host\npass\ncontrol","#e4ecf7",7.6)
arrow([(2.72,2.93),(3.32,2.93),(3.32,2.79)],True)
arrow([(2.99,1.72),(2.72,1.72)],True)
arrow([(.49,.58),(.17,.58),(.17,4.59),(.49,4.59)],True)
ax.text(.055,2.60,"Acknowledgement / next pass",rotation=90,ha="center",va="center",color=control,fontsize=7)
ax.text(1.97,2.16,"coupling sums",fontsize=6.8,ha="left",va="center",color=edge)
ax.text(1.93,1.025,"pass complete",fontsize=6.8,ha="left",va="center",color=edge)
ax.text(1.98,.10,"*Final pass: energy evaluation only; no state update.",fontsize=6.8,ha="center",va="center",color=edge)
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "fig_stream.pdf")
