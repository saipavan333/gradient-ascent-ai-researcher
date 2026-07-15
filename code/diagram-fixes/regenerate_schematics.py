"""Regenerate the hand-drawn SCHEMATIC diagrams for tracks 5, 6 and 14.

These five figures are laid out by hand (boxes + arrows), so they are the ones
prone to text/arrow overlap. This script rebuilds clean versions; the PNGs are
then quantized (128 colors) and embedded inline as base64 in the lessons:

  track-5/00-sequences-attention.html   fig0  <- tokenize()
  track-5/03-positional-encoding.html   fig0  <- positional()
  track-6/01-autoregressive.html        fig0  <- autoregressive()
  track-14/00-alignment-problem.html    fig0  <- alignment()
  track-14/01-goodhart-reward-hacking   fig0  <- goodhart()

Run: python code/diagram-fixes/regenerate_schematics.py  ->  ./rendered/*.png
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"rendered"); os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";GREEN="#059669";AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})
def box(ax,x,y,w,h,t,fc,ec,tc=INK,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(ax,x1,y1,x2,y2,color=SLATE,rad=0,lw=2,ms=14):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=ms,color=color,lw=lw,connectionstyle=f"arc3,rad={rad}"))

def alignment():
    fig,ax=plt.subplots(figsize=(10,5)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    ax.text(5,4.75,"Specification gaming: same agent, two reward specs",ha="center",fontsize=14,fontweight="bold",color=INK)
    xs=[0.4,2.25,4.1,5.95,7.8]; w=1.5; y=2.0; h=0.95
    lab=[("start","#f1f5f9",SLATE,False),("","#f8fafc","#e5e7eb",False),("coin\n+0.3/step","#fef9c3",AMBER,True),("","#f8fafc","#e5e7eb",False),("EXIT\n(true goal)","#dcfce7",GREEN,True)]
    for x,(t,fc,ec,b) in zip(xs,lab): box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=10,bold=b)
    sx=xs[0]+w/2; ex=xs[4]+w/2
    arrow(ax,sx,y+h,ex,y+h,color=GREEN,rad=-0.22,lw=2.6)
    ax.text(5.0,4.30,"TRUE reward (+1 exit): walks to the exit  ✓",ha="center",fontsize=12,fontweight="bold",color=GREEN)
    cx=xs[2]+w/2
    arrow(ax,sx,y,cx,y,color=RED,rad=0.30,lw=2.6)
    ax.add_patch(Ellipse((cx,1.30),0.42,0.24,fc=RED,ec="none"))
    ax.add_patch(FancyArrowPatch((cx+0.20,1.30),(cx+0.34,1.48),arrowstyle="-|>",mutation_scale=10,color=RED,lw=1.6,connectionstyle="arc3,rad=1.2"))
    ax.text(5.0,0.72,"PROXY reward (+0.3/step on coin): camps forever, never exits  ✗",ha="center",fontsize=12,fontweight="bold",color=RED)
    ax.text(5.0,0.26,"A capable optimizer maximizes whatever you actually reward — not what you meant.",ha="center",fontsize=10.5,style="italic",color=GREY)
    fig.savefig(os.path.join(OUT,"alignment.png"),bbox_inches="tight"); plt.close(fig)

def goodhart():
    x=np.linspace(0,1,240); proxy=1-np.exp(-3*x); true=np.maximum(0,3.1*x*(1-x)+0.02)
    fig,ax=plt.subplots(figsize=(8.6,4.6)); peak=x[np.argmax(true)]
    ax.axvspan(peak,1,color=RED,alpha=0.07)
    ax.plot(x,proxy,color=BLUE,lw=2.8,label="proxy score (what you measure)")
    ax.plot(x,true,color=GREEN,lw=2.8,label="TRUE value (what you want)")
    ax.axvline(peak,color=SLATE,ls=":",lw=1.5)
    ax.annotate("stop here",xy=(peak,float(np.max(true))),xytext=(0.15,1.02),fontsize=11,color=GREEN,fontweight="bold",
                arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=1.8,connectionstyle="arc3,rad=-0.15"))
    ax.text(0.8,-1.15,"over-optimization\n(reward hacking)",ha="center",fontsize=10,color=RED)
    ax.set_xlabel("optimization strength on the proxy"); ax.set_ylabel("reward")
    ax.set_title("Goodhart's law: the measure improves while true value collapses",fontsize=11.5,fontweight="bold",pad=12)
    ax.legend(loc="center left",fontsize=9.5,framealpha=0.95); ax.grid(alpha=0.2); ax.set_ylim(-1.6,1.35)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"goodhart.png"),bbox_inches="tight"); plt.close(fig)

def tokenize():
    rng=np.random.default_rng(3)
    fig,ax=plt.subplots(figsize=(9.2,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    ax.text(5,4.72,"Tokenization & embeddings: turn text into a sequence of vectors",ha="center",fontsize=12.5,fontweight="bold",color=INK)
    ax.text(0.5,4.16,'"The cat sat"',ha="left",fontsize=13,fontweight="bold",color=INK)
    for t,ec,ry in zip(["The","cat","sat"],[BLUE,GREEN,VIOLET],[3.25,2.35,1.45]):
        box(ax,0.5,ry-0.28,1.2,0.56,t,fc="#f8fafc",ec=ec,fs=11,bold=True); arrow(ax,1.8,ry,2.75,ry,color=GREY)
        for i,v in enumerate(rng.uniform(0,1,8)):
            ax.add_patch(Rectangle((2.95+i*0.34,ry-0.22),0.31,0.44,fc=cm.coolwarm(v),ec="white",lw=1))
        ax.text(2.95+8*0.34+0.15,ry,"embedding vector  (in ℝ$^{d}$)",va="center",fontsize=9.5,color=SLATE)
    ax.text(5,0.55,"each token → a learned vector (Track 1) — a row of a trainable embedding matrix",ha="center",fontsize=10,style="italic",color=GREY)
    fig.savefig(os.path.join(OUT,"tokenize.png"),bbox_inches="tight"); plt.close(fig)

def positional():
    T,d=40,64; pos=np.arange(T)[:,None]; i=np.arange(d)[None,:]
    ang=pos/np.power(10000,(2*(i//2))/d); PE=np.where(i%2==0,np.sin(ang),np.cos(ang))
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    im=ax[0].imshow(PE,aspect="auto",cmap="RdBu_r",vmin=-1,vmax=1)
    ax[0].set_xlabel("embedding dimension"); ax[0].set_ylabel("position in sequence")
    ax[0].set_title("Sinusoidal positional encoding (the stripe pattern)",fontsize=10.5,fontweight="bold")
    cb=fig.colorbar(im,ax=ax[0],fraction=0.046,pad=0.04); cb.set_label("value",fontsize=9)
    for dim,c in [(4,BLUE),(8,GREEN),(20,VIOLET),(40,AMBER)]: ax[1].plot(np.arange(T),PE[:,dim],color=c,lw=2,label=f"dim {dim}")
    ax[1].set_xlabel("position"); ax[1].set_ylabel("value")
    ax[1].set_title("Each dimension is a sine/cosine of a different frequency",fontsize=10.5,fontweight="bold")
    ax[1].legend(fontsize=8.5,ncol=2); ax[1].grid(alpha=0.25)
    fig.savefig(os.path.join(OUT,"positional.png"),bbox_inches="tight"); plt.close(fig)

def autoregressive():
    fig,ax=plt.subplots(figsize=(10,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    ax.text(5,4.7,"Autoregressive: factor the joint by the chain rule, generate left to right",ha="center",fontsize=12,fontweight="bold",color=INK)
    xs=[0.7,3.0,5.3,7.6]; w=1.4; y=3.05; h=0.7
    for k,xx in enumerate(xs): box(ax,xx,y,w,h,f"x$_{k+1}$",fc="#fffbeb",ec=AMBER,fs=13,bold=True)
    for k in range(3):
        x1=xs[k]+w; x2=xs[k+1]; arrow(ax,x1,y+h/2,x2,y+h/2,color=AMBER,lw=2.2)
        ax.text((x1+x2)/2,y+h/2+0.22,"sample",ha="center",fontsize=8.5,color=AMBER)
    for a in range(4):
        for b in range(a+1,4):
            if b-a==1: continue
            xa=xs[a]+w/2; xb=xs[b]+w/2; rad={2:0.16,3:0.24}[b-a]
            ax.add_patch(FancyArrowPatch((xa,y),(xb,y),arrowstyle="-",color=GREY,lw=1.2,alpha=0.75,connectionstyle=f"arc3,rad={rad}"))
    ax.text(5,1.55,"p(x$_1$,x$_2$,x$_3$,x$_4$) = p(x$_1$) p(x$_2$|x$_1$) p(x$_3$|x$_1$,x$_2$) p(x$_4$|x$_1$,x$_2$,x$_3$)",ha="center",fontsize=12,color=INK)
    ax.text(5,0.5,"Grey arcs: each variable conditions on all earlier ones.   Amber arrows: ancestral sampling, one at a time.",ha="center",fontsize=9.5,style="italic",color=GREY)
    fig.savefig(os.path.join(OUT,"autoregressive.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    for f in [alignment,goodhart,tokenize,positional,autoregressive]: f(); print("rendered",f.__name__)
