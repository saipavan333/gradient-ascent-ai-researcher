"""Generate Track 19 (Efficient & Scalable ML) diagrams -> assets/img/track-23/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.abspath(os.path.join(HERE,"..","..","assets","img","track-23"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";LIGHT="#eff6ff";GREEN="#059669";AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})
rng=np.random.default_rng(0)
def box(ax,x,y,w,h,t,fc=LIGHT,ec=BLUE,tc=INK,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(ax,x1,y1,x2,y2,color=SLATE,rad=0,lw=2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=color,lw=lw,connectionstyle=f"arc3,rad={rad}"))

# 1) distillation.png
def distillation():
    fig,ax=plt.subplots(figsize=(10,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,4.2); ax.axis("off")
    ax.text(5,4.0,"Knowledge distillation: a small student imitates a big teacher's soft predictions",ha="center",fontsize=11.5,fontweight="bold",color=INK)
    box(ax,0.5,1.6,2.2,1.6,"TEACHER\n(big, accurate,\nexpensive)",fc="#dbeafe",ec=BLUE,bold=True,fs=10)
    box(ax,7.3,1.7,2.2,1.4,"STUDENT\n(small, fast,\ncheap)",fc="#dcfce7",ec=GREEN,bold=True,fs=10)
    # soft targets
    ax.text(4.9,3.15,"soft targets  [0.80, 0.15, 0.05]",ha="center",fontsize=9.5,color=VIOLET,fontweight="bold")
    ax.text(4.9,2.75,"\"mostly cat, a bit dog\" — rich info",ha="center",fontsize=8.5,color=SLATE)
    ax.text(4.9,1.85,"hard label  [1, 0, 0]",ha="center",fontsize=9.5,color=GREY)
    ax.text(4.9,1.5,"\"cat\" — one bit of info",ha="center",fontsize=8.5,color=GREY)
    arrow(ax,2.7,2.7,7.3,2.6,color=VIOLET,lw=2.5)
    ax.text(5,0.7,"the student learns class relationships the hard label throws away  →  higher accuracy (0.909 → 0.920)",ha="center",fontsize=9.5,color=INK)
    fig.savefig(os.path.join(OUT,"distillation.png"),bbox_inches="tight"); plt.close(fig)

# 2) pruning.png (real curve)
def pruning():
    def data(n): C=np.array([[2,2],[-2,2],[0,-2.2]],float); X=np.vstack([c+rng.standard_normal((n//3,2)) for c in C]); return X,np.repeat(range(3),n//3)
    def relu(x): return np.maximum(0,x)
    def sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
    Xtr,ytr=data(900);Xte,yte=data(3000)
    W1=rng.standard_normal((2,64))*0.4;b1=np.zeros(64);W2=rng.standard_normal((64,3))*0.4;b2=np.zeros(3)
    for _ in range(400):
        h=relu(Xtr@W1+b1);P=sm(h@W2+b2);G=(P-np.eye(3)[ytr])/len(Xtr)
        gW2=h.T@G;dh=(G@W2.T)*(h>0);gW1=Xtr.T@dh
        W1-=0.2*gW1;W2-=0.2*gW2;b1-=0.2*dh.sum(0);b2-=0.2*G.sum(0)
    fracs=np.linspace(0,0.95,20); accs=[]
    for f in fracs:
        t1=np.quantile(np.abs(W1),f); t2=np.quantile(np.abs(W2),f)
        Wp1=W1*(np.abs(W1)>=t1);Wp2=W2*(np.abs(W2)>=t2)
        h=relu(Xte@Wp1+b1);P=sm(h@Wp2+b2); accs.append((P.argmax(1)==yte).mean())
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    allw=np.abs(np.concatenate([W1.ravel(),W2.ravel()]))
    ax[0].hist(allw,bins=40,color="#bfdbfe",edgecolor="#93c5fd")
    ax[0].axvline(np.quantile(allw,0.7),color=RED,lw=2,ls="--",label="prune below this (70%)")
    ax[0].set_title("Most weights are tiny",fontsize=10.5,fontweight="bold"); ax[0].set_xlabel("|weight|"); ax[0].legend(fontsize=8.5)
    ax[1].plot(fracs*100,accs,color=BLUE,lw=2.6,marker="o",ms=3)
    ax[1].axhline(accs[0],color=GREY,ls=":"); ax[1].set_xlabel("% of weights pruned (sparsity)"); ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy holds until very high sparsity",fontsize=10.5,fontweight="bold"); ax[1].grid(alpha=0.25)
    fig.suptitle("Pruning: delete the smallest weights — trained networks are hugely redundant",fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"pruning.png"),bbox_inches="tight"); plt.close(fig)

# 3) quantization.png
def quantization():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    w=rng.standard_normal(4000)*0.4
    lo,hi=w.min(),w.max(); L=15
    wq=np.round((w-lo)/(hi-lo)*L)/L*(hi-lo)+lo
    ax[0].hist(w,bins=60,color="#bfdbfe",alpha=0.6,label="32-bit (continuous)")
    ax[0].hist(wq,bins=60,color=RED,alpha=0.6,label="4-bit (16 levels)")
    ax[0].set_title("Weights snapped to a few levels",fontsize=10.5,fontweight="bold"); ax[0].set_xlabel("weight value"); ax[0].legend(fontsize=8.5)
    bits=[32,8,4,2]; acc=[0.974,0.974,0.972,0.950]; mem=[32,8,4,2]
    x=np.arange(4); ax[1].bar(x-0.2,acc,0.4,color=GREEN,label="accuracy",edgecolor="black")
    ax2=ax[1].twinx(); ax2.bar(x+0.2,mem,0.4,color=AMBER,label="bits/weight",edgecolor="black")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["32-bit","8-bit","4-bit","2-bit"]); ax[1].set_ylim(0.9,1.0); ax[1].set_ylabel("accuracy",color=GREEN)
    ax2.set_ylabel("bits per weight (memory)",color=AMBER); ax[1].set_title("Tiny accuracy loss, big memory win",fontsize=10.5,fontweight="bold")
    fig.suptitle("Quantization: fewer bits per weight — 8-bit matches 32-bit at 1/4 the memory",fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"quantization.png"),bbox_inches="tight"); plt.close(fig)

# 4) parallelism.png
def parallelism():
    fig,ax=plt.subplots(figsize=(10,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    ax.text(5,4.75,"Data parallelism: split the batch across workers, average their gradients",ha="center",fontsize=11.5,fontweight="bold",color=INK)
    box(ax,3.6,3.9,2.8,0.7,"big batch of data",fc="#f1f5f9",ec=GREY,fs=9)
    xs=[0.6,3.0,5.4,7.8]
    for i,x in enumerate(xs):
        box(ax,x,1.8,1.9,1.3,"Worker "+str(i+1)+"\ndata shard\n→ gradient",fc="#dbeafe",ec=BLUE,fs=8.5)
        arrow(ax,x+0.95,3.9,x+0.95,3.1,color=SLATE)
        arrow(ax,x+0.95,1.8,4.9,0.95,color=GREEN,rad=-0.1)
    box(ax,3.7,0.35,2.6,0.6,"average gradients → one update",fc="#dcfce7",ec=GREEN,bold=True,fs=9)
    ax.text(9.4,1.4,"avg of shard\ngradients =\nfull-batch\ngradient (exact)",fontsize=8,color=GREEN,ha="right")
    fig.savefig(os.path.join(OUT,"parallelism.png"),bbox_inches="tight"); plt.close(fig)

# 5) mixed_precision.png
def mixed_precision():
    fig,ax=plt.subplots(figsize=(10,4.2)); ax.set_xlim(-9,-1); ax.set_ylim(0,4.2); ax.axis("off")
    ax.text(-5.0,4.0,"Loss scaling: lift tiny gradients above the fp16 underflow floor",ha="center",fontsize=12,fontweight="bold",color=INK)
    ax.annotate("",xy=(-1.2,2),xytext=(-8.8,2),arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    for e in range(-8,-1): ax.plot([e,e],[1.9,2.1],color=SLATE,lw=1); ax.text(e,1.62,f"$10^{{{e}}}$",ha="center",fontsize=8,color=SLATE)
    floor=np.log10(6.1e-5)
    ax.axvspan(-9,floor,color="#fee2e2",alpha=0.6); ax.text((-9+floor)/2,3.25,"fp16 underflow zone\n(rounds to 0)",ha="center",fontsize=9.5,color=RED)
    ax.axvline(floor,color=RED,lw=2,ls="--"); ax.text(floor,3.75,"fp16 floor ~6e-5",ha="center",fontsize=8.5,color=RED)
    g=np.log10(2e-5); ax.scatter([g],[2],s=130,color=RED,zorder=5); ax.annotate("tiny gradient 2e-5\n→ LOST (underflows)",(g,2),(g,0.85),fontsize=8.5,color=RED,ha="center",arrowprops=dict(arrowstyle="->"))
    gs=np.log10(2e-5*1024); ax.scatter([gs],[2],s=130,color=GREEN,zorder=5); ax.annotate("×1024 → 0.02\nSAFE (survives fp16)",(gs,2),(gs,3.15),fontsize=8.5,color=GREEN,ha="center",arrowprops=dict(arrowstyle="->"))
    ax.annotate("",xy=(gs-0.15,2.35),xytext=(g+0.15,2.35),arrowprops=dict(arrowstyle="->",color=GREEN,lw=2.2,connectionstyle="arc3,rad=-0.25"))
    ax.text((g+gs)/2,2.95,"loss scaling",ha="center",fontsize=9.5,color=GREEN,fontweight="bold")
    fig.savefig(os.path.join(OUT,"mixed_precision.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    distillation(); pruning(); quantization(); parallelism(); mixed_precision()
    print("wrote:", sorted(os.listdir(OUT)))
