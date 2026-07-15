"""Generate Track 17 (Federated & Privacy) diagrams -> assets/img/track-18/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.abspath(os.path.join(HERE,"..","..","assets","img","track-18"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";LIGHT="#eff6ff";GREEN="#059669";AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})
rng=np.random.default_rng(0)
def box(ax,x,y,w,h,t,fc=LIGHT,ec=BLUE,tc=INK,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(ax,x1,y1,x2,y2,color=SLATE,rad=0,lw=2,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=color,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))

# 1) fedavg_cycle.png
def fedavg_cycle():
    fig,ax=plt.subplots(figsize=(10,4.8)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    box(ax,3.6,4.0,2.8,0.85,"SERVER: average the models\n→ new global model",fc="#dbeafe",ec=BLUE,bold=True,fs=10)
    xs=[0.6,3.0,5.4,7.8]
    for i,x in enumerate(xs):
        box(ax,x,0.5,1.9,1.5,"Client "+str(i+1)+"\n\nprivate data\n(never leaves)",fc="#dcfce7",ec=GREEN,fs=8.5)
        arrow(ax,x+1.5,2.0,4.4+0.0,4.0,color=AMBER,rad=-0.15)          # send weights up
        arrow(ax,5.0,4.0,x+0.5,2.0,color=BLUE,rad=0.15,ls="--")        # broadcast down
    ax.text(9.6,3.1,"↑ send only\nmodel weights",color=AMBER,fontsize=8.5,ha="right")
    ax.text(0.3,3.1,"↓ broadcast\nglobal model",color=BLUE,fontsize=8.5,ha="left")
    ax.text(5,4.92,"Federated learning: data stays put, only models travel",ha="center",fontsize=12,fontweight="bold",color=INK)
    fig.savefig(os.path.join(OUT,"fedavg_cycle.png"),bbox_inches="tight"); plt.close(fig)

# 2) non_iid_bars.png (real numbers)
def non_iid_bars():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    # left: data distribution illustration
    a=ax[0]; a.axis("off"); a.set_xlim(0,10); a.set_ylim(0,6)
    a.text(5,5.6,"How each client's data is distributed",ha="center",fontsize=11,fontweight="bold")
    a.text(2.5,4.7,"IID (balanced)",ha="center",fontsize=9.5,color=GREEN,fontweight="bold")
    a.text(7.5,4.7,"non-IID (skewed)",ha="center",fontsize=9.5,color=RED,fontweight="bold")
    import matplotlib.patches as mp
    for k in range(3):
        a.add_patch(mp.Wedge((2.5,2.0+0*k),0.6,0,180,fc=BLUE)); a.add_patch(mp.Wedge((2.5,2.0),0.6,180,360,fc=AMBER))
    a.add_patch(mp.Wedge((1.3,2.0),0.55,0,180,fc=BLUE)); a.add_patch(mp.Wedge((1.3,2.0),0.55,180,360,fc=AMBER))
    a.add_patch(mp.Wedge((2.5,2.0),0.55,0,180,fc=BLUE)); a.add_patch(mp.Wedge((2.5,2.0),0.55,180,360,fc=AMBER))
    a.add_patch(mp.Wedge((3.7,2.0),0.55,0,180,fc=BLUE)); a.add_patch(mp.Wedge((3.7,2.0),0.55,180,360,fc=AMBER))
    a.add_patch(Circle((6.3,2.0),0.55,fc=BLUE)); a.add_patch(Circle((7.5,2.0),0.55,fc=AMBER)); a.add_patch(Circle((8.7,2.0),0.55,fc=BLUE))
    a.text(2.5,0.9,"each client: both classes",ha="center",fontsize=8,color=SLATE)
    a.text(7.5,0.9,"each client: mostly one",ha="center",fontsize=8,color=SLATE)
    # right: accuracy bars (from non_iid.py)
    b=ax[1]; bars=b.bar(["IID\nFedAvg","non-IID\nFedAvg"],[0.997,0.879],color=[GREEN,RED],edgecolor="black",width=0.55)
    for bar,v in zip(bars,[0.997,0.879]): b.text(bar.get_x()+bar.get_width()/2,v+0.005,f"{v:.2f}",ha="center",fontweight="bold")
    b.set_ylim(0.7,1.03); b.set_ylabel("test accuracy"); b.set_title("Skewed data hurts FedAvg",fontsize=11,fontweight="bold")
    fig.suptitle("The non-IID challenge: when clients hold different data, averaged models clash",fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"non_iid_bars.png"),bbox_inches="tight"); plt.close(fig)

# 3) dp_distributions.png (real)
def dp_distributions():
    data=rng.random(1000); nb=data.copy(); nb[0]=1-nb[0]
    def pm(d,eps): return d.mean()+rng.laplace(0,(1/len(d))/eps)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for eps,style in [(1.0,'-')]:
        A=[pm(data,eps) for _ in range(30000)]; B=[pm(nb,eps) for _ in range(30000)]
        ax[0].hist(A,bins=80,density=True,alpha=0.55,color=BLUE,label="dataset WITH person 0")
        ax[0].hist(B,bins=80,density=True,alpha=0.55,color=RED,label="dataset WITHOUT person 0")
    ax[0].set_title("Output looks the same either way (ε=1)",fontsize=10.5,fontweight="bold")
    ax[0].set_xlabel("released (noisy) mean"); ax[0].legend(fontsize=8.5)
    eps=np.array([0.05,0.1,0.3,1,3,10]); err=(1/1000)/eps
    ax[1].plot(eps,err,'o-',color=VIOLET,lw=2.4)
    ax[1].set_xscale('log'); ax[1].set_xlabel("privacy budget ε (log scale)"); ax[1].set_ylabel("typical error added")
    ax[1].set_title("Privacy–utility trade-off",fontsize=10.5,fontweight="bold"); ax[1].grid(alpha=0.25)
    ax[1].annotate("more privacy\n(more noise)",(0.06,err[0]),(0.2,err[0]*0.8),fontsize=8.5,color=SLATE,arrowprops=dict(arrowstyle="->"))
    fig.suptitle("Differential privacy: calibrated noise hides any one person",fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"dp_distributions.png"),bbox_inches="tight"); plt.close(fig)

# 4) dp_sgd_diagram.png
def dp_sgd_diagram():
    fig,ax=plt.subplots(figsize=(10,3.8)); ax.set_xlim(0,10); ax.set_ylim(0,3.8); ax.axis("off")
    ax.text(5,3.5,"DP-SGD = normal SGD + two privacy steps",ha="center",fontsize=12.5,fontweight="bold",color=INK)
    # step 1: clip
    ax.text(2.2,2.9,"① Clip each example's gradient to length C",ha="center",fontsize=9.5,color=RED,fontweight="bold")
    ax.add_patch(Circle((2.2,1.5),0.8,fc="none",ec=RED,lw=1.6,ls="--"))
    for a in [20,80,160,250,300]:
        import math; dx=math.cos(math.radians(a)),; 
        x=2.2+math.cos(math.radians(a))*1.3; y=1.5+math.sin(math.radians(a))*1.3
        xc=2.2+math.cos(math.radians(a))*0.8; yc=1.5+math.sin(math.radians(a))*0.8
        arrow(ax,2.2,1.5,x,y,color=GREY,lw=1); arrow(ax,2.2,1.5,xc,yc,color=RED,lw=2)
    ax.text(2.2,0.3,"big gradients capped → no single\nperson can dominate",ha="center",fontsize=8,color=SLATE)
    # step 2: noise
    ax.text(7.2,2.9,"② Add Gaussian noise to the summed gradient",ha="center",fontsize=9.5,color=VIOLET,fontweight="bold")
    arrow(ax,5.8,1.5,6.9,1.5,color=SLATE,lw=2.5)
    ax.text(6.35,1.75,"Σ",fontsize=13,ha="center",color=INK)
    for _ in range(40):
        ax.plot(7.2+rng.standard_normal()*0.35, 1.5+rng.standard_normal()*0.55, '.', color=VIOLET, ms=3, alpha=0.5)
    arrow(ax,7.9,1.5,9.0,1.5,color=GREEN,lw=2.5); ax.text(9.4,1.5,"step",fontsize=9,color=GREEN,va="center")
    ax.text(7.2,0.3,"noise blurs the exact update →\nformal privacy guarantee",ha="center",fontsize=8,color=SLATE)
    fig.savefig(os.path.join(OUT,"dp_sgd_diagram.png"),bbox_inches="tight"); plt.close(fig)

# 5) secure_agg.png
def secure_agg():
    fig,ax=plt.subplots(figsize=(10,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    ax.text(5,4.75,"Secure aggregation: server learns the SUM, never an individual update",ha="center",fontsize=11.5,fontweight="bold",color=INK)
    box(ax,3.7,3.4,2.6,0.8,"SERVER\nsees only Σ updates",fc="#dbeafe",ec=BLUE,bold=True,fs=9.5)
    labels=["u₁ + s₁₂ + s₁₃","u₂ − s₁₂ + s₂₃","u₃ − s₁₃ − s₂₃"]
    xs=[0.6,3.9,7.2]
    for i,x in enumerate(xs):
        box(ax,x,0.6,2.4,1.2,"Client "+str(i+1)+"\nsends:\n"+labels[i],fc="#dcfce7",ec=GREEN,fs=8.5)
        arrow(ax,x+1.2,1.8,5.0,3.4,color=AMBER,rad=-0.12)
    ax.text(5,2.35,"the masks cancel:  (+s₁₂) + (−s₁₂) = 0,  etc.  →  Σ = u₁+u₂+u₃ exactly",
            ha="center",fontsize=9.5,color=RED,fontweight="bold")
    ax.text(5,0.2,"each message alone looks like random noise; only the total is meaningful",ha="center",fontsize=9,color=SLATE)
    fig.savefig(os.path.join(OUT,"secure_agg.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    fedavg_cycle(); non_iid_bars(); dp_distributions(); dp_sgd_diagram(); secure_agg()
    print("wrote:", sorted(os.listdir(OUT)))
