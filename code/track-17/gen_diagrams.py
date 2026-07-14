"""Generate Track 13 (Learning Paradigms) diagrams -> assets/img/track-17/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "assets", "img", "track-17"))
os.makedirs(OUT, exist_ok=True)
BLUE="#2563eb"; SLATE="#334155"; LIGHT="#eff6ff"; GREEN="#059669"
AMBER="#d97706"; RED="#dc2626"; GREY="#94a3b8"; INK="#0f172a"; VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})

def box(ax,x,y,w,h,t,fc=LIGHT,ec=BLUE,tc=INK,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(ax,x1,y1,x2,y2,color=SLATE,rad=0.0,lw=2,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,color=color,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))

# 1) paradigms_map.png — the landscape ordered by how many labels you need
def paradigms_map():
    fig,ax=plt.subplots(figsize=(11,4.2)); ax.set_xlim(0,11); ax.set_ylim(0,4.2); ax.axis("off")
    ax.text(5.5,3.9,"Learning paradigms, ordered by how many human labels they need",
            ha="center",fontsize=12.5,fontweight="bold",color=INK)
    arrow(ax,0.6,0.6,10.4,0.6,color=SLATE,lw=2.5)
    ax.text(0.6,0.25,"every example labelled",fontsize=9,color=SLATE)
    ax.text(10.4,0.25,"no labels at all",fontsize=9,color=SLATE,ha="right")
    items=[(1.4,"Supervised","all labels","train on (x, y) pairs",BLUE,"#dbeafe"),
           (3.5,"Active","a few, chosen","model asks for the\nmost useful labels",AMBER,"#fef3c7"),
           (5.6,"Semi-\nsupervised","few + many\nunlabelled","spread labels through\nthe data's structure",GREEN,"#dcfce7"),
           (7.7,"Self-\nsupervised","none (invented)","predict part of the\ninput from the rest",VIOLET,"#ede9fe"),
           (9.7,"Unsupervised","none","find structure /\nclusters",SLATE,"#f1f5f9")]
    for x,name,lab,desc,ec,fc in items:
        box(ax,x-0.85,2.3,1.7,1.1,name,fc=fc,ec=ec,tc=INK,fs=9.5,bold=True)
        ax.text(x,1.85,lab,ha="center",fontsize=8.5,color=ec,fontweight="bold")
        ax.text(x,1.1,desc,ha="center",fontsize=8,color=SLATE)
        arrow(ax,x,1.35,x,0.75,color=GREY,lw=1.4)
    fig.savefig(os.path.join(OUT,"paradigms_map.png"),bbox_inches="tight"); plt.close(fig)

# 2) kmeans_steps.png — assign then update, twice
def kmeans_steps():
    rng=np.random.default_rng(0)
    C0=np.array([[-3,0],[0,3],[3,0]]); X=np.vstack([c+rng.standard_normal((40,2)) for c in C0])
    def step(C):
        a=((X[:,None]-C[None])**2).sum(2).argmin(1)
        Cn=np.array([X[a==j].mean(0) if (a==j).any() else C[j] for j in range(3)]); return a,Cn
    cent=X[rng.choice(len(X),3,replace=False)].copy()
    fig,axs=plt.subplots(1,3,figsize=(12,3.9)); cols=[BLUE,GREEN,AMBER]
    titles=["Start: random centres","After 1 round","After 3 rounds"]
    snaps=[cent.copy()]
    c=cent.copy()
    for _ in range(3): a,c=step(c); snaps.append(c.copy())
    show=[0,1,3]
    for k,ax in enumerate(axs):
        cc=snaps[show[k]]; a=((X[:,None]-cc[None])**2).sum(2).argmin(1)
        for j in range(3): ax.scatter(*X[a==j].T,s=14,color=cols[j],alpha=0.6)
        ax.scatter(*cc.T,s=260,marker="X",c="black",edgecolor="white",lw=1.5,zorder=5)
        ax.set_title(titles[k],fontsize=10.5,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("k-means: assign each point to its nearest centre, move centres to the mean, repeat",
                 fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"kmeans_steps.png"),bbox_inches="tight"); plt.close(fig)

# 3) autoencoder.png — bottleneck architecture
def autoencoder():
    fig,ax=plt.subplots(figsize=(11,4.2)); ax.set_xlim(0,11); ax.set_ylim(0,4.2); ax.axis("off")
    def col(x,n,lab,ec,fc):
        ys=np.linspace(2.1-0.16*(n-1),2.1+0.16*(n-1),n)
        for y in ys: ax.add_patch(Circle((x,y),0.16,fc=fc,ec=ec,lw=1.6))
        ax.text(x,ys[0]-0.5,lab,ha="center",fontsize=9,color=INK); return ys
    xs=[1.0,2.8,4.6,6.4,8.2,10.0]; ns=[8,4,2,4,8,8]
    labs=["input x\n(8 numbers)","","code z\n(2 numbers)","","output x̂\n(rebuilt)",""]
    ecs=[GREY,BLUE,VIOLET,BLUE,GREEN,GREEN]; fcs=["#f1f5f9","#dbeafe","#ede9fe","#dbeafe","#dcfce7","#dcfce7"]
    Y=[]
    for x,n,lab,ec,fc in zip(xs[:5],ns[:5],labs[:5],ecs[:5],fcs[:5]): Y.append(col(x,n,lab,ec,fc))
    for i in range(4):
        for ya in Y[i]:
            for yb in Y[i+1]: ax.plot([xs[i],xs[i+1]],[ya,yb],color=GREY,lw=0.3,alpha=0.5,zorder=0)
    ax.text(2.9,3.5,"ENCODER  (compress)",ha="center",fontsize=10,color=BLUE,fontweight="bold")
    ax.text(6.4,3.5,"DECODER  (rebuild)",ha="center",fontsize=10,color=GREEN,fontweight="bold")
    ax.annotate("",xy=(4.6,0.9),xytext=(4.6,0.55),arrowprops=dict(arrowstyle="-"))
    ax.text(4.6,0.35,"bottleneck: too small to memorise —\nmust capture the real structure",ha="center",fontsize=8.5,color=VIOLET)
    ax.text(5.5,3.95,"Autoencoder: reconstruct the input through a tiny code — a pretext task that needs NO labels",
            ha="center",fontsize=11.5,fontweight="bold",color=INK)
    fig.savefig(os.path.join(OUT,"autoencoder.png"),bbox_inches="tight"); plt.close(fig)

# 4) contrastive.png — pull same together, push others apart
def contrastive():
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    rng=np.random.default_rng(3)
    # left: the idea
    a=ax[0]; a.set_xlim(0,10); a.set_ylim(0,10); a.axis("off")
    a.text(5,9.4,"The rule",ha="center",fontsize=12,fontweight="bold",color=INK)
    a.scatter([3],[6],s=300,color=BLUE,zorder=3); a.text(2.55,6,"anchor",ha="right",va="center",color=BLUE,fontsize=9.5,fontweight="bold")
    a.scatter([4.6],[7.4],s=240,color="#60a5fa",zorder=3); a.text(5.05,7.4,"view A",ha="left",va="center",color="#2563eb",fontsize=9)
    a.scatter([4.6],[4.6],s=240,color="#60a5fa",zorder=3); a.text(5.05,4.6,"view B",ha="left",va="center",color="#2563eb",fontsize=9)
    arrow(a,4.2,7.2,3.35,6.25,color=GREEN,lw=2.2); arrow(a,4.2,4.8,3.35,5.75,color=GREEN,lw=2.2)
    a.text(6.2,6,"pull together\n(same source)",color=GREEN,fontsize=9,fontweight="bold",va="center")
    for (x,y) in [(7.8,2.2),(2.0,2.4),(8.2,8.0)]:
        a.scatter([x],[y],s=280,color=RED,zorder=3); arrow(a,x,y,3.0,6.0,color=RED,lw=1.6,ls="--",rad=0.15)
    a.text(5,1.1,"push apart (different sources = 'negatives')",color=RED,fontsize=9,fontweight="bold",ha="center")
    # right: before/after embedding
    b=ax[1]
    W=rng.standard_normal((2,2))*0.3
    def normal(v): return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-9)
    anchors=rng.standard_normal((8,2))*2
    b.set_title("Result: same-source views land close together",fontsize=10.5,fontweight="bold")
    cols=plt.cm.tab10(np.linspace(0,1,8))
    for i,an in enumerate(anchors):
        views=normal((W@(an+0.15*rng.standard_normal((6,2))).T).T)*0  # placeholder
    # simulate a trained embedding by clustering views of same anchor
    for i,an in enumerate(anchors):
        centre=np.array([np.cos(i/8*2*np.pi),np.sin(i/8*2*np.pi)])
        pts=centre+0.06*rng.standard_normal((6,2))
        b.scatter(*pts.T,s=30,color=cols[i])
    b.set_xticks([]); b.set_yticks([]); b.set_xlim(-1.4,1.4); b.set_ylim(-1.4,1.4)
    b.set_aspect("equal")
    fig.suptitle("Contrastive learning: no labels — the only signal is 'these two came from the same example'",
                 fontsize=11.5,fontweight="bold",y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"contrastive.png"),bbox_inches="tight"); plt.close(fig)

# 5) label_prop.png — labels spread from a few seeds
def label_prop():
    rng=np.random.default_rng(3)
    Xa=rng.normal([-2.2,0],0.75,(60,2)); Xb=rng.normal([2.2,0],0.75,(60,2))
    X=np.vstack([Xa,Xb]); y=np.r_[np.zeros(60),np.ones(60)].astype(int)
    seed=np.array([0,1,2,60,61,62])
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].set_title("Start: only 6 points are labelled",fontsize=10.5,fontweight="bold")
    ax[0].scatter(*X.T,s=18,color="#cbd5e1")
    ax[0].scatter(*X[seed[:3]].T,s=90,color=BLUE,edgecolor="black",zorder=5,label="class A seed")
    ax[0].scatter(*X[seed[3:]].T,s=90,color=RED,edgecolor="black",zorder=5,label="class B seed")
    ax[0].legend(fontsize=8,loc="upper center"); ax[0].set_xticks([]); ax[0].set_yticks([])
    # propagate
    d=((X[:,None]-X[None])**2).sum(2); nn=np.argsort(d,1)[:,1:7]
    sc=np.zeros((len(X),2)); sc[seed,y[seed]]=1
    for _ in range(60):
        nw=np.array([sc[nn[i]].sum(0) for i in range(len(X))]); nw[seed]=0; nw[seed,y[seed]]=1
        s=nw.sum(1,keepdims=True); s[s==0]=1; sc=nw/s
    pred=sc.argmax(1)
    ax[1].set_title("After propagation: all 120 points labelled",fontsize=10.5,fontweight="bold")
    ax[1].scatter(*X[pred==0].T,s=18,color=BLUE,alpha=0.6)
    ax[1].scatter(*X[pred==1].T,s=18,color=RED,alpha=0.6)
    ax[1].scatter(*X[seed].T,s=90,color="black",marker="*",zorder=5)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.suptitle("Semi-supervised label propagation: nearby points share a label, so a few seeds spread everywhere",
                 fontsize=11.5,fontweight="bold",y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"label_prop.png"),bbox_inches="tight"); plt.close(fig)

# 6) active_vs_random.png — real curves from the lesson's own method
def active_vs_random():
    import importlib.util
    spec=importlib.util.spec_from_file_location("al",os.path.join(HERE,"active_learning.py"))
    al=importlib.util.module_from_spec(spec); spec.loader.exec_module(al)
    seeds=range(12)
    A=np.mean([al.run("active",s) for s in seeds],0); R=np.mean([al.run("random",s) for s in seeds],0)
    xs=np.arange(4+1,4+1+len(A))
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].plot(xs,A,color=GREEN,lw=2.5,marker="o",ms=3,label="active (label the unsure ones)")
    ax[0].plot(xs,R,color=GREY,lw=2.5,marker="s",ms=3,label="random (label anything)")
    ax[0].set_xlabel("number of labels used"); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("Active learning climbs faster",fontsize=10.5,fontweight="bold")
    ax[0].legend(fontsize=8.5,loc="lower right"); ax[0].grid(alpha=0.25)
    # right: which points active picks (near the boundary)
    X,yv=al.data(400,0)
    ax[1].scatter(*X[yv==0].T,s=12,color="#bfdbfe"); ax[1].scatter(*X[yv==1].T,s=12,color="#fecaca")
    near=np.abs(X[:,0])<0.5
    ax[1].scatter(*X[near].T,s=40,facecolor="none",edgecolor=GREEN,lw=1.4,label="most uncertain\n(picked first)")
    ax[1].axvline(0,color=SLATE,ls=":",lw=1.5)
    ax[1].set_title("Uncertain points sit at the boundary",fontsize=10.5,fontweight="bold")
    ax[1].legend(fontsize=8,loc="upper right"); ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.suptitle("Active learning: let the model request labels for the points it is least sure about",
                 fontsize=11.5,fontweight="bold",y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"active_vs_random.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    paradigms_map(); kmeans_steps(); autoencoder(); contrastive(); label_prop(); active_vs_random()
    print("wrote:", sorted(os.listdir(OUT)))
