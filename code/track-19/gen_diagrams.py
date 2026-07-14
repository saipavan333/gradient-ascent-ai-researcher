"""Generate Track 16 (Meta/Few-Shot/Continual) diagrams -> assets/img/track-19/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.abspath(os.path.join(HERE,"..","..","assets","img","track-19"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";GREEN="#059669";AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})
rng=np.random.default_rng(1)

# 1) few_shot_episode.png — support + query, nearest centroid
def few_shot_episode():
    cols=[BLUE,GREEN,AMBER]
    C=np.array([[-2.2,1.5],[2.2,1.8],[0.0,-2.2]])
    fig,ax=plt.subplots(figsize=(8.2,4.6)); ax.axis("off"); ax.set_xlim(-5,5); ax.set_ylim(-4.5,4.2); ax.set_aspect("equal")
    protos=[]
    for c in range(3):
        sup=C[c]+rng.standard_normal((3,2))*0.6
        qry=C[c]+rng.standard_normal((10,2))*0.6
        ax.scatter(*qry.T, s=22, color=cols[c], alpha=0.35)
        ax.scatter(*sup.T, s=90, color=cols[c], edgecolor="black", lw=1.3, zorder=4)
        pr=sup.mean(0); protos.append(pr)
        ax.scatter(*pr, s=260, marker="*", color=cols[c], edgecolor="black", lw=1.2, zorder=6)
    ax.set_title("A few-shot episode: 3-way, 3-shot",fontsize=12,fontweight="bold")
    ax.scatter([],[],s=90,color="grey",edgecolor="black",label="support (few labelled)")
    ax.scatter([],[],s=22,color="grey",alpha=0.4,label="query (to classify)")
    ax.scatter([],[],s=200,marker="*",color="grey",edgecolor="black",label="class prototype (mean)")
    ax.legend(loc="lower center",fontsize=8.5,ncol=3,bbox_to_anchor=(0.5,-0.13))
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"few_shot_episode.png"),bbox_inches="tight"); plt.close(fig)

# 2) maml_adaptation.png — recompute a small Reptile run, show fits + loss curve
def maml_adaptation():
    def init():
        s=0.3; return [rng.standard_normal((1,40))*s,np.zeros(40),rng.standard_normal((40,40))*s,np.zeros(40),rng.standard_normal((40,1))*s,np.zeros(1)]
    def fwd(P,x):
        W1,b1,W2,b2,W3,b3=P; h1=np.tanh(x@W1+b1); h2=np.tanh(h1@W2+b2); return h2@W3+b3,(x,h1,h2)
    def grad(P,x,y):
        W1,b1,W2,b2,W3,b3=P; o,(x_,h1,h2)=fwd(P,x); d=2*(o-y)/len(x)
        gW3=h2.T@d; gb3=d.sum(0); dh2=(d@W3.T)*(1-h2**2); gW2=h1.T@dh2; gb2=dh2.sum(0)
        dh1=(dh2@W2.T)*(1-h1**2); gW1=x_.T@dh1; gb1=dh1.sum(0); return [gW1,gb1,gW2,gb2,gW3,gb3]
    def steps(P,x,y,n,lr):
        P=[w.copy() for w in P]
        for _ in range(n):
            g=grad(P,x,y)
            for i in range(len(P)): P[i]-=lr*g[i]
        return P
    def loss(P,x,y): o,_=fwd(P,x); return float(((o-y)**2).mean())
    P=init()
    for it in range(2500):
        A=rng.uniform(0.5,3); ph=rng.uniform(0,np.pi); x=rng.uniform(-5,5,(20,1)); y=A*np.sin(x+ph)
        Pk=steps(P,x,y,8,0.02)
        for i in range(len(P)): P[i]+=0.05*(Pk[i]-P[i])
    # test task
    A,ph=2.0,0.5; xs=rng.uniform(-5,5,(10,1)); ys=A*np.sin(xs+ph)
    xg=np.linspace(-5,5,200).reshape(-1,1); yt=A*np.sin(xg+ph)
    Pm=steps(P,xs,ys,5,0.02); Pr0=init(); Pr=steps(Pr0,xs,ys,5,0.02)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].plot(xg,yt,color=GREY,lw=2,ls=":",label="true sine (new task)")
    ax[0].plot(xg,fwd(Pm,xg)[0],color=BLUE,lw=2.5,label="from META init (5 steps)")
    ax[0].plot(xg,fwd(Pr,xg)[0],color=RED,lw=2,label="from RANDOM init (5 steps)")
    ax[0].scatter(xs,ys,color="black",s=35,zorder=5,label="10 support points")
    ax[0].set_title("After 5 gradient steps on a new task",fontsize=10.5,fontweight="bold"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
    lm=[loss(steps(P,xs,ys,k,0.02),xg,yt) for k in range(0,11)]
    lr=[loss(steps(Pr0,xs,ys,k,0.02),xg,yt) for k in range(0,11)]
    ax[1].plot(range(11),lm,color=BLUE,lw=2.5,marker="o",ms=4,label="meta init")
    ax[1].plot(range(11),lr,color=RED,lw=2.5,marker="s",ms=4,label="random init")
    ax[1].set_xlabel("gradient steps on the new task"); ax[1].set_ylabel("query loss")
    ax[1].set_title("Meta-init starts close & adapts fast",fontsize=10.5,fontweight="bold"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.2)
    fig.suptitle("MAML/Reptile: learn an initialisation that any new task can reach in a few steps",fontsize=11.5,fontweight="bold",y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"maml_adaptation.png"),bbox_inches="tight"); plt.close(fig)

# 3) protonet.png — embedding space with prototypes
def protonet():
    cols=[BLUE,GREEN,AMBER,VIOLET,RED]
    C=np.array([[-2.5,1.8],[2.6,2.0],[-2.8,-2.0],[2.4,-2.2],[0,0.2]])
    fig,ax=plt.subplots(figsize=(7.6,5.0)); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for c in range(5):
        sup=C[c]+rng.standard_normal((5,2))*0.55
        ax.scatter(*sup.T,s=45,color=cols[c],alpha=0.55)
        pr=sup.mean(0); ax.scatter(*pr,s=320,marker="*",color=cols[c],edgecolor="black",lw=1.3,zorder=6)
    q=np.array([0.4,1.0])
    ax.scatter(*q,s=140,marker="X",color="black",zorder=7)
    ax.annotate("new query →\nnearest prototype wins",q,(q[0]+0.6,q[1]+1.2),fontsize=9,ha="left",arrowprops=dict(arrowstyle="->"))
    ax.set_title("Prototypical network: each class = the mean of its support (★)",fontsize=11.5,fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"protonet.png"),bbox_inches="tight"); plt.close(fig)

# 4) forgetting_curve.png — A accuracy collapses as B is learned
def forgetting_curve():
    def blobs(cs,n=200):
        X,y=[],[]
        for c,mu in enumerate(cs): X.append(np.array(mu)+rng.standard_normal((n,2))); y+=[c]*n
        return np.vstack(X),np.array(y)
    def relu(x): return np.maximum(0,x)
    def sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
    W1=rng.standard_normal((2,64))*0.3;b1=np.zeros(64);W2=rng.standard_normal((64,2))*0.3;b2=np.zeros(2)
    def fwd(X): h=relu(X@W1+b1); return sm(h@W2+b2),h
    def step(X,y,lr=0.1):
        nonlocal W1,b1,W2,b2
        P,h=fwd(X);G=P.copy();G[np.arange(len(y)),y]-=1;G/=len(y)
        gW2=h.T@G;gb2=G.sum(0);dh=(G@W2.T)*(h>0);gW1=X.T@dh;gb1=dh.sum(0)
        W1-=lr*gW1;b1-=lr*gb1;W2-=lr*gW2;b2-=lr*gb2
    def acc(X,y): return (fwd(X)[0].argmax(1)==y).mean()
    XA,yA=blobs([[-2,-2.5],[-2,2.5]]); XB,yB=blobs([[2,2.5],[2,-2.5]])
    for _ in range(400): step(XA,yA)
    accA=[acc(XA,yA)]; accB=[acc(XB,yB)]
    for _ in range(400): step(XB,yB); accA.append(acc(XA,yA)); accB.append(acc(XB,yB))
    fig,ax=plt.subplots(figsize=(8.4,4.4))
    ax.plot(accA,color=RED,lw=2.5,label="task A accuracy")
    ax.plot(accB,color=GREEN,lw=2.5,label="task B accuracy")
    ax.axvline(0,color=GREY,ls=":"); ax.text(3,0.1,"← now training on task B only",fontsize=9,color=SLATE)
    ax.set_xlabel("gradient steps while learning task B"); ax.set_ylabel("accuracy")
    ax.set_title("Catastrophic forgetting: learning B destroys task A",fontsize=11.5,fontweight="bold")
    ax.legend(fontsize=9.5,loc="center right"); ax.grid(alpha=0.2); ax.set_ylim(-0.03,1.05)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"forgetting_curve.png"),bbox_inches="tight"); plt.close(fig)

# 5) replay.png — bar chart naive vs replay
def replay_bar():
    fig,ax=plt.subplots(figsize=(7.6,4.2))
    labels=["Task A\n(naive)","Task A\n(replay)","Task B\n(replay)"]
    vals=[0.07,0.93,0.98]; cols=[RED,GREEN,GREEN]
    bars=ax.bar(labels,vals,color=cols,edgecolor="black",width=0.6)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontweight="bold")
    ax.axhline(0.5,color=GREY,ls=":",lw=1); ax.text(2.4,0.52,"chance",fontsize=8,color=GREY)
    ax.set_ylabel("accuracy after learning B"); ax.set_ylim(0,1.1)
    ax.set_title("Replay keeps task A alive while learning task B",fontsize=11.5,fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"replay.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    few_shot_episode(); maml_adaptation(); protonet(); forgetting_curve(); replay_bar()
    print("wrote:", sorted(os.listdir(OUT)))
