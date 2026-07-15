"""Generate Track 18 (Causal Inference) diagrams -> assets/img/track-22/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.abspath(os.path.join(HERE,"..","..","assets","img","track-22"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";GREEN="#059669";AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})
rng=np.random.default_rng(0)

def node(ax,x,y,lab,fc,r=0.32):
    ax.add_patch(Circle((x,y),r,fc=fc,ec="white",lw=2,zorder=5))
    ax.text(x,y,lab,ha="center",va="center",color="white",fontsize=14,fontweight="bold",zorder=6)
def edge(ax,a,b,color=SLATE,rad=0,ls="-",lw=2.4,r=0.32):
    import numpy as _n
    d=_n.array(b)-_n.array(a); L=_n.hypot(*d); u=d/L
    a2=(a[0]+u[0]*r, a[1]+u[1]*r); b2=(b[0]-u[0]*r, b[1]-u[1]*r)
    ax.add_patch(FancyArrowPatch(a2,b2,arrowstyle="-|>",mutation_scale=16,color=color,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))

# 1) confounder_dag.png
def confounder_dag():
    fig,ax=plt.subplots(figsize=(7.5,4.4)); ax.set_xlim(0,7.5); ax.set_ylim(0,4.4); ax.axis("off")
    Z=(3.75,3.6); X=(1.6,1.2); Y=(5.9,1.2)
    edge(ax,Z,X,color=SLATE); edge(ax,Z,Y,color=SLATE)
    edge(ax,X,Y,color=RED,ls="--",rad=0)
    node(ax,*Z,"Z",VIOLET); node(ax,*X,"X",BLUE); node(ax,*Y,"Y",GREEN)
    ax.text(3.75,4.05,"Z = common cause (e.g. hot weather)",ha="center",fontsize=10,color=VIOLET,fontweight="bold")
    ax.text(1.2,0.5,"X = ice-cream sales",ha="center",fontsize=9,color=BLUE)
    ax.text(6.1,0.5,"Y = drownings",ha="center",fontsize=9,color=GREEN)
    ax.text(3.75,1.55,"spurious correlation\n(no real X→Y link)",ha="center",fontsize=8.5,color=RED)
    ax.text(3.75,-0.0,"A confounder makes X and Y move together with no causal link between them",ha="center",fontsize=10.5,fontweight="bold",color=INK)
    fig.savefig(os.path.join(OUT,"confounder_dag.png"),bbox_inches="tight"); plt.close(fig)

# 2) simpson.png (real data)
def simpson():
    n=400
    sick=(rng.random(n)<0.5).astype(int)
    T=(rng.random(n)<np.where(sick==1,0.85,0.15)).astype(int)
    base=np.where(sick==1,30.,70.); Y=base+10*T+rng.standard_normal(n)*3
    fig,ax=plt.subplots(figsize=(8.2,4.6))
    jx=T+ (rng.random(n)-0.5)*0.16
    for s,col,lab in [(1,AMBER,"sick patients"),(0,BLUE,"healthy patients")]:
        m=sick==s; ax.scatter(jx[m],Y[m],s=14,color=col,alpha=0.45,label=lab)
        for t in (0,1):
            mm=m&(T==t); ax.scatter(t, Y[mm].mean(), s=180, marker="X", color=col, edgecolor="black", zorder=6)
    # within-group lines (up)
    for s,col in [(1,AMBER),(0,BLUE)]:
        m=sick==s; y0=Y[m&(T==0)].mean(); y1=Y[m&(T==1)].mean(); ax.plot([0,1],[y0,y1],color=col,lw=2.5)
    # overall line (down)
    o0=Y[T==0].mean(); o1=Y[T==1].mean(); ax.plot([0,1],[o0,o1],color=RED,lw=3,ls="--",label="overall trend (misleading)")
    ax.set_xticks([0,1]); ax.set_xticklabels(["untreated","treated"]); ax.set_ylabel("recovery outcome")
    ax.set_title("Simpson's paradox: the drug helps each group (↑) but looks harmful overall (↓)",fontsize=11,fontweight="bold")
    ax.legend(fontsize=8.5,loc="center left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"simpson.png"),bbox_inches="tight"); plt.close(fig)

# 3) do_surgery.png
def do_surgery():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for a,title,cut in [(ax[0],"Observational:  P(Y | T=t)  — SEEING",False),(ax[1],"Interventional:  P(Y | do(T=t))  — DOING",True)]:
        a.set_xlim(0,5); a.set_ylim(0,4.4); a.axis("off")
        Z=(2.5,3.6); T=(1.1,1.2); Y=(3.9,1.2)
        if not cut: edge(a,Z,T,color=SLATE)
        else:
            edge(a,Z,T,color="#e5e7eb"); a.plot([1.5,2.1],[2.6,2.2],color=RED,lw=3); a.plot([2.1,1.5],[2.6,2.2],color=RED,lw=3)
            a.text(1.35,2.75,"cut",color=RED,fontsize=9,fontweight="bold")
        edge(a,Z,Y,color=SLATE); edge(a,T,Y,color=BLUE)
        node(a,*Z,"Z",VIOLET); node(a,*T,"T",BLUE); node(a,*Y,"Y",GREEN)
        a.set_title(title,fontsize=10.5,fontweight="bold")
        if cut: a.text(1.1,0.4,"set T by fiat\n(no longer chosen by Z)",ha="center",fontsize=8.5,color=RED)
    fig.suptitle("The do-operator is graph surgery: delete the arrows INTO T, then set T",fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"do_surgery.png"),bbox_inches="tight"); plt.close(fig)

# 4) backdoor_bars.png (real)
def backdoor_bars():
    n=40000; Z=(rng.random(n)<0.5).astype(int); T=(rng.random(n)<np.where(Z==1,0.8,0.2)).astype(int)
    Y=20.*Z+5.*T+rng.standard_normal(n)*2
    naive=Y[T==1].mean()-Y[T==0].mean()
    adj=sum((Z==z).mean()*(Y[(T==1)&(Z==z)].mean()-Y[(T==0)&(Z==z)].mean()) for z in (0,1))
    fig,ax=plt.subplots(figsize=(7.6,4.2))
    bars=ax.bar(["naive\n(confounded)","backdoor-\nadjusted","true effect"],[naive,adj,5.0],color=[RED,GREEN,SLATE],edgecolor="black",width=0.6)
    for b,v in zip(bars,[naive,adj,5.0]): ax.text(b.get_x()+b.get_width()/2,v+0.3,f"{v:.1f}",ha="center",fontweight="bold")
    ax.axhline(5,color=GREY,ls=":",lw=1.2); ax.set_ylabel("estimated effect of T on Y")
    ax.set_title("Backdoor adjustment recovers the true effect from observational data",fontsize=11,fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"backdoor_bars.png"),bbox_inches="tight"); plt.close(fig)

# 5) ipw_bars.png (real)
def ipw_bars():
    def sig(z): return 1/(1+np.exp(-z))
    n=20000; Z=rng.standard_normal(n); T=(rng.random(n)<sig(1.5*Z)).astype(int); Y=3*Z+4*T+rng.standard_normal(n)
    naive=Y[T==1].mean()-Y[T==0].mean()
    w=0;b=0
    for _ in range(500): p=sig(w*Z+b); e=p-T; w-=0.1*(e*Z).mean(); b-=0.1*e.mean()
    eh=np.clip(sig(w*Z+b),0.05,0.95)
    ate=(T*Y/eh).sum()/(T/eh).sum()-((1-T)*Y/(1-eh)).sum()/((1-T)/(1-eh)).sum()
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].scatter(Z[T==1][:400],np.ones(400)+0.03*rng.standard_normal(400),s=8,color=BLUE,alpha=0.4,label="treated")
    ax[0].scatter(Z[T==0][:400],np.zeros(400)+0.03*rng.standard_normal(400),s=8,color=RED,alpha=0.4,label="untreated")
    ax[0].set_yticks([0,1]); ax[0].set_yticklabels(["untreated","treated"]); ax[0].set_xlabel("confounder Z")
    ax[0].set_title("Treatment depends on Z (confounded)",fontsize=10,fontweight="bold"); ax[0].legend(fontsize=8)
    bars=ax[1].bar(["naive","IPW\nestimate","true ATE"],[naive,ate,4.0],color=[RED,GREEN,SLATE],edgecolor="black",width=0.6)
    for bb,v in zip(bars,[naive,ate,4.0]): ax[1].text(bb.get_x()+bb.get_width()/2,v+0.1,f"{v:.1f}",ha="center",fontweight="bold")
    ax[1].axhline(4,color=GREY,ls=":"); ax[1].set_ylabel("estimated ATE"); ax[1].set_title("IPW recovers the true ATE",fontsize=10,fontweight="bold")
    fig.suptitle("Inverse propensity weighting rebalances confounded groups to estimate the causal effect",fontsize=11,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"ipw_bars.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    confounder_dag(); simpson(); do_surgery(); backdoor_bars(); ipw_bars()
    print("wrote:", sorted(os.listdir(OUT)))
