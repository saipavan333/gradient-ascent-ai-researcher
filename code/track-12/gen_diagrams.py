"""Diagrams for the Linear & Logistic Regression lesson (Classical ML)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","assets","img","track-12"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb"; RED="#dc2626"; GREEN="#059669"; SLATE="#334155"; GREY="#94a3b8"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","savefig.dpi":140})
rng=np.random.default_rng(0)

def reg_fig():
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    # linear regression
    X=rng.uniform(-3,3,40); y=2.0*X-1+1.4*rng.standard_normal(40)
    w=np.polyfit(X,y,1)
    xs=np.linspace(-3.2,3.2,50)
    ax[0].scatter(X,y,s=28,color=BLUE,alpha=.75,zorder=3)
    for xi,yi in zip(X,y):
        ax[0].plot([xi,xi],[yi,w[0]*xi+w[1]],color=GREY,lw=.8,zorder=1)  # residuals
    ax[0].plot(xs,w[0]*xs+w[1],color=RED,lw=2.6,zorder=2,label="best-fit line")
    ax[0].set_title("Linear regression: minimise squared residuals",fontsize=11,fontweight="bold")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].legend(loc="upper left",fontsize=9)
    # logistic regression
    n=80
    Xa=rng.normal([-1.4,-1],0.9,(n,2)); Xb=rng.normal([1.4,1],0.9,(n,2))
    def sig(z): return 1/(1+np.exp(-z))
    # quick fit
    Xall=np.vstack([Xa,Xb]); yall=np.r_[np.zeros(n),np.ones(n)]
    w2=np.zeros(2); b=0.0
    for _ in range(600):
        p=sig(Xall@w2+b); e=p-yall; w2-=0.1*Xall.T@e/(2*n); b-=0.1*e.mean()
    gx,gy=np.meshgrid(np.linspace(-4,4,120),np.linspace(-4,4,120))
    P=sig(np.c_[gx.ravel(),gy.ravel()]@w2+b).reshape(gx.shape)
    ax[1].contourf(gx,gy,P,levels=20,cmap="coolwarm",alpha=.55)
    ax[1].contour(gx,gy,P,levels=[0.5],colors="k",linewidths=2)
    ax[1].scatter(Xa[:,0],Xa[:,1],s=20,color=BLUE,label="class 0",edgecolor="w",lw=.3)
    ax[1].scatter(Xb[:,0],Xb[:,1],s=20,color=RED,label="class 1",edgecolor="w",lw=.3)
    ax[1].set_title("Logistic regression: a probability boundary",fontsize=11,fontweight="bold")
    ax[1].set_xlabel("x₁"); ax[1].set_ylabel("x₂"); ax[1].legend(loc="upper left",fontsize=9)
    ax[1].text(-3.8,-3.6,"colour = P(class 1)\nblack line = 0.5 boundary",fontsize=8.5,color=SLATE)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"regression.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    reg_fig(); print("wrote regression.png to",OUT)
