"""Generate Track 15 (Probabilistic & Bayesian ML) diagrams -> assets/img/track-21/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.abspath(os.path.join(HERE,"..","..","assets","img","track-21"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";GREEN="#059669";AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})
rng=np.random.default_rng(0)

# 1) bayes_update.png
def bayes_update():
    from math import lgamma
    def beta_pdf(x,a,b):
        B=np.exp(lgamma(a)+lgamma(b)-lgamma(a+b)); return x**(a-1)*(1-x)**(b-1)/B
    x=np.linspace(1e-3,1-1e-3,400); h,t=7,3
    prior=beta_pdf(x,2,2); lik=x**h*(1-x)**t; lik=lik/np.trapz(lik,x); post=beta_pdf(x,2+h,2+t)
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    ax.plot(x,prior,color=GREY,lw=2.4,label="prior  Beta(2,2)  — belief before")
    ax.plot(x,lik,color=AMBER,lw=2.4,ls="--",label="likelihood  (7 heads / 10 flips)")
    ax.plot(x,post,color=BLUE,lw=3,label="posterior  Beta(9,5)  — updated belief")
    ax.fill_between(x,post,color=BLUE,alpha=0.12)
    ax.axvline(0.7,color=GREEN,ls=":",lw=1.5); ax.text(0.71,ax.get_ylim()[1]*0.9,"true p",color=GREEN,fontsize=9)
    ax.set_xlabel("p  (the coin's bias)"); ax.set_ylabel("density")
    ax.set_title("Bayes' rule: posterior ∝ prior × likelihood",fontsize=12,fontweight="bold")
    ax.legend(fontsize=9,loc="upper left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"bayes_update.png"),bbox_inches="tight"); plt.close(fig)

# 2) mcmc_hist.png
def mcmc_hist():
    def logt(x): return -((x**2-4.0)**2)/8.0
    x=0.0; lp=logt(x); S=120000; samp=np.empty(S)
    for i in range(S):
        xp=x+1.0*rng.standard_normal(); lpp=logt(xp)
        if np.log(rng.random())<lpp-lp: x,lp=xp,lpp
        samp[i]=x
    burn=samp[10000:]
    grid=np.linspace(-6,6,600); dens=np.exp(logt(grid)); dens/=np.trapz(dens,grid)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].plot(samp[:1500],color=BLUE,lw=0.6); ax[0].set_title("The chain wandering (trace)",fontsize=10.5,fontweight="bold")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("x"); ax[0].axhline(2,color=GREY,ls=":"); ax[0].axhline(-2,color=GREY,ls=":")
    ax[1].hist(burn,bins=60,density=True,color="#bfdbfe",edgecolor="#93c5fd",label="MCMC samples")
    ax[1].plot(grid,dens,color=RED,lw=2.5,label="true target density")
    ax[1].set_title("Samples match the target",fontsize=10.5,fontweight="bold")
    ax[1].set_xlabel("x"); ax[1].legend(fontsize=9)
    fig.suptitle("MCMC (Metropolis–Hastings): draw samples from a two-humped distribution with no formula",
                 fontsize=11.5,fontweight="bold",y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"mcmc_hist.png"),bbox_inches="tight"); plt.close(fig)

# 3) vi_fit.png
def vi_fit():
    def logj(th,h=8,t=2):
        p=1/(1+np.exp(-th)); return h*np.log(p+1e-12)+t*np.log(1-p+1e-12)-0.5*(th/1.5)**2
    m,logs=0.0,0.0; S=200
    for it in range(4000):
        s=np.exp(logs); eps=rng.standard_normal(S); th=m+s*eps; hh=1e-4
        g=(logj(th+hh)-logj(th-hh))/(2*hh); m+=0.02*g.mean(); logs+=0.02*((g*eps).mean()+1/s)*s
    grid=np.linspace(-4,5,600); post=np.exp(logj(grid)); post/=np.trapz(post,grid)
    s=np.exp(logs); q=np.exp(-0.5*((grid-m)/s)**2)/(s*np.sqrt(2*np.pi))
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    ax.fill_between(grid,post,color=BLUE,alpha=0.2); ax.plot(grid,post,color=BLUE,lw=2.6,label="true posterior (intractable)")
    ax.plot(grid,q,color=VIOLET,lw=2.6,ls="--",label="variational q  (fitted Gaussian)")
    ax.set_xlabel("θ  (log-odds of the coin)"); ax.set_ylabel("density")
    ax.set_title("Variational inference: fit a simple q to the posterior by maximising the ELBO",fontsize=11.5,fontweight="bold")
    ax.legend(fontsize=9.5); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"vi_fit.png"),bbox_inches="tight"); plt.close(fig)

# 4) gp_posterior.png
def gp_posterior():
    def rbf(A,B,l=1.0): d=A[:,None]-B[None,:]; return np.exp(-(d**2)/(2*l**2))
    Xtr=np.array([-4.,-3.,-1.,0.,2.,3.5]); ytr=np.sin(Xtr)
    K=rbf(Xtr,Xtr)+1e-4*np.eye(len(Xtr)); Kinv=np.linalg.inv(K)
    Xte=np.linspace(-6,6,300); Ks=rbf(Xte,Xtr)
    mean=Ks@Kinv@ytr; var=np.clip(1-np.einsum('ij,jk,ik->i',Ks,Kinv,Ks),0,None); sd=np.sqrt(var)
    fig,ax=plt.subplots(figsize=(9,4.4))
    ax.fill_between(Xte,mean-2*sd,mean+2*sd,color="#bfdbfe",alpha=0.7,label="±2 std (uncertainty)")
    ax.plot(Xte,np.sin(Xte),color=GREY,lw=1.5,ls=":",label="true function")
    ax.plot(Xte,mean,color=BLUE,lw=2.5,label="GP mean")
    ax.scatter(Xtr,ytr,color=RED,s=55,zorder=5,label="data")
    ax.set_title("Gaussian process: confident near data, honestly uncertain far from it",fontsize=11.5,fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=9,loc="lower left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"gp_posterior.png"),bbox_inches="tight"); plt.close(fig)

# 5) calibration.png
def calibration():
    def sig(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
    z=rng.standard_normal(4000); y=(rng.random(4000)<sig(z)).astype(int); logits=2.3*z
    def reliability(p,bins=10):
        edges=np.linspace(0,1,bins+1); xs=[];ys=[]
        for k in range(bins):
            m=(p>=edges[k])&(p<=edges[k+1] if k==bins-1 else p<edges[k+1])
            if m.sum()>5: xs.append(p[m].mean()); ys.append(y[m].mean())
        return np.array(xs),np.array(ys)
    Ts=np.linspace(0.5,4,200); nll=[-(y*np.log(sig(logits/T)+1e-12)+(1-y)*np.log(1-sig(logits/T)+1e-12)).mean() for T in Ts]
    T=Ts[int(np.argmin(nll))]
    fig,ax=plt.subplots(figsize=(6.4,5.2))
    ax.plot([0,1],[0,1],color=SLATE,ls="--",lw=1.5,label="perfect calibration")
    x0,y0=reliability(sig(logits)); ax.plot(x0,y0,"o-",color=RED,lw=2,ms=6,label="over-confident model")
    x1,y1=reliability(sig(logits/T)); ax.plot(x1,y1,"o-",color=GREEN,lw=2,ms=6,label=f"after temperature scaling (T={T:.1f})")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("actual accuracy")
    ax.set_title("Reliability diagram: below the line = over-confident",fontsize=11.5,fontweight="bold")
    ax.legend(fontsize=9,loc="upper left"); ax.grid(alpha=0.2); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"calibration.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    bayes_update(); mcmc_hist(); vi_fit(); gp_posterior(); calibration()
    print("wrote:", sorted(os.listdir(OUT)))
