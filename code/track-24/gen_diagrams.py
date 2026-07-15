"""Generate Track 20 (Frontier Architectures & Agents) diagrams -> assets/img/track-24/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.abspath(os.path.join(HERE,"..","..","assets","img","track-24"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";LIGHT="#eff6ff";GREEN="#059669";AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})
def box(ax,x,y,w,h,t,fc=LIGHT,ec=BLUE,tc=INK,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(ax,x1,y1,x2,y2,color=SLATE,rad=0,lw=2,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=color,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))

# 1) moe.png
def moe():
    fig,ax=plt.subplots(figsize=(10,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    ax.text(5,4.75,"Mixture of Experts: a gate routes each input to a few of many experts",ha="center",fontsize=11.5,fontweight="bold",color=INK)
    box(ax,0.4,2.1,1.5,0.9,"input",fc="#f1f5f9",ec=GREY,fs=10)
    box(ax,2.4,2.1,1.5,0.9,"GATE\n(router)",fc="#ede9fe",ec=VIOLET,bold=True,fs=9)
    arrow(ax,1.9,2.55,2.4,2.55)
    ys=[3.7,2.7,1.7,0.7]; active=[True,False,True,False]
    for i,y in enumerate(ys):
        act=active[i]
        box(ax,5.0,y,1.7,0.7,"Expert "+str(i+1),fc="#dcfce7" if act else "#f8fafc",ec=GREEN if act else "#e5e7eb",bold=act,fs=9,tc=INK if act else "#cbd5e1")
        arrow(ax,3.9,2.55,5.0,y+0.35,color=VIOLET if act else "#e5e7eb",lw=2.4 if act else 1)
        if act: arrow(ax,6.7,y+0.35,7.8,2.55,color=GREEN,rad=0.05)
    box(ax,7.8,2.1,1.6,0.9,"output",fc="#fee2e2",ec=RED,fs=10,bold=True)
    ax.text(5.85,0.15,"only the chosen experts run → huge total capacity, small compute per input",ha="center",fontsize=9.5,color=INK)
    fig.savefig(os.path.join(OUT,"moe.png"),bbox_inches="tight"); plt.close(fig)

# 2) ssm.png (real memory curve)
def ssm():
    T=200
    def scan(x,a): h=0.0; ys=[]
    def mem(a):
        h=0.0; ys=[]; x=np.zeros(T); x[0]=1.0
        for t in range(T): h=a*h+x[t]; ys.append(abs(h))
        return np.array(ys)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    a=ax[0]; a.set_xlim(0,10); a.set_ylim(0,3); a.axis("off")
    for k in range(4):
        box(a,0.3+k*2.4,1.3,1.3,0.8,("h$_{"+str(k)+"}$"),fc="#dbeafe",ec=BLUE,bold=True,fs=11)
        box(a,0.3+k*2.4,0.1,1.3,0.5,("x$_{"+str(k)+"}$"),fc="#f1f5f9",ec=GREY,fs=9)
        arrow(a,0.95+k*2.4,0.6,0.95+k*2.4,1.3,color=GREY)
        if k<3: arrow(a,1.6+k*2.4,1.7,2.7+k*2.4,1.7,color=BLUE)
    a.text(5,2.6,"h$_t$ = a·h$_{t-1}$ + b·x$_t$   —   O(n) linear recurrence",ha="center",fontsize=10.5,fontweight="bold",color=INK)
    a.text(5,0.9,"one state carried forward · no O(n²) attention",ha="center",fontsize=9,color=SLATE)
    ax[1].plot(mem(0.995),color=GREEN,lw=2.6,label="a=0.995 (remembers)")
    ax[1].plot(mem(0.5),color=RED,lw=2.6,label="a=0.5 (forgets fast)")
    ax[1].set_xlabel("time steps after a pulse at t=0"); ax[1].set_ylabel("state magnitude")
    ax[1].set_title("How long the state remembers",fontsize=10.5,fontweight="bold"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
    fig.suptitle("State-space models: a linear recurrence carries long-range memory in O(n) time",fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"ssm.png"),bbox_inches="tight"); plt.close(fig)

# 3) clip.png
def clip():
    fig,ax=plt.subplots(figsize=(9.5,4.6)); ax.set_xlim(0,9.5); ax.set_ylim(0,5); ax.axis("off")
    ax.text(4.75,4.75,"CLIP: image & text in one shared space, aligned by contrastive training",ha="center",fontsize=11,fontweight="bold",color=INK)
    box(ax,0.3,3.2,1.7,0.8,"image",fc="#dbeafe",ec=BLUE,fs=10)
    box(ax,0.3,0.9,1.7,0.8,"\"a photo\nof a cat\"",fc="#dcfce7",ec=GREEN,fs=9)
    box(ax,2.4,3.2,1.6,0.8,"image\nencoder",fc="#eff6ff",ec=BLUE,fs=9)
    box(ax,2.4,0.9,1.6,0.8,"text\nencoder",fc="#f0fdf4",ec=GREEN,fs=9)
    arrow(ax,2.0,3.6,2.4,3.6); arrow(ax,2.0,1.3,2.4,1.3)
    # shared space
    ax.add_patch(FancyBboxPatch((5.0,1.2),3.9,2.7,boxstyle="round,pad=0.02,rounding_size=0.1",fc="#faf5ff",ec=VIOLET,lw=2))
    ax.text(6.95,3.65,"shared embedding space",ha="center",fontsize=9.5,color=VIOLET,fontweight="bold")
    ax.scatter([6.2,6.4],[2.6,2.4],s=[90,90],color=[BLUE,GREEN]); ax.plot([6.2,6.4],[2.6,2.4],color=VIOLET,lw=2)
    ax.text(6.3,2.0,"matched:\nclose",fontsize=8,color=VIOLET,ha="center")
    ax.scatter([8.2],[3.2],s=90,color=BLUE); ax.scatter([7.9],[1.6],s=90,color=GREEN)
    ax.text(8.05,2.3,"mismatched: far",fontsize=8,color=GREY,ha="center")
    arrow(ax,4.0,3.6,5.0,2.9,color=BLUE); arrow(ax,4.0,1.3,5.0,2.1,color=GREEN)
    ax.text(4.75,0.35,"zero-shot: label a new image by its nearest text prompt — no task-specific training",ha="center",fontsize=9.5,color=INK)
    fig.savefig(os.path.join(OUT,"clip.png"),bbox_inches="tight"); plt.close(fig)

# 4) agent_loop.png
def agent_loop():
    fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.set_xlim(0,8.5); ax.set_ylim(0,5); ax.axis("off")
    ax.text(4.25,4.75,"An LLM agent: the reason → act → observe loop",ha="center",fontsize=12,fontweight="bold",color=INK)
    box(ax,3.2,3.4,2.1,0.8,"REASON\n(what to do?)",fc="#ede9fe",ec=VIOLET,bold=True,fs=9)
    box(ax,5.6,1.9,2.1,0.8,"ACT\n(call a tool)",fc="#dbeafe",ec=BLUE,bold=True,fs=9)
    box(ax,0.8,1.9,2.1,0.8,"OBSERVE\n(tool result)",fc="#dcfce7",ec=GREEN,bold=True,fs=9)
    arrow(ax,5.3,3.6,5.9,2.7,color=SLATE,rad=-0.3)
    arrow(ax,5.6,2.1,2.9,2.1,color=SLATE)
    arrow(ax,1.9,2.7,3.4,3.5,color=SLATE,rad=-0.3)
    ax.text(4.25,2.35,"repeat until\nit can answer",ha="center",fontsize=8.5,color=SLATE)
    box(ax,3.0,0.3,2.5,0.7,"tools: calculator, search, code",fc="#fff7ed",ec=AMBER,fs=9)
    arrow(ax,6.6,1.9,5.5,1.0,color=AMBER,rad=0.2); arrow(ax,4.0,1.0,2.0,1.9,color=AMBER,rad=0.2)
    fig.savefig(os.path.join(OUT,"agent_loop.png"),bbox_inches="tight"); plt.close(fig)

# 5) planning.png
def planning():
    fig,ax=plt.subplots(figsize=(9.5,4.4)); ax.set_xlim(0,9.5); ax.set_ylim(0,5); ax.axis("off")
    ax.text(4.75,4.8,"Planning with memory: search ahead to find a valid sequence of actions",ha="center",fontsize=11,fontweight="bold",color=INK)
    def nd(x,y,lab,goal=False,onpath=False):
        fc="#bbf7d0" if goal else ("#dcfce7" if onpath else "#dbeafe"); ec=GREEN if (goal or onpath) else BLUE
        ax.add_patch(Circle((x,y),0.34,fc=fc,ec=ec,lw=2,zorder=5)); ax.text(x,y,lab,ha="center",va="center",fontsize=10,fontweight="bold",zorder=6)
    # valid tree with ops +3 (x -> x+3) and x2 (x -> 2x); green path 1 ->+3 4 ->x2 8 ->x2 16
    E=[((1,2.5),(3,3.6),True),((1,2.5),(3,1.4),False),((3,3.6),(5,4.3),False),((3,3.6),(5,3.1),True),
       ((3,1.4),(5,1.4),False),((5,3.1),(7,3.9),False),((5,3.1),(7,2.6),True)]
    for a,b,g in E: ax.plot([a[0],b[0]],[a[1],b[1]],color=GREEN if g else SLATE,lw=2.6 if g else 1.4,zorder=1)
    nd(1,2.5,"1",onpath=True); nd(3,3.6,"4",onpath=True); nd(3,1.4,"2"); nd(5,4.3,"7"); nd(5,3.1,"8",onpath=True); nd(5,1.4,"5"); nd(7,3.9,"11"); nd(7,2.6,"16",goal=True)
    ax.text(7,2.05,"goal!",fontsize=9,color=GREEN,ha="center",fontweight="bold")
    ax.text(2.0,3.25,"+3",fontsize=8,color=GREEN); ax.text(4.0,3.55,"×2",fontsize=8,color=GREEN); ax.text(6.0,3.05,"×2",fontsize=8,color=GREEN)
    ax.plot([],[],color=GREEN,lw=2.6,label="plan found by search (BFS + memory of visited states)")
    ax.plot([],[],color=SLATE,lw=1.4,label="other states explored (a greedy agent gets stuck)")
    ax.legend(fontsize=9,loc="lower center",bbox_to_anchor=(0.5,-0.03))
    fig.savefig(os.path.join(OUT,"planning.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    moe(); ssm(); clip(); agent_loop(); planning()
    print("wrote:", sorted(os.listdir(OUT)))
