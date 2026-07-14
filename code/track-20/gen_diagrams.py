"""Generate Track 14 (Graph Neural Networks) diagrams -> assets/img/track-20/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.abspath(os.path.join(HERE,"..","..","assets","img","track-20"))
os.makedirs(OUT,exist_ok=True)
BLUE="#2563eb";SLATE="#334155";LIGHT="#eff6ff";GREEN="#059669"
AMBER="#d97706";RED="#dc2626";GREY="#94a3b8";INK="#0f172a";VIOLET="#6d28d9"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"savefig.dpi":140})

def box(ax,x,y,w,h,t,fc=LIGHT,ec=BLUE,tc=INK,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(ax,x1,y1,x2,y2,color=SLATE,rad=0.0,lw=2,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,color=color,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))
def node(ax,x,y,lab,fc=BLUE,r=0.28,tc="white",fs=11):
    ax.add_patch(Circle((x,y),r,fc=fc,ec="white",lw=2,zorder=5))
    ax.text(x,y,lab,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold",zorder=6)

# 1) graph_and_matrix.png
def graph_and_matrix():
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    a=ax[0]; a.set_xlim(0,6); a.set_ylim(0,6); a.axis("off"); a.set_aspect("equal")
    pos={0:(1.2,4.6),1:(3.0,5.2),2:(4.8,4.4),3:(1.4,2.2),4:(3.2,2.8),5:(4.6,1.8)}
    edges=[(0,1),(1,2),(0,3),(1,4),(3,4),(4,5),(2,5)]
    for i,j in edges: a.plot([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]],color=GREY,lw=2,zorder=1)
    for k,(x,y) in pos.items(): node(a,x,y,str(k),fc=BLUE)
    a.set_title("A graph: nodes joined by edges",fontsize=11.5,fontweight="bold")
    # adjacency matrix
    b=ax[1]; N=6; Amat=np.zeros((N,N))
    for i,j in edges: Amat[i,j]=Amat[j,i]=1
    b.imshow(Amat,cmap="Blues",vmin=0,vmax=1.4)
    b.set_xticks(range(N)); b.set_yticks(range(N))
    for i in range(N):
        for j in range(N):
            b.text(j,i,int(Amat[i,j]),ha="center",va="center",color=INK if Amat[i,j] else "#cbd5e1",fontsize=11)
    b.set_title("...stored as an adjacency matrix A",fontsize=11.5,fontweight="bold")
    b.set_xlabel("A[i,j] = 1 if edge i–j exists")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"graph_and_matrix.png"),bbox_inches="tight"); plt.close(fig)

# 2) message_passing.png
def message_passing():
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.4))
    # left: one node aggregating
    a=ax[0]; a.set_xlim(0,6); a.set_ylim(0,6); a.axis("off"); a.set_aspect("equal")
    ctr=(3,3); nbs=[(1,4.5),(5,4.3),(1.2,1.6),(4.8,1.8)]
    for x,y in nbs:
        a.add_patch(FancyArrowPatch((x,y),ctr,arrowstyle="-|>",mutation_scale=15,color=GREEN,lw=2,connectionstyle="arc3,rad=0.12",zorder=1))
        node(a,x,y,"",fc="#a7f3d0",r=0.26,tc=INK)
    node(a,ctr[0],ctr[1],"i",fc=BLUE,r=0.34)
    a.text(3,0.6,"node i = combine (average) the messages from its neighbours",ha="center",fontsize=9.5,color=INK)
    a.set_title("Message passing: aggregate from neighbours",fontsize=11.5,fontweight="bold")
    # right: diffusion of a signal, 3 snapshots stacked
    b=ax[1]; b.axis("off"); b.set_xlim(0,9); b.set_ylim(0,6)
    rng=np.random.default_rng(1)
    pos={0:(1,3),1:(2.2,4.3),2:(2.4,1.8),3:(3.8,3.2),4:(5.2,4.2),5:(5.4,1.9),6:(6.8,3)}
    edges=[(0,1),(0,2),(1,3),(2,3),(3,4),(3,5),(4,6),(5,6)]
    vals={0:1.0,1:0.6,2:0.6,3:0.15,4:-0.6,5:-0.6,6:-1.0}
    import matplotlib.cm as cm
    for i,j in edges: b.plot([pos[i][0]+1,pos[j][0]+1],[pos[i][1],pos[j][1]],color=GREY,lw=1.6,zorder=1)
    for k,(x,y) in pos.items():
        c=cm.RdBu_r((vals[k]+1)/2)
        b.add_patch(Circle((x+1,y),0.28,fc=c,ec="white",lw=1.5,zorder=5))
    b.set_title("A signal diffuses: neighbours pull toward each other",fontsize=11.5,fontweight="bold")
    b.text(4.9,0.5,"red = +1  ·  blue = −1  ·  after a few steps each side agrees",ha="center",fontsize=9,color=SLATE)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"message_passing.png"),bbox_inches="tight"); plt.close(fig)

# 3) gcn_layer.png
def gcn_layer():
    fig,ax=plt.subplots(figsize=(11,3.6)); ax.set_xlim(0,11); ax.set_ylim(0,3.6); ax.axis("off")
    ax.text(5.5,3.35,"One GCN layer:   H′ = σ( Â H W )",ha="center",fontsize=13,fontweight="bold",color=INK)
    box(ax,0.4,1.3,1.7,1.1,"H\nnode features",fc="#f1f5f9",ec=GREY,fs=9)
    box(ax,2.7,1.3,1.9,1.1,"Â H\nmix each node\nwith neighbours",fc="#dbeafe",ec=BLUE,fs=9,bold=True)
    box(ax,5.2,1.3,1.9,1.1,"(ÂH) W\nlearnable\ntransform",fc="#ede9fe",ec=VIOLET,fs=9,bold=True)
    box(ax,7.7,1.3,1.6,1.1,"σ(·)\nnonlinearity",fc="#dcfce7",ec=GREEN,fs=9)
    box(ax,9.5,1.3,1.3,1.1,"H′\nnew\nfeatures",fc="#fee2e2",ec=RED,fs=9,bold=True)
    for x in [2.1,4.6,7.1,9.3]: arrow(ax,x,1.85,x+0.6,1.85,color=SLATE)
    ax.text(5.5,0.7,"Â = neighbour-mixing (from the graph) · W = what to learn · stack 2–3 layers to reach further",
            ha="center",fontsize=9.5,color=SLATE)
    fig.savefig(os.path.join(OUT,"gcn_layer.png"),bbox_inches="tight"); plt.close(fig)

# 4) gat_vs_gcn.png
def gat_vs_gcn():
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    for a,title,mode in [(ax[0],"GCN: every neighbour weighted the same","gcn"),(ax[1],"GAT: learned attention per neighbour","gat")]:
        a.set_xlim(0,6); a.set_ylim(0,6); a.axis("off"); a.set_aspect("equal")
        ctr=(3,3); nbs=[(1,4.6),(5,4.4),(1.1,1.5),(4.9,1.7)]
        ws=[0.5,0.5,0.5,0.5] if mode=="gcn" else [0.9,0.15,0.75,0.2]
        for (x,y),w in zip(nbs,ws):
            a.add_patch(FancyArrowPatch((x,y),ctr,arrowstyle="-|>",mutation_scale=12,
                        color=AMBER if mode=="gat" else GREY,lw=1+w*6,connectionstyle="arc3,rad=0.12",zorder=1,alpha=0.85))
            node(a,x,y,"",fc="#fde68a" if mode=="gat" else "#e2e8f0",r=0.24,tc=INK)
            if mode=="gat": a.text(x+0.42*(ctr[0]-x),y+0.42*(ctr[1]-y),f"{w:.2f}",fontsize=9,color="#b45309",fontweight="bold")
        node(a,ctr[0],ctr[1],"i",fc=BLUE,r=0.32)
        a.set_title(title,fontsize=11,fontweight="bold")
    fig.suptitle("GCN fixes the neighbour weights; GAT learns them, so it can ignore unhelpful neighbours",
                 fontsize=11.5,fontweight="bold",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"gat_vs_gcn.png"),bbox_inches="tight"); plt.close(fig)

# 5) graph_readout.png
def graph_readout():
    fig,ax=plt.subplots(figsize=(11,3.4)); ax.set_xlim(0,11); ax.set_ylim(0,3.4); ax.axis("off")
    ax.text(5.5,3.15,"Graph classification: node embeddings → pool → one vector → label",
            ha="center",fontsize=12.5,fontweight="bold",color=INK)
    # mini graph
    pos={0:(0.8,1.6),1:(1.5,2.3),2:(1.6,0.9),3:(2.3,1.6)}
    for i,j in [(0,1),(0,2),(1,3),(2,3),(1,2)]: ax.plot([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]],color=GREY,lw=1.6,zorder=1)
    for k,(x,y) in pos.items(): node(ax,x,y,"",fc=BLUE,r=0.16)
    arrow(ax,2.7,1.6,3.5,1.6,color=SLATE)
    box(ax,3.6,1.05,1.9,1.1,"message\npassing\n→ embeddings",fc="#dbeafe",ec=BLUE,fs=9,bold=True)
    arrow(ax,5.6,1.6,6.3,1.6,color=SLATE)
    box(ax,6.4,1.05,1.7,1.1,"mean-pool\n(readout)",fc="#dcfce7",ec=GREEN,fs=9,bold=True)
    arrow(ax,8.2,1.6,8.9,1.6,color=SLATE)
    box(ax,9.0,1.05,1.7,1.1,"classifier\n→ label",fc="#fee2e2",ec=RED,fs=9,bold=True)
    ax.text(5.5,0.4,"pooling makes the output independent of node count and node order",ha="center",fontsize=9.5,color=SLATE)
    fig.savefig(os.path.join(OUT,"graph_readout.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    graph_and_matrix(); message_passing(); gcn_layer(); gat_vs_gcn(); graph_readout()
    print("wrote:", sorted(os.listdir(OUT)))
