import os,sys,json
import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np

def use_notebook():
    from bokeh.io import output_notebook
    output_notebook()

def is_url(x): return type(x)==str and x.strip().startswith('http')
def is_path(x): 
    return type(x) == str and os.path.exists(x)
def is_graph(x): return type(x) in {nx.Graph, nx.DiGraph}

def tupper(x): return x[0].upper()+x[1:]


def condense_booknlp_output(df=None,url=None):
    if df is None and url: df=pd.read_csv(url)
    if df is None: return None
    
    nameld=[]
    gby='name_real'
    other_cols='gender	race	class	other	notes'.split()
    for name,namedf in df.groupby(gby):
        #names=  ', '.join(tupper(x) for x in namedf.names)
        names = {tupper(nm.strip())
                 for nms in namedf.names
                 for nm in nms.split(',')
                 if nm.strip()
                }
        
        
        namedx={}
        namedx['ID']=tupper(name)
        namedx['Label']=tupper(name)
        namedx['Names'] = ', '.join(names)
        namedx['Num'] = sum(namedf.num)
        for oc in other_cols: namedx[oc]=namedf[oc].iloc[0]
        nameld.append(namedx)
    newdf=pd.DataFrame(nameld).sort_values('Num',ascending=False).fillna('')
    newdf['Rank']=[i+1 for i,x in enumerate(newdf.index)]
    return newdf






class Hairball():
    def __init__(self, url_or_path_or_g = None, is_directed=False):
        # init
        self.df_nodes = None
        self.df_edges = None
        self.is_directed = is_directed
        self.g = None

        # boot?
        if is_url(url_or_path_or_g): self.from_url(url_or_path_or_g)
        elif is_path(url_or_path_or_g): self.from_url(url_or_path_or_g)
        elif is_graph(url_or_path_or_g): self.from_nx(url_or_path_or_g)
    
    def from_nx(self,g):
        self.g=g
        # make tables
        self.nx2df()
    
        
    def get_url_or_path(self,url_or_path,tmpfn='/tmp/hairball.download_url.data'):
        # download
        path=None
        print(f'Downloading URL ({url_or_path[:10]}...{url_or_path[-10:]})')
        if url_or_path.startswith('http'):
            import urllib
            urllib.request.urlretrieve(url_or_path, tmpfn)
            path = tmpfn
        else:
            path = url_or_path
        return path
    
    def from_url(self,url_or_path):
        if 'output=ods' in url_or_path or url_or_path.endswith('.ods'):
            return self.from_ods(url_or_path)
        if 'output=csv' in url_or_path or url_or_path.endswith('.csv'):
            return self.from_csv(url_or_path)
    
    def from_csv(self,url_or_path,col_id='id',col_edges='rels'):
        ### Format
        path = self.get_url_or_path(url_or_path)
        
        self.df_nodes = df = pd.read_csv(path).set_index(col_id)
        
        # make edges
        eld = []
        for idx,row in df.iterrows():
            edgestr = str(row[col_edges])
            
            # @TODO can only handle initial snapshot
            edgestr = edgestr.split('-->')[-1].strip()
            
            # loop over each one
            for e in edgestr.split(';'):
                if not ')' in e: continue
                reltype,etrgt=e.replace('(','').strip().split(')',1)
                etrgtmeta=''
                if '[' in etrgt:
                    etrgt,etrgtmeta=etrgt.split('[',1)
                    etrgtmeta.replace(']','').strip()
                etrgt=etrgt.strip()
                for etrgtx in etrgt.split(','):
                    edx = {'source':idx, 'target':etrgtx.strip(), 'reltype':reltype.strip(), 'meta':etrgtmeta}
                    eld.append(edx)
        self.df_edges = pd.DataFrame(eld)
        # from df
        self.df2nx()
        
    
    def from_ods(self,url_or_path):
        from pandas_ods_reader import read_ods
        path = self.get_url_or_path(url_or_path)

        # nodes are in first sheet, edges next
        self.df_nodes = read_ods(path, 1).set_index('id')
        self.df_edges = read_ods(path, 2)
        
        self.df2nx()
        
    def nx2df(self):
        # make node table
        self.df_nodes = pd.DataFrame([
            {
                **{'id':n},
                **d
            }
            for n,d in tqdm(self.g.nodes(data=True),desc='Building df_nodes from g')
        ]).set_index('id')
        
        # make edge table
        self.df_edges = pd.DataFrame([
            {
                **{'source':a, 'target':b},
                **d
            }
            for a,b,d in tqdm(self.g.edges(data=True), desc='Building df_edges from g')
        ])
        
    def df2nx(self):
        if self.df_nodes is None or self.df_edges is None: return
        
        # add nodes
        g=self.g=nx.Graph() if not self.is_directed else nx.DiGraph()
        for row in tqdm(self.df_nodes.reset_index().to_dict(orient="records"),desc='Generating graph nodes from df_nodes'):
            idx=row['id']
            g.add_node(idx, **row)
        
        # add edges
        for row in tqdm(self.df_edges.reset_index().to_dict(orient="records"),desc='Generating graph edges from df_edges'):
            idx1=row['source']
            idx2=row['target']
            g.add_edge(idx1, idx2, **row)
        
        print(f'Generated graph: {g.order()} nodes, {g.size()} edges')
        
        
    def gen_stats(self,stats = ['degree', 'betweenness_centrality']):
        for st in stats:
            func=getattr(nx,st)
            res=func(self.g)
            self.df_nodes[st]=[res[idx] for idx in self.df_nodes.index]
        self.df2nx()
            
    
    @property
    def g_bokeh(self):
        from bokeh.plotting import from_networkx
        return from_networkx(self.g, nx.spring_layout, scale=10, center=(0, 0))
    
    def draw_bokeh(self,
        title='Networkx Graph', 
        save_to=None,
        color_by=None,
        size_by=None,
        default_color='skyblue',
        default_size=15,
        min_size=5,
        max_size=30,
    ):
        from bokeh.io import output_notebook, show, save
        from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet
        from bokeh.plotting import figure
        from bokeh.plotting import from_networkx
        from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
        from bokeh.transform import linear_cmap
        from networkx.algorithms import community

        #Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [("ID", "@index")]#, ("Relations")]

        #Create a plot — set dimensions, toolbar, and title
        # possible tools are pan, xpan, ypan, xwheel_pan, ywheel_pan, wheel_zoom, xwheel_zoom, ywheel_zoom, zoom_in, xzoom_in, yzoom_in, zoom_out, xzoom_out, yzoom_out, click, tap, crosshair, box_select, xbox_select, ybox_select, poly_select, lasso_select, box_zoom, xbox_zoom, ybox_zoom, save, undo, redo, reset, help, box_edit, line_edit, point_draw, poly_draw, poly_edit or hover
        plot = figure(
            tooltips = HOVER_TOOLTIPS,
            tools="pan,wheel_zoom,save,reset,point_draw",
#             active_scroll='wheel_zoom',
#             tools="",
            x_range=Range1d(-10.1, 10.1),
            y_range=Range1d(-10.1, 10.1),
            title=title
        )

        #Create a network graph object with spring layout
        # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html

        #Set node size and color
        
        # size?
        size_opt = default_size
        if size_by is not None:
            size_opt = '_size'
            data_l = X = np.array([d.get(size_by,0) for n,d in self.g.nodes(data=True)])
            data_l_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            data_scaled = [(min_size + (max_size * x)) for x in data_l_norm]
            for x,n in zip(data_scaled, self.g.nodes()):
                self.g.nodes[n]['_size']=x
                

        # get network
        network_graph = self.g_bokeh
        
        
        # render nodes
        network_graph.node_renderer.glyph = Circle(
            size=size_opt, 
            fill_color=color_by if color_by is not None else default_color
        )

        #Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

        #Add network graph to the plot
        plot.renderers.append(network_graph)

        #Add Labels
        x, y = zip(*network_graph.layout_provider.graph_layout.values())
        node_labels = list(self.g.nodes())
        source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
        labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)
        plot.renderers.append(labels)

        show(plot)
        if save_to: save(plot, filename=save_to)