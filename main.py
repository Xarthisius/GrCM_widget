#!/usr/bin/python
import os
import pandas
import numpy as np
import networkx as nx
from scipy import integrate
from bokeh.io import curdoc
from bokeh.charts import output_file
from bokeh.layouts import row
from bokeh.models import (
    CustomJS, ColumnDataSource, HoverTool, TapTool
)
from bokeh.plotting import figure


def get_graph(sif_file):
    data = np.loadtxt(sif_file, dtype='S')

    interaction_types = [_.decode() for _ in np.unique(data[:, 1])]
    nodes = [_.decode() for _ in np.unique(
        np.concatenate((data[:, 0], data[:, 2])))]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    c = ['black', 'blue', 'red', 'yellow', 'green']
    for i, itype in enumerate(interaction_types):
        ind = np.where(data[:, 1] == itype.encode())[0]
        # color = np.random.random((3,))
        color = c[i]
        start = [_.decode() for _ in data[ind, 0]]
        end = [_.decode() for _ in data[ind, 2]]
        G.add_edges_from(zip(start, end), color=color)

    # values = [0.25 for node in G.nodes()]
    return nx.spring_layout(G), G


def Protein_translation_RNA(t, y, L, U, D, mRNA):
    """
    Defines ODE function Conversion of Amb_mRNA to protein
    p1,p2,p3....: Protein concentrations for all ODEs
    It will have list of all parameter values for all my ODEs, so 36 values :
    L, U, D for each mRNA to protein conversion equation

    """
    # Output from ODE function must be a COLUMN vector, with n rows
    return (L * mRNA) / (1.0 + y / D) - U * y


"""
    Have the ODE functions called and have the initial values of time, end
    time, dt value and initial values for which we are running ODE solver.
    # Start by specifying the integrator: use ``vode`` with "backward
    # differentiation formula"
"""


def get_levels(data_static, data_inp,
               t_start=0.0, t_final=200.0, delta_t=0.1):

    t = np.arange(t_start, t_final + delta_t, delta_t)
    # Number of time steps: 1 extra for initial condition
    num_steps = t.shape[0]

    # Set initial condition(s): for integrating variable and time! Get the
    # initial protein concentrations from Yu's file
    initial_protein_conc = data_static["Initial_protein_content"]

    ###
    L = data_inp["L"]  # protein synthesis rate per day
    U = data_inp["U"]  # protein degradation rate per day
    # factor affecting feedback from protein concentration to rate of protein
    # synthesis from mRNA
    D = data_inp["D"]

    integrator = integrate.ode(Protein_translation_RNA).set_integrator(
        'vode', method='bdf')

    result = ()

    for key in ['mRNA_Amb', 'mRNA_ele']:
        mRNA = data_static[key]
        integrator.set_initial_value(initial_protein_conc,
                                     t_start).set_f_params(L, U, D, mRNA)

        # Integrate the ODE(s) across each delta_t timestep
        k = 1
        temp = integrator.y[:]
        while integrator.successful() and k < num_steps:
            integrator.integrate(integrator.t + delta_t)
            temp = np.vstack((temp, integrator.y[:]))
            k += 1

        result += (temp[:],)
    return result

output_file("GrCM.html")

TDIR = os.path.dirname(os.path.abspath(__file__))
data_static = pandas.read_csv(
    os.path.join(TDIR, "Input", "GrCM_static.txt"), sep="\t")
data_inp = pandas.read_csv(
    os.path.join(TDIR, "Input", "GrCM_input.txt"), sep="\t")

t_start = 0.0
t_final = 200.0
delta_t = 0.1
t = np.arange(t_start, t_final + delta_t, delta_t)
amb, ele = get_levels(data_static, data_inp)
data_inp["amb"] = amb[-1, :]
data_inp["ele"] = ele[-1, :]

keys = data_inp['Glyma_ID'].values.astype(str)
a = dict(zip(keys, np.vstack((amb[-1, :], ele[-1, :])).T))
a["CO2"] = ["Ambient", "Elevated"]

source = ColumnDataSource({'x': t, 'y1': np.array(amb[:, 0]),
                           'y2': np.array(ele[:, 0])})
amb_source = ColumnDataSource(
    dict(zip(keys, [amb[:, i] for i in range(amb.shape[-1])]))
)
ele_source = ColumnDataSource(
    dict(zip(keys, [ele[:, i] for i in range(ele.shape[-1])]))
)
callback_args = dict(source=source, amb=amb_source, ele=ele_source)

sizing_mode = 'scale_width'
# create a new plot
s1 = figure(tools='tap', title=None, sizing_mode=sizing_mode,
            x_axis_type="log", x_axis_label='Time [units]',
            y_axis_label='Concentration')
s1.line('x', 'y1', source=source, color='blue', legend="Ambient CO2")
s1.line('x', 'y2', source=source, color='red', legend="Elevated CO2")

layout, graph = get_graph(os.path.join(TDIR, 'Input', 'example.sif'))
nodes, nodes_coordinates = zip(*sorted(layout.items()))
nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
nodes_source = ColumnDataSource(dict(x=nodes_xs, y=nodes_ys,
                                     name=nodes))
d = dict(xs=[], ys=[], color=[])
for (u, v), color in nx.get_edge_attributes(graph, 'color').items():
    d['xs'].append([layout[u][0], layout[v][0]])
    d['ys'].append([layout[u][1], layout[v][1]])
    d['color'].append(color)
lines_source = ColumnDataSource(d)

hover = HoverTool(tooltips=[('name', '@name'), ('id', '$index')])
net = figure(sizing_mode=sizing_mode,
             tools=['tap', hover, 'box_zoom', 'reset'])
r_circles = net.circle('x', 'y', source=nodes_source, size=10,
                       color='blue', level='overlay')

taptool = net.select(type=TapTool)
callback_args = dict(source=source, amb=amb_source, ele=ele_source)
callback = CustomJS(args=callback_args, code="""
    var data = source.get('data');
    var id = cb_obj.selected['1d'].indices[0];
    var f = cb_obj.properties.data.spec.value.name[id];
    id.length;
    data['y1'] = amb.get('data')[f];
    data['y2'] = ele.get('data')[f];
    source.trigger('change');
""")
taptool.callback = callback

r_lines = net.multi_line('xs', 'ys', line_width=2.5,
                         line_color='color',
                         source=lines_source)

p = row(net, s1)

curdoc().add_root(p)
curdoc().title = 'GrCM'
