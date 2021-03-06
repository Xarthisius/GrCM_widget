#!/usr/bin/python
import json
import os
import pandas
import numpy as np
from scipy import integrate
from bokeh.io import curdoc
from bokeh.charts import output_file
from bokeh.layouts import row
from bokeh.models import (
    Arrow, OpenHead, CustomJS, ColumnDataSource, HoverTool, TapTool
)
from bokeh.plotting import figure


def get_graph(cyjs_file):
    with open(cyjs_file, 'r') as fh:
        data = json.loads(fh.read())

    nodes = {}
    for node in data['elements']['nodes']:
        nodes[node['data']['id']] = dict(
            x=node['position']['x'], y=node['position']['y'],
            name=node['data']['name'])

    edges = []
    for edge in data['elements']['edges']:
        esource = nodes[edge['data']['source']]
        etarget = nodes[edge['data']['target']]
        inter = edge['data']['interaction']
        arrow = 'Reg' in inter or 'reg' in inter
        if inter.startswith('Neg'):
            color = 'blue'
        elif inter.startswith('Pos'):
            color = 'red'
        else:
            color = 'black'
        edges.append(dict(
            xs=esource['x'], ys=esource['y'], xe=etarget['x'], ye=etarget['y'],
            arrow=arrow, color=color))
    return nodes, edges


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

nodes, edges = get_graph('Input/example.cyjs')

with open('Input/NetworkListWithTFAnnotation.txt', 'r') as fh:
    tfdata = fh.read().split()
tf_or_not = dict(zip(tfdata[::2], tfdata[1::2]))

tf_sources = dict(
    x=[], y=[], name=[], color=[], label=[], mRNA_e=[], mRNA_a=[]
)
mg_sources = dict(
    x=[], y=[], name=[], color=[], label=[], mRNA_e=[], mRNA_a=[]
)
df = data_static  # shortcut
for gene in nodes.values():
    name = gene['name']
    if tf_or_not[name] == 'TF':
        d = tf_sources
    else:
        d = mg_sources
    try:
        gene_amb = amb_source.data[name][-1]
        gene_ele = ele_source.data[name][-1]
        mRNA_e = df.loc[df.Glyma_ID == name, 'mRNA_ele'].values[0]
        mRNA_a = df.loc[df.Glyma_ID == name, 'mRNA_Amb'].values[0]
        if gene_ele > gene_amb:
            color = 'red'
            label = 'mRNA higher for elevated CO2'
        else:
            color = 'blue'
            label = 'mRNA higher for ambient CO2'
    except KeyError:
        color = 'black'
        label = 'mRNA unknown'
        mRNA_e = mRNA_a = -1.0
    d['x'].append(gene['x'])
    d['y'].append(gene['y'])
    d['name'].append(name)
    d['color'].append(color)
    d['label'].append(label)
    d['mRNA_e'].append(mRNA_e)
    d['mRNA_a'].append(mRNA_a)

tf_sources = ColumnDataSource(tf_sources)
mg_sources = ColumnDataSource(mg_sources)

hover = HoverTool(tooltips=[
    ('name', '@name'), ('id', '$index'),
    ('mRNA level (elevated CO2)', '@mRNA_e'),
    ('mRNA level (ambient CO2)', '@mRNA_a')])
net = figure(sizing_mode=sizing_mode,
             tools=['tap', hover, 'box_zoom', 'reset'])
r_circles = net.circle('x', 'y', source=mg_sources, size=10,
                       color='color', level='overlay')
r_triang = net.triangle('x', 'y', source=tf_sources, size=10,
                        color='color', level='overlay')

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

# add arrows to show interactions
for edge in edges:
    color = edge['color']
    if edge['arrow']:
        head = OpenHead(line_color=color, line_width=1.5, size=8)
        line_dash = [0, 0]
    else:
        head = None
        line_dash = [4, 4]
    xs, ys = edge['xs'], edge['ys']
    xe, ye = edge['xe'], edge['ye']
    net.add_layout(
        Arrow(end=head, x_start=xs, y_start=ys, x_end=xe, y_end=ye,
              line_color=color, line_dash=line_dash)
    )

# create legend manually
net.circle([0], [0], color='red', legend='mRNA higher for elevated CO2',
           size=0)
net.circle([0], [0], color='blue', legend='mRNA higher for ambient CO2',
           size=0)
net.circle([0], [0], color='black', legend='mRNA unknown', size=0)
net.circle([0], [0], color='gray', fill_color='white',
           legend='Metabolic gene', size=0)
net.triangle([0], [0], color='gray', fill_color='white',
             legend='Transcription factor', size=0)
net.line([0, 0.1], [0, 0.1], color='red', legend='Positive correlation')
net.line([0, 0.1], [0, 0.1], color='blue', legend='Negative correlation')
net.line([0, 0.1], [0, 0.1], color='black', legend='Just regulation')
net.line([0, 0.1], [0, 0.1], color='gray',
         line_dash=[4, 4], legend='Correlation')
net.line([0, 0.1], [0, 0.1], color='gray',
         legend='Correlation with Regulation')

p = row(net, s1)

curdoc().add_root(p)
curdoc().title = 'GrCM'
