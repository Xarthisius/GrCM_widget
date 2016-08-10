#!/usr/bin/python
import os
import pandas
import numpy as np
from scipy import integrate
from bokeh.io import curdoc
from bokeh.charts import Bar
from bokeh.charts.attributes import cat
from bokeh.charts.operations import blend
from bokeh.layouts import widgetbox, column, row
from bokeh.models import (
    CustomJS, ColumnDataSource, Select
)
from bokeh.plotting import figure


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
df = pandas.DataFrame(a)

bar = Bar(df,
          legend=None,
          values=blend(*keys, name='genes', labels_name='gene'),
          stack=cat(columns='gene', sort=False),
          label=cat(columns='CO2', sort=False),
          tooltips=[('gene', '@gene'), ('CO2 Level', '@CO2'),
                    ('final concentration', '@height')],
          title="Genes per CO2 level")

source = ColumnDataSource({'x': t, 'y1': np.array(amb[:, 0]),
                           'y2': np.array(ele[:, 0])})
amb_source = ColumnDataSource(
    dict(zip(keys, [amb[:, i] for i in range(amb.shape[-1])]))
)
ele_source = ColumnDataSource(
    dict(zip(keys, [ele[:, i] for i in range(ele.shape[-1])]))
)
callback_args = dict(source=source, amb=amb_source, ele=ele_source)
callback = CustomJS(args=callback_args, code="""
        var data = source.get('data');
        var f = cb_obj.get('value');
        data['y1'] = amb.get('data')[f];
        data['y2'] = ele.get('data')[f];
        source.trigger('change');
    """)

sizing_mode = 'scale_width'
selector = Select(title='Gene:', value=keys[0], options=keys.tolist(),
                  callback=callback)
# create a new plot
s1 = figure(tools='tap', title=None, sizing_mode=sizing_mode,
            x_axis_type="log", x_axis_label='Time [units]',
            y_axis_label='Concentration')
s1.line('x', 'y1', source=source, color='blue', legend="Ambient CO2")
s1.line('x', 'y2', source=source, color='red', legend="Elevated CO2")
p = row(bar, column(s1, widgetbox(selector, sizing_mode=sizing_mode)))

curdoc().add_root(p)
curdoc().title = 'GrCM'
