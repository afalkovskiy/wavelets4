import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import math

st.set_page_config(layout="wide")

pi = math.pi

st.title('Ricker wavelet with synthetic trace')

col00, col10, col20 = st.columns(3)
with col00:
    st.text('Select model parameters')
with col10:
    dr = st.slider('Reflector interval (sec)', value=0.1, min_value=0.01, max_value=0.5, step=0.01, format="%.2f")
with col20:
    nr = st.number_input('Number of reflectors', min_value=1, max_value=20, value=8, step=1)

# st.write('The number of reflectors is ', nr,'Reflector interval: ', dr)
str0 = "Model: " + str(int(nr)) + " reflectors, distance between reflectors: " + str(dr) + " sec"
st.subheader(str0)

def ricker(f, length=0.512, dt=0.001):
    t = np.linspace(-length/2, (length-dt)/2, int(length/dt))
    y = (1.-2.*(np.pi**2)*(f**2)*(t**2))*np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y

st.text('Select wavelet parameters')
st.latex(r'''
Ricker(t) = (1-2\pi^2 f^2 t^2)e^{-\pi^2 f^2 t^2}
''') 

col100, col200, col300 = st.columns(3)
with col100:
         f = st.slider('Frequency from [1, 240] Hz', value=30., min_value=1., max_value=240., step=1., format="%.1f")   
with col200:
        phi = st.slider('Phase rotation angle (deg)', value=0.0, min_value=0., max_value=360., step=45., format="%.1f")
with col300:
    envelope = st.checkbox('Envelope')    

str1 = "Ricker " + str(int(f + 0.5)) + " Hz, Phase = " + str(int(phi+0.5)) + "°"
# st.subheader(str1)

col1, col2, col3 = st.columns(3)
# with col1:
#     f = st.slider('Select wavelet frequency from [1, 240] Hz', value=30., min_value=1., max_value=240., step=1., format="%.1f")

t, y = ricker (f)

with col1:
    # envelope = st.checkbox('Envelope')
    st.subheader(str1)

    
    z= hilbert(y) #form the analytical signal
    inst_amplitude = np.abs(z) #envelope extraction
    inst_phase = np.unwrap(np.angle(z))#inst phase
    
    phase = phi * pi/180
    x_rotate = math.cos(phase)*z.real - math.sin(phase)*z.imag

with col1:
    if envelope:
        chart_data = pd.DataFrame(
           {
               "t": t,
               #"y": y
               "y": x_rotate,
               "y_env2": inst_amplitude,
               "y_env3": -1*inst_amplitude
           }
        )
        st.line_chart(chart_data, x="t", y=["y", "y_env2", "y_env3"], color=["#d62728", "#D3D3D3", "#D3D3D3"], width=450, height=450)
    
    else:
        chart_data = pd.DataFrame(
           {
               "t": t,
               "y": x_rotate
           }
        )

    st.line_chart(chart_data, x="t", y=["y"], color=["#d62728"])
length1 = 1.0
dt1=0.001
x1 = np.linspace(0, length1, int(length1/dt1))

# x1 = np.arange(0, 2000., 0.5)
# y1 = np.square(x1) -10 * x1
y1 = 0.* x1
# y1[400] = -1.
# y1[500] = 1.
# y2 = np.cos(0.02*x1)
ns = int(dr/dt1)
st.write('dr =', dr, ' dt = ', dt1, ' ns = ', ns)
y1[ns] = -1.
for i in range(nr):
    ni = ns*(i + 1)
    if ni > len(y1) - 1:
        break
    rf = -1       
    if i%6 == 0:
        rf = -1
    if i%6 == 1:
        rf = 1
    if i%6 == 2:
        rf = 1
    if i%6 == 3:
        rf = -0.5  

    if i%6 == 4:
        rf = 0.5  
    if i%6 == 5:
        rf = 1.  
        
    y1[ni] = rf

y2 = np.convolve(y1, x_rotate, mode='same')
y2[0] = 0.

fig1 = plt.figure(figsize=(4,12))
# fig1.suptitle('Reflectivity')

plt.subplot(111)
plt.plot(y1, x1)
plt.gca().invert_yaxis()
# plt.title("Reflectivity")
plt.xlabel("Reflectivity")
plt.ylabel("Two-way time (sec)")

fig2 = plt.figure(figsize=(4,12))
# fig2.suptitle('Convolved')
plt.xlabel("Synthetic trace")
plt.ylabel("Two-way time (sec)")

plt.subplot(111)
plt.plot(y2, x1)
# plt.plot(np.maximum(0,y2), x1)
# plt.plot(y2, np.minimum(0*x1,x1))
# plt.fill_between(y2, np.maximum(0,x1), x1,  color='blue', alpha=.2)
# plt.fill_between(y2, 0*x1, x1,  color='blue', alpha=.2)

# plt.fill_between(x1, np.maximum(0*x1,y2), y2, y2,  color='red', alpha=.4)
# plt.fill_between(x1, np.maximum(0*x1,y2), y2,  color='orange', alpha=.4)
# plt.fill_betweenx(y2, np.maximum(0*y2,y2), 0*x1,  color='blue', alpha=.4)

y2pos = np.maximum(0,y2)
y2pos[10] = 0.
x1[10] = .25

# plt.fill_between(y2pos, x1, 0,  color='green', alpha=.4)
plt.fill_betweenx(x1, y2pos, 0,  color='navy', alpha=.6)
plt.gca().invert_yaxis()

with col2:
    st.subheader('Reflectivity')
    st.pyplot(fig1) 

with col3:
    st.subheader('Synthetic trace')
    st.pyplot(fig2)



url1 = "https://www.rmseismic.com/lasviewer.html"
st.write("More geophysical apps: [rmseismic.com](%s)" % url1)
