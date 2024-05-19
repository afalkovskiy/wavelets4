import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import math

st.set_page_config(layout="wide")

pi = math.pi

st.title('Ormsby wavelet with synthetic trace')
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


def ORMSBY(f1=5., f2=10., f3=40., f4=45., length=0.512, dt=0.001):
    p = np.pi
    t = np.linspace(-length/2, (length-dt)/2, int(length/dt))

    # y = p*p*f4**2 * (np.sinc(f4*t))**2/(p*f4-p*f3) - p*p*f3**2 * (np.sinc(f3*t))**2/(p*f4-p*f3) - \
    #     p*p*f2**2 * (np.sinc(f2*t))**2/(p*f2-p*f1) - p*p*f1**2 * (np.sinc(f1*t))**2/(p*f2-p*f1)
    y = p*f4**2 * (np.sinc(f4*t))**2/(f4-f3) - p*f3**2 * (np.sinc(f3*t))**2/(f4-f3) - \
        p*f2**2 * (np.sinc(f2*t))**2/(f2-f1) - p*f1**2 * (np.sinc(f1*t))**2/(f2-f1)

    y = y / np.amax(abs(y))

    return t, y

# st.title('ORMSBY wavelet')
st.text('Select wavelet parameters')
st.latex(r'''
    Ormsby(t) = \frac{\pi f_4^2 sinc^2 (\pi f_4 t) - \pi f_3^2 sinc^2 (\pi f_3 t)}{f_4 - f_3}  
    - \frac{\pi f_2^2 sinc^2 (\pi f_2 t) - \pi f_1^2 sinc^2 (\pi f_1 t)}{f_2 - f_1}
    ''') 

col100, col200, col300, col400, col500, col600 = st.columns(6)
with col100:
    f1 = st.slider('f1 (Hz)', value=5., min_value=1., max_value=240., step=1., format="%.1f")
    #phi = st.slider('Phase rotation angle (deg)', value=0.0, min_value=0., max_value=360., step=45., format="%.1f")

with col200:
    f2 = st.slider('f2 (Hz)', value=10., min_value=1., max_value=240., step=1., format="%.1f")
    #envelope = st.checkbox('Display wavelet envelope')
    # st.write("Rotate phase:")

with col300:
    f3 = st.slider('f3 (Hz)', value=60., min_value=1., max_value=240., step=1., format="%.1f")

with col400:
    f4 = st.slider('f4 (Hz)', value=70., min_value=1., max_value=240., step=1., format="%.1f")

with col500:
    phi = st.slider('Phase (deg)', value=0.0, min_value=0., max_value=360., step=45., format="%.1f")

with col600:
    envelope = st.checkbox('Envelope')

#st.write(f1, " - ", f2, " - ", f3, " - ", f4)

#st.write("Phi = ", phi)
str1 = "Ormsby " + str(int(f1 + 0.5)) + " - " + str(int(f2 + 0.5))  + " - " + str(int(f3 + 0.5)) + " - " + str(int(f4 + 0.5)) + " Hz, Phase " + str(int(phi+0.5)) + "°"
# st.subheader(str1)

t, y = ORMSBY(f1, f2, f3, f4, 0.512, 0.001)




col1, col2, col3 = st.columns(3)
with col1:
    # phi = st.slider('Phase rotation angle (deg)', value=0.0, min_value=0., max_value=360., step=45., format="%.1f")
    # envelope = st.checkbox('Envelope')

    # str1 = "Wavelet: " + str(int(f + 0.5)) + " Hz, Phase = " + str(int(phi+0.5)) + "°"
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
