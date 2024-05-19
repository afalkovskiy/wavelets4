import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import math
from scipy.signal import butter, lfilter
from scipy import signal
from scipy.signal import freqz

st.set_page_config(layout="wide")

pi = math.pi

st.title('Butterworth wavelet with synthetic trace')
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

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Butter(T=6., lowcut=10., highcut=40., length=0.512, dt=0.001, n=1):
    fs = 500.
    T = 0.4
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    x = signal.unit_impulse(nsamples, [0])
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=n)
    maxAmp = max(y)
    scl = 1/maxAmp

    return t, scl*y


def Klauder(T=6., f1=10., f2=40., length=0.512, dt=0.001):
    k = (f2 - f1)/T
    f0 = (f2 + f1)/2
    i = complex(0, 1)
    p = np.pi
    t = np.linspace(-length/2, (length-dt)/2, int(length/dt))
    #y = t**2

    a = np.exp(2*pi*i*f0*t)
    y = (a * np.sin(pi*k*t*(T - t)) / (pi*k*t)).real

    return t, y


st.text('Select wavelet parameters (wavelet derived by applying Butterworth filter to a unit impulse):')
st.latex(r'''
    G(w) = \frac{1}{\sqrt{1 + w^{2n}}},
    where \; w \; is \; angular \; frequency (rad/sec), \;n \; is \; a \; filter \; order.
    ''')

col100, col200, col300, col400 = st.columns(4)
with col100:
    f1 = st.slider('Lower bandpass frequency (Hz)', value=10., min_value=1., max_value=39., step=1., format="%.1f")

with col200:    
    f2 = st.slider('Upper bandpass frequency (Hz)', value=40., min_value=f1+1., max_value=240., step=1., format="%.1f")
    
with col300:   
    order = st.slider('Order of the Butterworth filter', value=1, min_value=1, max_value=10, step=1, format="%d")

with col400:      
    phi = st.slider('Phase rotation angle (deg)', value=0.0, min_value=0., max_value=360., step=45., format="%.1f")

# with col500:    
#     envelope = st.checkbox('Envelope')
    
#st.write(f1, " - ", f2, "Hz, T =", T, " s")

#f1 = 5
#f2 = 10
T = 6.
t, y = Butter(T, f1, f2, 0.512, 0.001, order)
str1 = "Butterworth wavelet " + str(int(f1 + 0.5)) + " - " + str(int(f2 + 0.5))  + " Hz, " + " Phase " + str(int(phi+0.5)) + "°"
# st.subheader(str1)


col1, col2, col3 = st.columns(3)
   
z= hilbert(y) #form the analytical signal
inst_amplitude = np.abs(z) #envelope extraction
inst_phase = np.unwrap(np.angle(z))#inst phase
    
phase = phi * pi/180
x_rotate = math.cos(phase)*z.real - math.sin(phase)*z.imag

with col1:
    st.subheader(str1)

    chart_data = pd.DataFrame(
        {
            "t": t,
            "y": x_rotate
        }
    )

    st.line_chart(chart_data, x="t", y=["y"], color=["#d62728"])


    st.subheader("Butterworth filter")
    fs = 500.
    b, a = butter_bandpass(f1, f2, fs, order)
    w, h = freqz(b, a, worN=2000)

    chart_data2 = pd.DataFrame(
        {
            "Frequency (Hz)": (fs * 0.5 / np.pi) * w,
            "Density": abs(h)
        } 
    )  
    st.line_chart(chart_data2, x="Frequency (Hz)", y=["Density"], color=["#d62728"])
    
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
N = len(y1)
y2 = np.convolve(y1, x_rotate, mode='full')[:N]

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


y2pos = np.maximum(0,y2)
# y2pos[10] = 0.
# x1[10] = .25

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
