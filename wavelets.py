import streamlit as st
from streamlit.components.v1 import html

st.title('Seismic Wavelets v.4')
st.subheader('Alex Falkovskiy')
url = "https://www.rmseismic.com"
st.write("RM Seismic Software [rmseismic.com](%s)" % url)
st.write('The purpose of this web app is to illustrate how basic seismic wavelets change with the change of parameters, and calculate synthetic seisimc.')
st.write('You can also covolve any of these wavelets with simple reflectivity series and calculate a synthetic trace.')
st.write('Wavelet equations used in this app are from a paper by Harold Ryan: Ricker, Ormsby; Klauder, Butterworth - A Choice of wavelets, publised in CSEG Recorder, September 1994.')
st.write('Here is the list of wavelets you can display and calculate a synthetic trace:')
st.write('**Ricker, Ormsby, Klauder, Butterworth**. Use a left side menu to select a wavelet and sliders to change parameters.')

url1 = "https://www.rmseismic.com/lasviewer.html"
st.write("More geophysical apps: [rmseismic.com](%s)" % url1)
st.write("A.F., May 19 2024")

google_js = """
<!-- Google tag (gtag.js) --><script async src="https://www.googletagmanager.com/gtag/js?id=G-VBX865DFKL"></script><script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-VBX865DFKL');
</script>
"""

#html1 = f"{google_js}"
 
