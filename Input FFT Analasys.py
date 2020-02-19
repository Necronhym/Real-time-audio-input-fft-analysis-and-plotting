import pyaudio
import numpy
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
RATE=44100
RECORD_SECONDS = 1
CHUNKSIZE = 1024	
# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
def getAudio(i):
    #Gets Data:
    frames = [] # A python-list of chunks(numpy.ndarray)
    for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
        data = stream.read(CHUNKSIZE)
        frames.append(numpy.frombuffer(data, dtype=numpy.int16))
    #Convert the list of numpy-arrays into a 1D array (column-wise)
    numpydata = numpy.hstack(frames)
    #fftw	
    #Performs FFT on array
    fftData = sp.fft.fft(numpydata)
    #Gets length of sample for data
    fftDataLen = len(fftData)/2 
    #Plots
    plt.subplot(2,1,1)
    plt.cla()
    plt.plot(numpydata)
    plt.subplot(2,1,2)
    plt.cla()
    plt.plot(abs(fftData[0:int(fftDataLen-1)]),'r')
ani = FuncAnimation(plt.gcf(), getAudio)
#Names the file
plt.show()
# close stream
stream.stop_stream()
stream.close()
p.terminate()
