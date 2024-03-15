import matplotlib.pyplot as plt
import numpy as np
import time as time

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(x + phase))
    fig.canvas.draw() # draw new contents
    fig.canvas.flush_events() 
    #time.sleep(2)
    #plt.close(fig) # close new figure
    #time.sleep(2)
