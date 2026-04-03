import pypret
import numpy as np

# Set up simulation with 256 bins in frequency (DFT size)

# The grid has a temporal axis with 2.5fs spacing, and freq values correspond with that

ft = pypret.FourierTransform(256, dt=2.5e-15)

# Create a pulse with central wavelength of 800nm

pulse = pypret.Pulse(ft, 800e-9)

# generate pulse with Gaussian spectrum and field standard deviation
# of 20 nm

pulse.spectrum = pypret.lib.gaussian(pulse.wl, x0=800e-9, sigma=20e-9)

# print the accurate FWHM of the temporal intensity envelope

print(pulse.fwhm(dt=pulse.dt/100))

# propagate it through 1cm of BK7 (remove first ord)

phase = np.exp(1.0j * pypret.material.BK7.k(pulse.wl) * 0.01)
pulse.spectrum = pulse.spectrum * phase

# print the temporal FWHM again

print(pulse.fwhm(dt=pulse.dt/100))

# finally plot the pulse

pypret.graphics.PulsePlot(pulse)

# Create FROG pulse and plot

delay = np.linspace(-2.5e-13,2.5e-13,1000)

insertion = np.linspace(-0.025, 0.025, 128)  # insertion in m
pnps = pypret.PNPS(pulse, "frog", "shg")
# calculate the measurement trace
pnps.calculate(pulse.spectrum, delay)
original_spectrum = pulse.spectrum
# and plot it
pypret.MeshDataPlot(pnps.trace)

# Retrieve the original signal using copra

ret = pypret.Retriever(pnps, "copra", verbose=True, maxiter=300)
# start with a Gaussian spectrum with random phase as initial guess
pypret.random_gaussian(pulse, 50e-15, phase_max=0.0)
# now retrieve from the synthetic trace simulated above
ret.retrieve(pnps.trace, pulse.spectrum)
# and print the retrieval results
ret.result(original_spectrum)