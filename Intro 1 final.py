# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:38:29 2021

@author: Olivia Keene
"""
import numpy as np
import matplotlib.pyplot as plt

#loading IV data for each LED
rv,ri=np.loadtxt("Data\Intro 1\RedVI.txt",unpack=True)
ov,oi=np.loadtxt("Data\Intro 1\OrangeVI.txt",unpack=True)
yv,yi=np.loadtxt("Data\Intro 1\YellowVI.txt",unpack=True)
gv,gi=np.loadtxt("Data\Intro 1\GreenVI.txt",unpack=True)
bv,bi=np.loadtxt("Data\Intro 1\BlueVI.txt",unpack=True)
vv,vi=np.loadtxt("Data\Intro 1\VioletVI.txt",unpack=True)
#calculating voltage errors
errors_r=0.08*rv
errors_o=0.08*ov
errors_y=0.08*yv
errors_g=0.08*gv
errors_b=0.08*bv
errors_v=0.08*vv
errors_ir=0.1*ri
errors_io=0.1*oi
errors_iy=0.1*yi
errors_ig=0.1*gi
errors_ib=0.1*bi
errors_iv=0.1*vi
#producing fit data for the linear regions of each graph
fit_r,cov_r = np.polyfit(rv,ri,1,w=1/(errors_r+errors_ir),cov=True)#add w=1/errors when you have defined errors
fit_o,cov_o = np.polyfit(ov,oi,1,w=1/(errors_o+errors_io),cov=True)
fit_y,cov_y = np.polyfit(yv,yi,1,w=1/(errors_y+errors_iy),cov=True)
fit_g,cov_g = np.polyfit(gv,gi,1,w=1/(errors_g+errors_ig),cov=True)
fit_b,cov_b = np.polyfit(bv,bi,1,w=1/(errors_b+errors_ib),cov=True)
fit_v,cov_v = np.polyfit(vv,vi,1,w=1/(errors_v+errors_iv),cov=True)
#creating functions for trendlines for linear regions
pr=np.poly1d(fit_r)
po=np.poly1d(fit_o)
py=np.poly1d(fit_y)
pg=np.poly1d(fit_g)
pb=np.poly1d(fit_b)
pv=np.poly1d(fit_v)
#organising arrays of the gradients and y-intercepts of each LED's trendline
iv_gradients=np.array([fit_r[0],fit_o[0],fit_y[0],fit_g[0],fit_b[0],fit_v[0]])
iv_yintercepts=np.array([fit_r[1],fit_o[1],fit_y[1],fit_g[1],fit_b[1],fit_v[1]])
#%%
#plot Red I-V graph
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Red I-V Plot") 
plt.errorbar(rv,ri,xerr=errors_r,yerr=errors_ir, fmt='b.', mew=2, ms=3, capsize=2.5)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.plot(rv,pr(rv),color='orange')
plt.plot(0,0,'b.',ms=4)
plt.savefig("Output/Red_plot.png")
#%%
#plot Orange I-V graph
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Orange I-V Plot") 
plt.errorbar(ov,oi,xerr=errors_o,yerr=errors_io, fmt='b.', mew=2, ms=3, capsize=2.5)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.plot(ov,po(ov),color='orange')
plt.plot(0,0,'b.',ms=4)
plt.savefig("Output/Orange_plot.png")
#%%
#plot Yellow I-V graph
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Yellow I-V Plot") 
plt.errorbar(yv,yi,xerr=errors_y,yerr=errors_iy, fmt='b.', mew=2, ms=3, capsize=2.5)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.plot(yv,py(yv),color='orange')
plt.plot(0,0,'b.',ms=4)
plt.savefig("Output/Yellow_plot.png")
#%%
#plot Green I-V graph
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Green I-V Plot") 
plt.errorbar(gv,gi,xerr=errors_g,yerr=errors_ig, fmt='b.', mew=2, ms=3, capsize=2.5)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.plot(gv,pg(gv),color='orange')
plt.plot(0,0,'b.',ms=4)
plt.savefig("Output/Green_plot.png")
#%%
#plot Blue I-V graph
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Blue I-V Plot") 
plt.errorbar(bv,bi,xerr=errors_b,yerr=errors_ib, fmt='b.', mew=2, ms=3, capsize=2.5)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.plot(bv,pb(bv),color='orange')
plt.plot(0,0,'b.',ms=4)
plt.savefig("Output/Blue_plot.png")
#%%
#plot Violet I-V graph
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Violet I-V Plot") 
plt.errorbar(vv,vi,xerr=errors_v,yerr=errors_iv, fmt='b.', mew=2, ms=3, capsize=2.5)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.plot(vv,pv(vv),color='orange')
plt.plot(0,0,'b.',ms=4)
plt.savefig("Output/Violet_plot.png")
#%%
#Creating my subplot
plt.subplot(3,2,1)
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Red I-V Plot") 
plt.errorbar(rv,ri,xerr=errors_r,yerr=errors_ir, fmt='bo', mew=2, ms=3, capsize=4)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.yticks(np.arange(0,0.035,0.005))
plt.plot(rv,pr(rv),color='orange')
plt.plot(0,0,'bo',ms=4)

plt.subplot(3,2,2)
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Orange I-V Plot") 
plt.errorbar(ov,oi,xerr=errors_o, yerr=errors_io,fmt='bo', mew=2, ms=3, capsize=4)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.yticks(np.arange(0,0.035,0.005))
plt.plot(ov,po(ov),color='orange')
plt.plot(0,0,'bo',ms=4)

plt.subplot(3,2,3)
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Yellow I-V Plot") 
plt.errorbar(yv,yi,xerr=errors_y, yerr=errors_iy,fmt='bo', mew=2, ms=3, capsize=4)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.yticks(np.arange(0,0.035,0.005))
plt.plot(yv,py(yv),color='orange')
plt.plot(0,0,'bo',ms=4)

plt.subplot(3,2,4)
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Green I-V Plot") 
plt.errorbar(gv,gi,xerr=errors_g,yerr=errors_ig, fmt='bo', mew=2, ms=3, capsize=4)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.yticks(np.arange(0,0.035,0.005))
plt.plot(gv,pg(gv),color='orange')
plt.plot(0,0,'bo',ms=4)

plt.subplot(3,2,5)
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Blue I-V Plot") 
plt.errorbar(bv,bi,xerr=errors_b,yerr=errors_ib, fmt='bo', mew=2, ms=3, capsize=4)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.yticks(np.arange(0,0.035,0.005))
plt.plot(bv,pb(bv),color='orange')
plt.plot(0,0,'bo',ms=4)

plt.subplot(3,2,6)
plt.grid()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (A)") 
plt.title("Violet I-V Plot") 
plt.errorbar(vv,vi,xerr=errors_v,yerr=errors_iv, fmt='bo', mew=2, ms=3, capsize=4)
plt.xticks(np.arange(0, 4.0, 0.5))
plt.yticks(np.arange(0,0.035,0.005))
plt.plot(vv,pv(vv),color='orange')
plt.plot(0,0,'bo',ms=4)

plt.tight_layout()
plt.savefig("Output/Allplots_subplot.png")
plt.show()
#%%
#Using the polyfit uncertainty data to find uncertainties for each LED's trendlien gradient
graderror_r = np.sqrt(cov_r[0,0])
graderror_o = np.sqrt(cov_o[0,0])
graderror_y = np.sqrt(cov_y[0,0])
graderror_g = np.sqrt(cov_g[0,0])
graderror_b = np.sqrt(cov_b[0,0])
graderror_v = np.sqrt(cov_v[0,0])
grad_errors=np.array([graderror_r,graderror_o,graderror_y,graderror_g,graderror_b,graderror_v])

#calculating voltage threshold for each LED
v_ts=-iv_yintercepts/iv_gradients

#Using the polyfit uncertainty data to find uncertainties for each LED's trendline y-intercept
yint_errors=np.array(np.sqrt([cov_r[1,1],cov_o[1,1],cov_y[1,1],cov_g[1,1],cov_b[1,1],cov_v[1,1]]))
#Converting to relative uncertainties
rel_grad_errors=grad_errors/iv_gradients
rel_yint_errors=yint_errors/iv_yintercepts
#Propagating uncertainties to find uncertainties for voltage threshold values
vt_errors=(rel_grad_errors**2+rel_yint_errors**2)**0.5

#Loading inverse wavelength data and uncertainties (uncertainties calculated from FWHM in excel)
inv_wavelength=np.loadtxt("Data\Intro 1\wavelength.txt")
inv_wavelength_errors=np.loadtxt("Data\Intro 1\wavelength uncertainties.txt")

#Combining errors simply in order to weight points on final graph
point_errors=inv_wavelength_errors+vt_errors
#Fitting final graph 
fit_vt,cov_vt = np.polyfit(inv_wavelength,v_ts,1,w=1/point_errors,cov=True)
pvt=np.poly1d(fit_vt)

#%%
#plotting final graph
plt.grid()
plt.xlabel("1/Wavelength (m^-1)") 
plt.ylabel("Voltage Threshold (V)") 
plt.title("Graph to determine h") 
plt.xticks(np.arange(1.0E6,3.0E6,2.5E4))
plt.xlim(1.25E6,2.75E6)
plt.yticks(np.arange(0,3.5,0.5))
plt.ylim(1,3.5)
plt.plot(inv_wavelength,pvt(inv_wavelength))
plt.errorbar(inv_wavelength,v_ts,xerr=inv_wavelength_errors,yerr=vt_errors,fmt='bo',mew=2,ms=3,capsize=4)
plt.show()








plt.grid()
plt.xlabel("1/Wavelength ($\mathregular{m^{-1}}$)") 
plt.ylabel("Voltage Threshold (V)") 
plt.title("Graph to determine h") 
plt.xticks(np.arange(1.0E6,3.0E6,2.5E5))
plt.xlim(1.25E6,2.75E6)
plt.yticks(np.arange(0,3.5,0.5))
plt.ylim(1,3.5)
plt.errorbar(inv_wavelength,v_ts,xerr=inv_wavelength_errors,yerr=vt_errors,fmt='b.',mew=2,ms=3,capsize=2.5)
plt.plot(inv_wavelength,pvt(inv_wavelength),'-')
plt.savefig('Output/Final graph to determine h')
plt.show()
#%%
#print final gradient
print(pvt[1])
#calculating final h value and uncertainty from gradient and fit gradient uncertainty respectively
h_value=pvt[1]*1.60217662e-19/299792458
h_error=np.sqrt(cov_vt[0,0])*1.60217662e-19/299792458
#printing final results
print("%.3e ± %.3e Js" %(h_value,h_error))