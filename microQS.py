import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import trapz, simps
from scipy import optimize
from scipy.optimize import leastsq
import os
import subprocess


def cont_and_line(reg_EL1,reg_EL2,reg_cont1,reg_cont2,reg_cont3,reg_cont4,spectra_array,lamb,name):
# reg_EL1,2 : limites para hacer zoom al rededor de la linea 
# ref_cont1,2,3,4 : limites para las dos zonas de continuo
# spectra_array : archivo con las lambdas y flujo/cuentas del espectro
	SL1=np.abs(lamb-reg_cont1).argmin()
	SL2=np.abs(lamb-reg_cont2).argmin()
	SL3=np.abs(lamb-reg_cont3).argmin()
	SL4=np.abs(lamb-reg_cont4).argmin()
	SL5 = np.abs(lamb-reg_EL1).argmin()
	SL6 = np.abs(lamb-reg_EL2).argmin()

	merged0 = [*lamb[SL1:SL2],*lamb[SL3:SL4]]
	merged1 = [*spectra_array[SL1:SL2],*spectra_array[SL3:SL4]]

	lamb = lamb[SL5:SL6]
	yline = spectra_array[SL5:SL6]

	fit0,Err = np.polyfit(merged0,merged1,1,cov=True)
	fit = np.poly1d(fit0)

	cont = fit(lamb)
	line = yline-cont

	fig1 = plt.figure()
	plt.plot(lamb,yline,'r--',label=name)
	plt.plot(lamb,cont,'k-',label='fit')
	plt.xlabel('Wavelength [$\AA$]')
	plt.ylabel('Flux [Arbitrary]')
	plt.legend(numpoints=1)
	plt.savefig(name+'.png')
	plt.close(fig1)
	return cont, line , lamb, fit0, [np.sqrt(Err[0][0]),np.sqrt(Err[1][1])],merged0,merged1

def cont_region(sp1,sp2,lamb1,lamb2,factor,line,name,**kwargs):
	x1 = kwargs.get('x1', None)
	x2 = kwargs.get('x2', None)
	x3 = kwargs.get('x3', None)
	x4 = kwargs.get('x4', None)
	if x1 == None:
		if line == 'CIV':
			x1,x2,x3,x4 = 1400,1500,1580,1600
		if line == 'CIII':
			x1,x2,x3,x4 = 1800,1850,2000,2050
		if line == 'MgII':
			x1,x2,x3,x4 = 2600,2650,2800,2850

		print('by default continuum region are between: ',x1,x2,'and: ',x3,x4)

		val='y' #input('do you want to select the continuum region values? y/n ')
		while val == 'y' :
			fig1 = plt.figure()
			plt.plot(lamb1,sp1,'b-',label='A')
			plt.plot(lamb2,sp2*factor,'k-',label='B*'+str(factor))
			plt.fill_between((x1,x2),min(sp2),max(sp1),facecolor='green', alpha=0.1,label='cont')
			plt.fill_between((x3,x4),min(sp2),max(sp1),facecolor='green', alpha=0.1)
			plt.axis([x1,x4, min(sp2*factor),max(sp1)])
			plt.xlabel('Wavelength [$\AA$]')
			plt.ylabel('Flux [Arbitrary]')
			plt.title(line)
			plt.legend(numpoints=1)
			plt.savefig(name+'_'+line+'_cont_region.png')
			plt.show(fig1)
			plt.close(fig1)
			val=input('do you want to select a new continuum region values y/n ')
			if val=='y':
				x1,x2,x3,x4 = eval(input('continuum region: (format: x1,x2,x3,x4) ')) #select the continuum region to be plotted

	cA,lA,xA,pA,EA,merxA,meryA=cont_and_line(x1-50,x4+50,x1,x2,x3,x4,sp1,lamb1,name+'_'+line+'-A') 
	cB,lB,xB,pB,EB,merxB,meryB=cont_and_line(x1-50,x4+50,x1,x2,x3,x4,sp2,lamb2,name+'_'+line+'-B')
	fig2 = plt.figure()
	plt.plot(xA,lA,'k-',xB,lB,'r-')
	z=np.zeros(len(xA))
	plt.plot(xA,z,'k--')
	plt.axis([x1,x4, min(lA)+(min(lA)/20),max(lA)+(max(lA)/10)])
	plt.savefig(name+'_'+str(line)+'lines_W-out_cont.png')
	plt.close(fig2)
	return x1,x2,x3,x4,cA,lA,xA,pA,EA,cB,lB,xB,pB,EB

def integration(intL1,intL2,lamb1,lamb2,file):
#intL1,2 : limites de integracion de la region del espectro
#lamb : archivo con las lambdas de la region a integrar (linea o continuo)
#file : flujo/cuentas de la region a integrar
	SI1 = np.abs(lamb1-intL1).argmin()
	SI2 = np.abs(lamb2-intL2).argmin()
	Method_sum = sum(file[SI1:SI2])
	Method_trapz = trapz(file[SI1:SI2],lamb1[SI1:SI2])
	Method_simps = simps(file[SI1:SI2],lamb2[SI1:SI2])
	return Method_trapz, Method_simps,Method_sum

def INT2(line1,line2,cont1,cont2,lamb1,lamb2,factor,line,shift,intrange,name,**kwargs):
	i1=kwargs.get('int1',None)
	i2=kwargs.get('int2',None)

	if line == 'Lyalpha':
		lcen = 1216
	if line == 'CIV':
		lcen = 1549
	if line == 'CIII':
		lcen = 1909
	if line == 'MgII':
		lcen = 2798

	if i1 == None:
		val='y'
		while val=='y':
		
			fig1 = plt.figure()
			plt.plot(lamb1,line1,'k',label='A')
			plt.plot(lamb2,line2*factor,'b',label='B*'+str(factor))

			ir = intrange/2.0 
			i1 = (lcen+shift) - ir
			i2 = (lcen+shift) + ir
			print('The integration range is: ',i1,i2)
			plt.fill_between((i1,i2),min(line1), max(line1),facecolor='red', alpha=0.3,label='integration')
			plt.axis([min(lamb1),max(lamb1),min(line1),max(line1)])
			plt.xlabel('Wavelength [$\AA$]')
			plt.ylabel('Flux [Arbitrary]')
			plt.title(line)
			plt.legend(numpoints=1)
			plt.savefig(name+'_'+line+'_line_int.png')
			plt.show(fig1)
			plt.close(fig1)
			val=input('do you want to select a different range, shift or multiplication factor? y/n ')
			if val =='y':
				intrange,shift,factor = eval(input('continuum region: (format: range,shift,factor) '))
	fig1 = plt.figure()
	plt.plot(lamb1,line1,'k',label='A')
	plt.plot(lamb2,line2*factor,'b',label='B*'+str(factor))
	plt.fill_between((i1,i2),min(line1)-50, max(line1)+80,facecolor='red', alpha=0.3,label='integration')
	plt.axis([min(lamb1),max(lamb1),min(line1)-0,max(line1)+0])
	plt.xlabel('Wavelength [$\AA$]')
	plt.ylabel('Flux [Arbitrary]')
	plt.title(line)
	plt.legend(numpoints=1)
	plt.savefig(name+'_'+line+'_line_int.png')
	plt.close(fig1)
	_,c1,_=integration(i1,i2,lamb1,lamb2,cont1)
	_,c2,_=integration(i1,i2,lamb1,lamb2,cont2)
	_,l1,_=integration(i1,i2,lamb1,lamb2,line1)
	_,l2,_=integration(i1,i2,lamb1,lamb2,line2)
	return i1,i2,c1,c2,l1,l2

def flujo(b,a,lamb):
	return (b*lamb) + (a/2.0)*(lamb**2)

def error_flujo(db,da,lamb):
	return lamb*db+((da*lamb**2)/2.0)

def MD(p1,Ep1,p2,Ep2,mincont,maxcont,xint1,xint2,lin1,con1,lin2,con2):
#p1,Ep1,p2,Ep2 son los valores de pendientes e interceptos (p) y sus respectivos errores (Ep) entregados por cont_and_line
#mincont y maxcont son los valores extremos elegidos para ajustar el continuo
#xint1,xint2 lambdas entre las que se integro el flujo
#lin1,con1,lin2,con2 los valores de integracion de lineas (lin) y continuos (con)	
	Mlin = -2.5 * np.log10( lin2/lin1 )
	Mcon = -2.5 * np.log10( con2/con1 )
	lamb_difl1 = maxcont-mincont
	FA_l1 = flujo(p1[1],p1[0],lamb_difl1)
	FB_l1 = flujo(p2[1],p2[0],lamb_difl1)
	eFA_l1 = error_flujo(Ep1[1],Ep1[0],lamb_difl1)
	eFB_l1 = error_flujo(Ep2[1],Ep2[0],lamb_difl1)
	e_cont = np.sqrt((eFA_l1/FA_l1)**2 + (eFB_l1/FB_l1)**2)
	e_lin = np.sqrt(2.0)*e_cont
	lamb_cen = (xint1+xint2)/2.0
	return Mlin, e_lin, Mcon, e_cont, lamb_cen


def Gaussians(x, Amp1, cen1, Amp2, cen2, sig1,sig2):
    return Amp1 * np.exp(-(x - cen1)**2 / (2 * sig1**2)) + Amp2 * np.exp(-(x - cen2)**2 / (2 * sig2**2))

def Gx1(x,Amp1,cen1,sig1):
	return Amp1 * np.exp(-(x - cen1)**2 / (2 * sig1**2))

def fitNLR_BLR(lin1,guesses,xlamb,linename):
	def chi(guesses,xlamb):
		return (lin1.flatten() - Gaussians(xlamb, guesses[0], guesses[1], guesses[2], guesses[3], guesses[4],guesses[5]).flatten())
	[A1,x1,A2,x2,s1,s2],_ = optimize.leastsq(chi,guesses,xlamb)
	
	fig3=plt.figure()
	plt.plot(xlamb,lin1,'r-',xlamb,Gaussians(xlamb,A1,x1,A2,x2,s1,s2),'k--',xlamb,Gx1(xlamb,A1,x1,s1),'g--',xlamb,Gx1(xlamb,A2,x2,s2),'b--')
	plt.savefig(linename+'Gaussianfit.png')
	plt.close(fig3)
	
	return A1,x1,A2,x2,s1,s2	

def plot_linearfit(lim1,lim2,lamb,data,yerror,sig,color):
	x0 = np.arange(lim1,lim2)
	fit0,cov =np.polyfit(lamb,data,1,w=1/np.asarray(yerror),cov=True)
	Err = np.sqrt(np.diag(cov))
	fit = np.poly1d(fit0)
	plt.plot(x0,fit(x0),':',color=color)
	plt.fill_between(x0,((fit0[1]-sig*Err[1])+(fit0[0])*x0),((fit0[1]+sig*Err[1])+fit0[0]*x0),facecolor=color, alpha=0.1)
	return fit0,Err*sig,fit

def CM_3lam(function,e_fit,lamb,baseline,e_baseline,dataset):
	print(str(dataset)+' dataset:')
	[print('Magnification at: {:2.2f}  = {:2.2f} +- {:2.2f}'.format(lamb[i],function(lamb[i])-baseline,np.sqrt(e_baseline**2 + e_fit**2))) for i in range(len(lamb))]


def lineprof_paper(lamb,line1,line2,factor,xlabel,ylabel,lminx,lmaxx,lminy,lmaxy,line):
	fig1 = plt.figure(figsize=(7.5,7))
	plt.plot(lamb,line1,'k--')
	plt.plot(lamb,line2*factor,'b-')
	plt.axis([lminx,lmaxx,lminy,lmaxy])
	plt.xlabel(xlabel,fontsize=32)
	plt.ylabel(ylabel,fontsize=32)
	plt.tick_params(axis='both', labelsize=25)
	if max(line1)> 1000:
		plt.ticklabel_format( axis='y', style='scientific',scilimits=(2,2))
	plt.subplots_adjust(left=0.20, bottom=0.190, right=0.920, top=0.905)#, wspace=None, hspace=None)
	plt.savefig(str(line)+'_lineprofile.pdf')
	plt.savefig(str(line)+'_lineprofile.png')
	plt.close(fig1)


def cont_integrations(lamb,cont1,cont2,int_list):
	c1=[]
	c2=[]
	mag_dif = []
	l_cen = []
	i=0
	while i < len(int_list):
		i1 = int_list[i]
		i2 = int_list[i+1]
		_,con1,_=integration(i1,i2,lamb,cont1)
		_,con2,_=integration(i1,i2,lamb,cont2)
		c1.append(con1)
		c2.append(con2)
		mag_dif.append( -2.5 * np.log10( con2/con1 ) )
		l_cen.append((i1+i2)/2)
		i+=2
	return c1,c2,l_cen,mag_dif

def plot_linearfitmin4(lim1,lim2,lamb,data,yerror,sig,color):
	x0 = np.arange(lim1,lim2)
	fit0 =np.polyfit(lamb,data,1,w=1/np.asarray(yerror),full=True)
	Err = np.sqrt(fit0[1]/len(data))
	fit = np.poly1d(fit0[0])
	plt.plot(x0,fit(x0),':',color=color)
	plt.fill_between(x0,fit(x0)-sig*Err,fit(x0)+sig*Err,facecolor=color, alpha=0.1)
	return fit0,Err*sig,fit


def plot_contour(data,step,level,color,colormap,name):
	x = [data[:,0][i:i+step] for i in range(0, len(data), step)]
	y =[data[:,1][i:i+step] for i in range(0, len(data), step)]
	z = [data[:,2][i:i+step] for i in range(0, len(data), step)]
	a = x[0][0]
	b = x[len(x)-1][len(x[0])-1]
	c = y[0][0]
	d = y[len(y)-1][len(y[1])-1]
	plt.axvline(4.0/3.0,color='k')
	CS = plt.contourf(x,y,z, 100, cmap=colormap)
	con=np.exp(1)
	plt.contour(x,y,z,levels=[level*(con)**(-16./8.),level*(con)**(-9./8.),level*(con)**(-4./8.),level*(con)**(-1./8.)],colors=color,linestyles='dashed')
	plt.axis([a, b, c, d])
	plt.xlabel("p",fontsize=12)
	plt.ylabel("Ln(r$_s$)/light-days",fontsize=12)
	plt.savefig(str(name)+'_ADS.png')
	plt.close()


def extinction_func(x,Rv):
    if 0.0 <= x <= 1.1:
        ax = 0.574*x**1.61
        bx = -0.527*x**1.61
        return ax + bx/Rv
    if 1.1 <= x <= 3.3:
        yx = (x-1.82)
        ax = 1 + 0.17699*yx - 0.50447*yx**2 - 0.02427*yx**3 + 0.72085*yx**4 + 0.01979*yx**5 - 0.77530*yx**6 + 0.32999*yx**7
        bx = 1.41338*yx + 2.28305*yx**2 + 1.07233*yx**3 - 5.38434*yx**4 - 0.62251*yx**5 + 5.30260*yx**6 - 2.09002*yx**7
        return ax + bx/Rv
    if 3.3 <= x <= 5.9:
        ax = 1.752 - 0.316*x - 0.104/((x - 4.67)**2 + 0.341)
        bx = -3.090 + 1.825*x + 1.206/((x - 4.62)**2 + 0.263)
        return ax + bx/Rv
    if 5.9 <= x <= 8.0:
        fa = -0.04473*(x - 5.9)**2 - 0.009779*(x - 5.9)**3
        fb = 0.2130*(x - 5.9)**2 + 0.1207*(x - 5.9)**3
        ax = 1.752 - 0.316*x - 0.104/((x - 4.67)**2 + 0.341) + fa
        bx = -3.090 + 1.825*x + 1.206/((x - 4.62)**2 + 0.263) + fb
        return ax + bx/Rv
    if 8.0 <= x <= 10.0:
        ax = -1.073 - 0.628*(x - 8.0) + 0.137*(x - 8.0)**2 - 0.070*(x - 8.0)**3
        bx = 13.670 + 4.257*(x - 8.0) - 0.420*(x - 8.0)**2 + 0.374*(x - 8.0)**3
        return ax + bx/Rv
    else:
        return 0

def ext_plot(M,E,R,M_err,color):
	xext = np.arange(0,10.0,0.05)
	yext= np.asarray( [extinction_func(xext[i],R) for i in range(len(xext))] )
	fun1= M+E*yext**2
	plt.fill_between(xext,fun1-M_err,fun1+M_err,facecolor=color, alpha=0.1)	
	plt.plot(xext,fun1,'--',color=color,linewidth=0.7)

#def chromatic_variations(Mi,DtRF,LRF):
#	''' Mi   : i-band absolute magnitude of a quasar in units of magnitude
#		DtRF : the rest-frame time lag between observations in units of days
#	    LRF  : rest-frame wavelength in units of Å.
#	   '''
#	return (1+0.0024*Mi)*(DtRF/LRF)**(0.3)


def ADS(stat,lev):
	col = np.where( stat[:,1] == min(stat[:,1], key=lambda x:abs(x-lev)))[0][0]
	data = stat[col,:]
	lnrs = data[4]
	errlnrs = data[5]
	r0=np.exp(lnrs)
	rp=np.exp(lnrs+errlnrs)
	rm=np.exp(lnrs-errlnrs)
	print('p = ',data[2],'+-',data[3])
	print('rs = ',r0,'+',rp-r0,'-',r0-rm,'light-days')
	return data, r0, rp-r0, r0-rm

def inter_ADS(data12,mult3,folder):
	dat1 = data12[:,0]
	dat2 = data12[:,1]
	data = np.concatenate(([dat1],[dat2],[mult3]))
	np.savetxt(folder+'/noutput_orden_log.dat',data.T,fmt='%.4f')
	np.savetxt(folder+'/input_stat',data.T,fmt='%.4f')
	subprocess.call(["f77","-o",folder+"/stat2D.e",folder+"/stat2D.f"])
	os.chdir(folder+'/')
	os.system('./stat2D.e > stat.out')
	out = np.loadtxt('stat.out',skiprows= 6)
	return data.T, out, out[0,0]

