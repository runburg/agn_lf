import numpy
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ticks_font = FontProperties(family='times new roman', size=12, weight='normal',stretch='normal')
import astropy.io.fits as fits


fig = plt.figure(figsize=(10.0,5.0))

# get swire data and data from L13
hdu1=fits.open('SWIRE_not_XSERVS_in_overlap.fits')
#hdu2=fits.open('L13_table3.fits')
hdu3=fits.open('SWIRE_and_XSERVS.fits')

# form the flux ratios
data1=hdu1[1].data
#data2=hdu2[1].data
data3=hdu3[1].data

#lacy wedge colors, SWIRE_not_XSERVS
fr5p8o3p6=log10(data1['flux_ap2_58']/data1['flux_ap2_36'])
fr8p0o4p5=log10(data1['flux_ap2_80']/data1['flux_ap2_45'])

#stern wedge colors, SWIRE
md3p6m4p5=-2.5*log10(data1['flux_ap2_36']/data1['flux_ap2_45'])+0.485
md5p8m8p0=-2.5*log10(data1['flux_ap2_58']/data1['flux_ap2_80'])+0.634

#same for SWIRE_and_XSERVS
#lacy wedge colors, SWIRE
xsfr5p8o3p6=log10(data3['flux_ap2_58']/data3['flux_ap2_36'])
xsfr8p0o4p5=log10(data3['flux_ap2_80']/data3['flux_ap2_45'])

#stern wedge colors, SWIRE
xsmd3p6m4p5=-2.5*log10(data3['flux_ap2_36']/data3['flux_ap2_45'])+0.485
xsmd5p8m8p0=-2.5*log10(data3['flux_ap2_58']/data3['flux_ap2_80'])+0.634


det24 = data1['flux_ap2_24'] > 10.0

#qr5p8o3p6=log10(data2['[5.8]']/data2['[3.6]'])
#qr8p0o4p5=log10(data2['[8.0]']/data2['[4.5]'])

c1mod = np.arange(-0.3,1.5,0.1)
c2mod = np.arange(-0.3,0.3,0.1)
c1modo = np.arange(-0.1,1.5,0.1)
c2modo = np.arange(-0.2,0.4,0.1)
line  = np.zeros(18)-0.3
lineo = np.zeros(16)-0.2
linev = np.zeros(6)-0.3
linevo = np.zeros(7)-0.1

diagmd = 0.8*c1mod+0.5

ax1 = fig.add_subplot(121,autoscale_on=False,xlim=(-0.6,0.8),ylim = (-0.8,1.3))
for label in ax1.get_xticklabels():
    label.set_fontproperties(ticks_font)
for label in ax1.get_yticklabels():
    label.set_fontproperties(ticks_font)


ax1.plot(fr5p8o3p6,fr8p0o4p5,'k.',alpha=0.1,ms=1)
ax1.plot(fr5p8o3p6[det24],fr8p0o4p5[det24],marker='.',color='#ffcc66',linestyle='None',alpha=1.0,ms=1.0,label='24 $\mu$m detections')

ax1.plot(xsfr5p8o3p6,xsfr8p0o4p5,'c.',alpha=1.0,ms=1,label='X-ray detections')
    
ax1.set_xlabel('Log$_{10}$($S_{5.8}/S_{3.6}$)',family='times new roman')
ax1.set_ylabel('Log$_{10}$($S_{8.0}/S_{4.5}$)',family='times new roman')

# Jenn's criteria:

c1dond1 = np.arange(0.35,1.5,0.01)
c1donh = np.arange(0.08,0.35,0.01)
c1dond2 = np.arange(0.08,1.5,0.01)
c2donv= np.arange(0.15,0.35,0.01)
l1don = c1donh*0.0+0.15
l2don = c2donv*0.0+0.08
diagdon1 = 1.21*c1dond1-0.27
diagdon2 = 1.21*c1dond2+0.27

#My wedge:
#c1mod = np.arange(-0.3,1.5,0.1)
#c2mod = np.arange(-0.3,0.3,0.1)
c1modo = np.arange(-0.1,1.5,0.1)
c2modo = np.arange(-0.2,0.4,0.1)
#line  = np.zeros(18)-0.3
lineo = np.zeros(16)-0.2
#linev = np.zeros(6)-0.3
linevo = np.zeros(7)-0.1

diagmd = 0.8*c1modo+0.5


ax1.plot(c1donh,l1don,'r-.')
ax1.plot(c1dond1,diagdon1,'r-.')
ax1.plot(c1dond2,diagdon2,'r-.')
ax1.plot(l2don,c2donv,'r-.',label='Donley+12')

ax1.plot(c1modo,lineo,'b:')
ax1.plot(c1modo,lineo,'b:')
ax1.plot(linevo,c2modo,'b:')
ax1.plot(c1modo,diagmd,'b:',label='Lacy+04')

ax1.legend(loc=4)


# figure out which objects are inside the Donley wedge, but lack X-ray IDs
donley= (fr5p8o3p6 > 0.08) & (fr8p0o4p5 > 0.15) & (fr8p0o4p5 > 1.21*fr5p8o3p6-0.27) & (fr8p0o4p5 < 1.21*fr5p8o3p6+0.27) & (data1['flux_ap2_36'] < data1['flux_ap2_45']) & (data1['flux_ap2_45'] < data1['flux_ap2_58']) & (data1['flux_ap2_58'] < data1['flux_ap2_80'])
#ax1.plot(fr5p8o3p6[donley],fr8p0o4p5[donley],'b.',alpha=1,ms=3)
# same for Lacy
lacy=(fr5p8o3p6 > -0.1) & (fr8p0o4p5 > -0.2) & (fr8p0o4p5 < 0.8*fr5p8o3p6+0.5)
#ax1.plot(fr5p8o3p6[lacy],fr8p0o4p5[lacy],'b.',alpha=1,ms=3)

ax2 = fig.add_subplot(122,autoscale_on=False,xlim=(-1.0,3.5),ylim = (-0.2,1.5))
for label in ax1.get_xticklabels():
    label.set_fontproperties(ticks_font)
for label in ax1.get_yticklabels():
    label.set_fontproperties(ticks_font)

ax2.set_xlabel('[5.8]-[8.0] (Vega)',family='times new roman')
ax2.set_ylabel('[3.6]-[4.5] (Vega)',family='times new roman')
    
# stern wedge
swax1=np.arange(0.6,3.5,0.01)
swax2=np.arange(0.3,1.5,0.01)
swdiag1=0.2*swax1+0.18
swdiag2=2.5*swax1-3.5
swl1=np.zeros(np.size(swax2))+0.6

g=swax1 < 1.61
ng=swax1 > 1.60

stern= (md5p8m8p0 > 0.6) & (md3p6m4p5 > 0.2*md5p8m8p0+0.18) & (md3p6m4p5 > 2.5*md5p8m8p0-3.5)

ax2.plot(swax1[g],swdiag1[g],'m-.')
ax2.plot(swax1[ng],swdiag2[ng],'m-.')
ax2.plot(swl1,swax2,'m-.',label='Stern+05')

    
ax2.plot(md5p8m8p0,md3p6m4p5,'k.',alpha=0.1,ms=0.3)
#ax2.plot(md5p8m8p0[stern],md3p6m4p5[stern],'b.',alpha=0.1,ms=3)
ax2.plot(md5p8m8p0[donley],md3p6m4p5[donley],'r.',alpha=1,ms=1,label='Donley+12 AGN cands.')
ax2.plot(xsmd5p8m8p0,xsmd3p6m4p5,'c.',alpha=1.0,ms=1.0,label='X-ray sources')
ax2.legend(loc=4)

plt.savefig('wedges.png',dpi=300)
