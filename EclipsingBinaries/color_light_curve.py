# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:45:40 2020
Created on Tue Feb 16 19:29:16 2021
@author: Alec Neal

Last Edited: 03/16/2022
Editor: Kyle Koeller
"""

# import vseq  # testing purposes
from . import vseq_updated as vseq
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from tkinter import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)
from matplotlib.figure import Figure
import textwrap


def occ2(B_HJD, V_HJD, period, tolerance=0.01):
    tol = tolerance * period
    master_goods = []
    good_before = []
    index_before = []
    good_after = []
    index_after = []
    good_diff = []
    bad_obs = 0
    for n in range(len(V_HJD)):
        master_goods.append([])
        good_before.append([])
        good_after.append([])
        index_before.append([])
        index_after.append([])
        for i in range(len(B_HJD)):
            if abs(V_HJD[n] - B_HJD[i]) < tol:
                master_goods[n].append(B_HJD[i])
                if V_HJD[n] - B_HJD[i] > 0:
                    good_before[n].append(B_HJD[i])
                    index_before[n].append(i)
                    good_diff.append(abs((V_HJD[n] - B_HJD[i]) / period))
                elif V_HJD[n] - B_HJD[i] < 0:
                    good_after[n].append(B_HJD[i])
                    index_after[n].append(i)
                    good_diff.append(abs((V_HJD[n] - B_HJD[i]) / period))
        if len(good_before[n]) == 0 or len(good_after[n]) == 0:
            bad_obs += 1
    # print(bad_obs,'('+str(round(bad_obs/len(V_HJD)*100,2))+' %)',tolerance)
    return bad_obs, good_before, good_after, index_before, index_after, good_diff


def best_tol(B_HJD, V_HJD, period, lower_lim=0.05, max_tol=0.03):
    tol0 = 0.003
    tol = tol0
    obs = len(V_HJD)
    dtol = 0.001
    while occ2(B_HJD, V_HJD, period, tolerance=tol)[0] / obs > lower_lim:
        tol = round(tol + dtol, 8)
        if tol > max_tol:
            break
    return tol


lin_interp = lambda x, x1, x2, y1, y2: y1 + (x - x1) * ((y2 - y1) / (x2 - x1))
mean_mag = lambda maglist: -2.5 * np.log10(np.mean(10 ** (-0.4 * np.array(maglist))))


def subtract_LC(Bfile, Vfile, Epoch, period,
                max_tol=0.03, lower_lim=0.05, FTinterp=True, quad_range=0.075):
    """
    This function actually creates the B-V and R-V values

    :param Bfile: input B file
    :param Vfile: input V file
    :param Epoch: epoch number
    :param period: period of the system
    :param max_tol: maximum tolerance
    :param lower_lim: lower limit
    :param FTinterp: interpolates number
    :param quad_range: ?
    :return: returns the B-V value and other assorted values
    """
    B_HJD, B_mag, B_magerr = vseq.io.importFile_pd(Bfile)[:3:]
    V_HJD, V_mag, V_magerr = vseq.io.importFile_pd(Vfile)[:3:]

    Bpoly = vseq.binning.polybinner(Bfile, Epoch, period, sections=2, section_order=8)
    Bphase = Bpoly[1][0][0][1]
    aB = Bpoly[0][0]
    bB = Bpoly[0][1]
    Bnorm = Bpoly[1][0][4]
    Vphase = list(vseq.calc.astro.convert.HJD_phase(V_HJD, period, Epoch))
    B_flux = np.array(10 ** (-0.4 * np.array(B_mag))) / Bnorm
    obs = len(V_HJD)
    tolerance = best_tol(B_HJD, V_HJD, period, lower_lim=lower_lim, max_tol=max_tol)
    before_after = occ2(B_HJD, V_HJD, period, tolerance=tolerance)
    befores = before_after[1]
    afters = before_after[2]
    i_before = before_after[3]
    i_after = before_after[4]
    mean_diff = np.mean(before_after[5])

    B_interp_flux = []
    for n in range(obs):
        if len(befores[n]) == 0 or len(afters[n]) == 0:
            B_interp_flux.append(vseq.FT.sumatphase(Vphase[n], 10, aB, bB))
        else:
            B_interp_flux.append(lin_interp(V_HJD[n], befores[n][-1], afters[n][0],
                                            B_flux[i_before[n][-1]], B_flux[i_after[n][0]]))

    B_interp_mag = -2.5 * np.log10(np.array(B_interp_flux) * Bnorm)
    # quad_range=0.075
    BVquadphase = []
    BVquadmag = []
    # Vquad=[]
    for n in range(len(Vphase)):
        if 0.25 - quad_range < Vphase[n] < 0.25 + quad_range or 0.75 - quad_range < Vphase[n] < 0.75 + quad_range:
            BVquadphase.append(Vphase[n])
            BVquadmag.append(B_interp_mag[n] - V_mag[n])
    quadcolor = mean_mag(BVquadmag)
    colorerr = st.stdev(BVquadmag, xbar=quadcolor)
    print(quadcolor, '+/-', colorerr)

    B_minus_V = B_interp_mag - np.array(V_mag)
    BV_mean = mean_mag(B_minus_V)
    # print(B_minus_V)
    BV_err = st.stdev(B_minus_V, xbar=BV_mean)

    print('ave diff =', round(mean_diff * 100, 3), '% of period')
    aVphase, aV_mag, aB_interp_mag = vseq.plot.aliasing2(Vphase, V_mag, B_interp_mag)[:3:]
    aBphase, aB_mag = vseq.plot.aliasing2(Bphase, B_mag, B_mag)[:2:]
    aB_minus_V = vseq.plot.aliasing2(Vphase, B_minus_V, B_minus_V)[1]
    B_V = [B_minus_V, BV_mean, BV_err, aB_minus_V]
    B = [aBphase, aB_mag, aB_interp_mag]
    V = [aVphase, aV_mag]

    print('T =', vseq.Flower.T.Teff(quadcolor - (0.641 / 3.1)))

    return B_V, B, V, quadcolor, colorerr


# use this function below
def color_plot(Bfile, Vfile, Epoch, period, max_tol=0.03, lower_lim=0.05, Rfile='', FTinterp=True,
               save=False, outName='noname_color.png', fs=12):
    """
    This is a function version of the GUI and produces the same values but without the plotting aspect

    :param Bfile: input B text file
    :param Vfile: input V text file
    :param Epoch: epoch number
    :param period: period of the system
    :param max_tol: maximum tolerance
    :param lower_lim: lower limit
    :param Rfile: input R text file
    :param FTinterp: interpolate number
    :param save: save the output image
    :param outName: output image name
    :param fs:
    :return: assorted values
    """
    B_V = subtract_LC(Bfile, Vfile, Epoch, period, max_tol=max_tol, lower_lim=lower_lim, FTinterp=FTinterp)
    Bphase, Bmag, B_interp_mag = B_V[1][:3:]
    Vphase, Vmag = B_V[2][:2:]
    aB_minus_V = B_V[0][3]
    quadcolor, colorerr = B_V[3:5:]
    if Rfile == '':
        axs, fig = vseq.plot.multiplot((7, 7.5), height_ratios=[8, 4.5])
        mag = axs[0]
        bv = axs[1]
        mag.plot(Vphase, Vmag, 'og', ms=2)
        mag.plot(Bphase, Bmag, 'ob', ms=2)
        bv.plot(Vphase, aB_minus_V, 'ok', ms=2)
        bv.margins(y=0.1, x=1 / 24)
        mag.set_ylim(mag.get_ylim()[::-1])
        bv.set_ylim(bv.get_ylim()[::-1])
        vseq.plot.sm_format(mag, X=0.25, x=0.05, Y=None, numbersize=fs, xbottom=False, bottomspine=False, tickwidth=1,
                            Xsize=7, xsize=3.5)
        vseq.plot.sm_format(bv, X=0.25, x=0.05, numbersize=fs, xtop=False, topspine=False, tickwidth=1, Xsize=7,
                            xsize=3.5)

        maxtick = max(list(map(len, (list(map(str, np.array(mag.get_yticks()).round(8)))))))
        if maxtick == 5:
            ytickpad = -0.835
        else:
            ytickpad = -0.81
        mag.text(ytickpad, (max(Bmag) + min(Bmag)) / 2, 'B', rotation=90, fontsize=fs * 1.2)
        mag.text(ytickpad, (max(Vmag) + min(Vmag)) / 2, 'V', rotation=90, fontsize=fs * 1.2)
        # bv.set_xlabel('$\Phi$',fontsize=fs*1.2)
        bv.set_xlabel('$\Phi$', fontsize=fs * 1.5, usetex=False)
        bv.set_ylabel(r'$\rm B-V$', fontsize=fs * 1.2)
        # quadcolor,colorerr=B_V[3:5:]
        bv.axhline(quadcolor, color='gray', linewidth=None)
    else:
        V_R = subtract_LC(Vfile, Rfile, Epoch, period, max_tol, lower_lim=lower_lim)
        Rphase, Rmag = V_R[2][:2:]
        V_interp_mag = V_R[1][2]
        aV_minus_R = V_R[0][3]
        axs = vseq.plot.multiplot((7, 9), height_ratios=[8, 3, 3])
        mag = axs[0]
        bv = axs[2]
        vr = axs[1]
        mag.plot(Vphase, Vmag, 'og', ms=2)
        mag.plot(Bphase, Bmag, 'ob', ms=2)
        mag.plot(Rphase, Rmag, 'or', ms=2)

        bv.plot(Vphase, aB_minus_V, 'ok', ms=3)
        vr.plot(Rphase, aV_minus_R, 'ok', ms=3)
        bv.margins(y=0.07, x=1 / 24)
        vr.margins(y=0.07)
        # mag.margins(y=0.09)
        mag.set_ylim(mag.get_ylim()[::-1])
        bv.set_ylim(bv.get_ylim()[::-1])
        vr.set_ylim(vr.get_ylim()[::-1])
        vseq.plot.sm_format(mag, X=0.25, x=0.05, numbersize=fs, xbottom=False, bottomspine=False)
        vseq.plot.sm_format(vr, X=0.25, x=0.05, numbersize=fs, xtop=False, topspine=False, xbottom=False,
                            bottomspine=False)
        vseq.plot.sm_format(bv, X=0.25, x=0.05, numbersize=fs, xtop=False, topspine=False)
        maxtick = max(list(map(len, (list(map(str, np.array(mag.get_yticks()).round(8)))))))
        if maxtick == 5:
            ytickpad = -0.835
        else:
            ytickpad = -0.81
        mag.text(ytickpad, (max(Bmag) + min(Bmag)) / 2, r'$\rm B$', rotation=90, fontsize=fs * 1.2)
        mag.text(ytickpad, (max(Vmag) + min(Vmag)) / 2, r'$\rm V$', rotation=90, fontsize=fs * 1.2)
        mag.text(ytickpad, (max(Rmag) + min(Rmag)) / 2, r'$\rm R_C$', rotation=90, fontsize=fs * 1.2)
        bv.set_ylabel(r'$\rm B-V$', fontsize=fs * 1.2)
        vr.set_ylabel(r'$\rm V-R_C$', fontsize=fs * 1.2)
        bv.set_xlabel(r'$\Phi$', fontsize=fs * 1.2)
    if save:
        plt.savefig(outName, bbox_inches='tight')
    plt.show()
    return quadcolor, colorerr


# ==
class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """Display text in tooltip window"""
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


class gui:
    def Field(root, text, row, column, def_val=None, ftype='str', colspan=1, width=30):
        Flab = Label(root, text=text)
        Flab.grid(row=row, column=column, columnspan=colspan)
        Fent = Entry(root, width=width)
        Fent.grid(row=row, column=column + 1, columnspan=colspan)
        if def_val is not None:
            Fent.insert(0, def_val)
        if ftype == 'float':
            got = float(Fent.get())
        elif ftype == 'int':
            got = int(Fent.get())
        elif ftype == 'bool':
            got = bool(Fent.get() == 'True')
        else:
            got = Fent.get()
        # self.value=got
        return got


# ==
autowrap = lambda text, width=70: '\n'.join(textwrap.wrap(text, width=width))


def color_gui(developer=False):
    """
    Creates the GUI that the user sees and interacts with
    """
    # default_font = font.nametofont("TkDefaultFont")
    # default_font.configure(size=12)
    root = Tk()
    root.option_add('*Font', '12')
    root.title('CLC-gui v0.2.1')
    # root.geometry('1200x800')
    # disp=3
    # T=Text(root,height=disp,width=25)
    # T.grid(row=0,column=1)
    Intro = Label(root, text='Color Light Curve - gui\nversion 0.2.1 (2/19/21)\nby Alec Neal\n')
    Intro.grid(row=0, column=0, columnspan=2)
    # Intro.config(font=('Arial',12))
    # T.insert(END,'Hello world!')

    Label(root, text=autowrap('Program to determine light curve colors. Mouse over the fields for more information.',
                              width=50) + '\n').grid(row=1, column=0, columnspan=2)
    entries = [['B file'],
               ['V file'],
               ['R (optional)'],
               ['Epoch'],
               ['Period'],
               ['Max. tolerance'],
               ['Lower limit'],
               ['Save? (True/False)'],
               ['Output file']]

    for parameter in range(len(entries)):
        # if parameter < 3:
        # wid=30
        # else:
        # wid=15
        if entries[parameter][0] == 'Save? (True/False)':
            var = IntVar()
            c = Checkbutton(root, text='', variable=var)
            c.grid(row=parameter + 2, column=1, sticky='w')
            entries[parameter].append(var)
            Label(root, text='Save').grid(row=parameter + 2, column=0)
        else:
            entries[parameter].append(Entry(root, width=30))
            Label(root, text=entries[parameter][0]).grid(row=parameter + 2, column=0)
            entries[parameter][1].grid(row=parameter + 2, column=1)

    # B=['B file',Entry(root,width),Label(root,text=B[0])]

    B, V, R, Epoch, Period, MaxT, LL, Save, Out = entries
    defaults = lambda entr, value: entr.append(entr[1].insert(0, value))
    defaults(MaxT, 0.03)
    defaults(LL, 0.05)
    defaults(R, '')
    # defaults(Save,'False')
    defaults(Out, 'color_light_plot.pdf')
    if developer:
        defaults(B, 'NSVS2854398_B2_mags.txt')
        defaults(V, 'NSVS2854398_V2_mags.txt')
        defaults(R, 'NSVS2854398_R2_mags.txt')
        defaults(Epoch, 2458308.729976)
        defaults(Period, 0.290374)
    # === tool tips! =====
    """
    Creates tool tips that users look at when hovering certain aspects of the GUI
    """
    CreateToolTip(MaxT[1],
                  text=autowrap('The largest fraction of the period to find adjacent points for linear interpolation.'
                                ' We don\'t want to use the next observation if that observation is days or weeks away!'
                                ' If the next interp. point is beyond this, the program interpolates using a Fourier transform of the light curve.',
                                width=70))
    CreateToolTip(Intro, text=autowrap('Program to determine the color index of a light curve, given up to 3 filters.'
                                       ' Because we can\'t take filter images at the same time, interpolation '
                                       'is required in order to get the instantaneous color index at each observation. '
                                       'This program calculates B-V, and V-R if R is given. The bluer color is the color that '
                                       'is interpolated from the redder filter\'s times of observation (B in B-V and V in V-R).'
                                       ' In the case of B, it takes B observations lying between two V times and linear interpolates to the time.'
                                       ' However, when the bordering magnitudes are too far apart, it interpolates using a Fourier transform'
                                       ' of the B light curve.', width=70))

    CreateToolTip(LL[1],
                  text=autowrap('The tolerance will increase (unless it reaches max tolerance (MaxT) first) until only '
                                'this percentage (default 5%) of the light curve is FT interpolated. So for densely '
                                'sampled data, expect 5% of the B mags to be FT interpolated, but > 5% for sparsely sampled'
                                ', since it won\'t go over MaxT. This is somewhat unnecessary (and potentially harmful), but '
                                'works if the given MaxT is not desirable and is only last ditch. You can set it to 0, which will'
                                ' essentially set the tolerance to the MaxT.'
                                , width=70))
    # CreateToolTip(Save[1],text=autowrap('Type True here if you want your image saved to the specified output name (below).',width=70))
    CreateToolTip(Out[1], text=autowrap(
        'Enter your desired output name. The extension indicates the type of file it will be saved as.'
        ' Go to the matplotlib website to see the full list, but recommended formats'
        ' are vector graphic types like .pdf, .svg (.eps can be iffy). For raster'
        ' images use .png (I don\'t think jpegs (.jpg) are supported).', width=70))
    CreateToolTip(Epoch[1], text=autowrap('Phase zero for the light curve. This is arbitrary with respect to'
                                          ' the individual B-V colors, but important to the resulting color index'
                                          ' since it is calculated using the B-V near quadrature (both stars showing), '
                                          'so the Epoch is ideally a time of minimum light. Future versions will likely allow more flexibility.',
                                          70))
    CreateToolTip(Period[1], text=autowrap('Orbital period of the star.'))
    CreateToolTip(B[1], text=autowrap('Enter the file names for the individual filters. The B and V fields'
                                      ' are required, but R is optional. The files should have no'
                                      ' column headers, and be in the format of HJD, mag, magerr.'
                                      ' Currently the files need to be in the same folder as this program.'
                                      ' The fields are placeholders and don\'t need be the corresponding filters---'
                                      'feel free to put the same file in three times for some silliness!'))
    CreateToolTip(R[1], text=autowrap(
        'Only two bandpasses are required to calculate color, so this field is optional. Moreover, calibrated R magnitudes are not as common.'
        ' If this field is left blank, it will make only a B-V plot. If R is entered,'
        ' the three light curves will be shown along with interpolated B-V, V-R colors.'))
    # ====================
    getit = lambda entr: entr[1].get()
    temp = Label(root, text='')
    BVL = Label(root, text='')
    BVL.grid(row=len(entries) + 6, column=0, columnspan=2)
    VRL = Label(root, text='')
    VRL.grid(row=len(entries) + 5, column=0, columnspan=2)
    # B2=gui.Field(root,'B file',2,0)
    fs = 14

    def call_colorplot2():
        """
        Calls to create the color plot after clicking the plot button.

        This creates plots for both the B-V and the V-R
        """
        B_V = subtract_LC(Bfile=getit(B), Vfile=getit(V), Epoch=float(getit(Epoch)), period=float(getit(Period)),
                          max_tol=float(getit(MaxT)), lower_lim=float(getit(LL)))
        # B_V=subtract_LC(Bfile=B,Vfile=V,Epoch=Epoch,period=Period,
        # max_tol=MaxT,lower_lim=LL)
        Bphase, Bmag, B_interp_mag = B_V[1][:3:]
        Vphase, Vmag = B_V[2][:2:]
        aB_minus_V = B_V[0][3]
        quadcolor, colorerr = B_V[3:5:]
        if getit(R) == '':
            """
            Checks whether the user has entered a R band text file
            """
            fig = Figure(figsize=(7, 7.8), dpi=90, tight_layout=True)
            # canvas = FigureCanvasTkAgg(fig, master=root)
            # canvas.destroy()
            axs = vseq.plot.multiplot(height_ratios=[8, 4.5], fig=fig)
            mag = axs[0]
            bv = axs[1]
            mag.plot(Vphase, Vmag, 'o', ms=2, color='#00FF00')
            mag.plot(Bphase, Bmag, 'ob', ms=2)
            bv.plot(Vphase, aB_minus_V, 'ok', ms=3)
            bv.margins(y=0.1, x=1 / 24)
            mag.set_ylim(mag.get_ylim()[::-1])
            bv.set_ylim(bv.get_ylim()[::-1])
            vseq.plot.sm_format(mag, X=0.25, x=0.05, Y=None, numbersize=fs, xbottom=False, bottomspine=False,
                                tickwidth=1, Xsize=7, xsize=3.5)
            vseq.plot.sm_format(bv, X=0.25, x=0.05, numbersize=fs, xtop=False, topspine=False, tickwidth=1, Xsize=7,
                                xsize=3.5)

            maxtick = max(list(map(len, (list(map(str, np.array(mag.get_yticks()).round(8)))))))
            if maxtick == 5:
                ytickpad = -0.835
            else:
                ytickpad = -0.81
            mag.text(ytickpad, (max(Bmag) + min(Bmag)) / 2, 'B', rotation=90, fontsize=fs * 1.2)
            mag.text(ytickpad, (max(Vmag) + min(Vmag)) / 2, 'V', rotation=90, fontsize=fs * 1.2)
            # bv.set_xlabel('$\Phi$',fontsize=fs*1.2)
            bv.set_xlabel('$\Phi$', fontsize=fs * 1.2, usetex=False)
            bv.set_ylabel(r'$\rm B-V$', fontsize=fs * 1.2)
            # quadcolor,colorerr=B_V[3:5:]
            # bv.axhline(quadcolor,color='gray',linewidth=None)
            VRL.config(text='', bg=None, relief=None, padx=0, pady=0, borderwidth=0)
            # canvas.draw()
            # canvas.get_tk_widget().grid(row=0,column=3,rowspan=100,padx=5)
            # if getit(Save) == 'True':
            # plt.savefig(getit(Out),bbox_inches='tight')
        else:
            V_R = subtract_LC(Bfile=getit(V), Vfile=getit(R), Epoch=float(getit(Epoch)), period=float(getit(Period)),
                              max_tol=float(getit(MaxT)), lower_lim=float(getit(LL)))
            # V_R=subtract_LC(Bfile=V,Vfile=R,Epoch=Epoch,period=Period,
            # max_tol=MaxT,lower_lim=LL)
            VRc, VRerr = V_R[3:5:]
            Rphase, Rmag = V_R[2][:2:]
            V_interp_mag = V_R[1][2]
            aV_minus_R = V_R[0][3]
            fig = Figure(figsize=(7, 9), dpi=90, tight_layout=True)
            # canvas = FigureCanvasTkAgg(fig, master=root)
            # canvas.destroy()
            axs = vseq.plot.multiplot(height_ratios=[8, 3, 3], fig=fig)
            mag = axs[0]
            bv = axs[2]
            vr = axs[1]
            mag.plot(Vphase, Vmag, 'o', ms=2, color='#00FF00')
            mag.plot(Bphase, Bmag, 'ob', ms=2)
            mag.plot(Rphase, Rmag, 'or', ms=2)

            bv.plot(Vphase, aB_minus_V, 'ok', ms=3)
            vr.plot(Rphase, aV_minus_R, 'ok', ms=3)
            bv.margins(y=0.07, x=1 / 24)
            vr.margins(y=0.07)
            # mag.margins(y=0.09)
            mag.set_ylim(mag.get_ylim()[::-1])
            bv.set_ylim(bv.get_ylim()[::-1])
            vr.set_ylim(vr.get_ylim()[::-1])
            vseq.plot.sm_format(mag, X=0.25, x=0.05, numbersize=fs, xbottom=False, bottomspine=False)
            vseq.plot.sm_format(vr, X=0.25, x=0.05, numbersize=fs, xtop=False, topspine=False, xbottom=False,
                                bottomspine=False)
            vseq.plot.sm_format(bv, X=0.25, x=0.05, numbersize=fs, xtop=False, topspine=False)
            maxtick = max(list(map(len, (list(map(str, np.array(mag.get_yticks()).round(8)))))))
            if maxtick == 5:
                ytickpad = -0.835
            else:
                ytickpad = -0.825
                # ytickpad=-0.81
            mag.text(ytickpad, (max(Bmag) + min(Bmag)) / 2, r'$\rm B$', rotation=90, fontsize=fs * 1.2)
            mag.text(ytickpad, (max(Vmag) + min(Vmag)) / 2, r'$\rm V$', rotation=90, fontsize=fs * 1.2)
            mag.text(ytickpad, (max(Rmag) + min(Rmag)) / 2, r'$\rm R_C$', rotation=90, fontsize=fs * 1.2)
            bv.set_ylabel(r'$\rm B-V$', fontsize=fs * 1.2)
            vr.set_ylabel(r'$\rm V-R_C$', fontsize=fs * 1.2)
            bv.set_xlabel(r'$\Phi$', fontsize=fs * 1.2)

            VRL.config(text='(V-R) = ' + str(round(VRc, 6)) + ' +/- ' + str(round(VRerr, 6)), bg='white',
                       relief='solid', borderwidth=1, padx=5, pady=5, font=('None', 14))
            show_color = False
            if show_color:
                # vr.annotate(r'$V-R_{\rm C}='+str(round(VRc,4))+'\pm'+str(round(VRerr,4))+'$',xy=(0.25,vr.get_ylim()[-1]),ha='center')
                # vr.plot([''])
                # vr.annotate(r'$V-R_{\rm C}='+str(round(VRc,4))+'\pm'+str(round(VRerr,4))+'$',xy=(0.25,VRc),ha='center',va='center',bbox=dict(facecolor='white', edgecolor='gray',boxstyle='round',pad=0.1),fontsize=11)

                vr.annotate(r'$V-R_{\rm C}=' + str(round(VRc, 4)) + '\pm' + str(round(VRerr, 4)) + '$',
                            xytext=(0.25, vr.get_ylim()[-1]), xy=(0, VRc), ha='center', va='center',
                            bbox=dict(facecolor='white', edgecolor='gray', pad=0.1), fontsize=11,
                            arrowprops=dict(arrowstyle='-', color='gray', linewidth=1.5))
                bv.annotate(r'$B-V=' + str(round(quadcolor, 4)) + '\pm' + str(round(colorerr, 4)) + '$',
                            xytext=(0.25, bv.get_ylim()[-1]), xy=(0, quadcolor), ha='center', va='center',
                            bbox=dict(facecolor='white', edgecolor='gray', linewidth=1.5), fontsize=11,
                            arrowprops=dict(arrowstyle='-', color='gray', linewidth=1.5))

                # bv.annotate(r'$B-V='+str(round(quadcolor,4))+'\pm'+str(round(colorerr,4))+'$',
                # xy=(0.25,bv.get_ylim()[-1]),ha='center')
                # vr.axhline(VRc,ls='-',color='k',lw=3)
                # vr.axhline(VRc,ls='-',color='lime',lw=2)
                # vr.axhline(VRc,ls='-',color='red',lw=1)
                vr.axhline(VRc, ls='-', color='gray', lw=1.5)
                # bv.axhline(quadcolor,ls='-',color='blue',lw=2)
                # bv.axhline(quadcolor,ls='-',color='lime',lw=1)
                bv.axhline(quadcolor, ls='-', color='gray', lw=1.5)
                # vr.annotate('>',xy=(vr.get_xlim()[0],VRc),va='center',ha='center',color='magenta')
                # vr.annotate('<',xy=(vr.get_xlim()[-1],VRc),va='center',ha='center',color='magenta')
            # canvas.draw()
            # canvas.get_tk_widget().grid(row=0,column=3,rowspan=100,padx=5)
            # if getit(Save) == 'True':
            # plt.savefig(getit(Out),bbox_inches='tight')

        temp.config(text='(B-V) = ' + str(round(quadcolor, 6)) + ' +/- ' + str(round(colorerr, 6)), bg='white',
                   relief='solid', borderwidth=1, padx=5, pady=5, font=('None', 14))

        BVL.config(text='(B-V) = ' + str(round(quadcolor, 6)) + ' +/- ' + str(round(colorerr, 6)), bg='white',
                   relief='solid', borderwidth=1, padx=5, pady=5, font=('None', 14))
        CreateToolTip(BVL, text=autowrap(
            'These values are calculated using an average of the (X-Y) values within phase 0.075 of quadrature (phase = +/- 0.25).'
            ' The given error is the standard\ndeviation of these values.', 50))

        # =============================
        canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        # canvas.delete('all')
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=3, rowspan=100, padx=5)
        # if getit(Save) == 'True':
        # fig.savefig(getit(Out),bbox_inches='tight')
        if Save[1].get() == 1:
            # if getit(Out)[:-4] == '.png':
            # fig.savefig(getit(Out),bbox_inches='tight')
            # else:
            fig.savefig(getit(Out), bbox_inches='tight', dpi=256)
        # plt.show()
        return 'finishcallback'

    Label(root, text='').grid(row=len(entries) + 2, column=0)
    plot_button = Button(root, text='Plot!', padx=30, pady=10, command=call_colorplot2, bg='gray', fg='white')
    # plot_button.grid(row=0,column=0)
    plot_button.grid(row=len(entries) + 3, column=0, columnspan=2)

    root.mainloop()


# =======================

# If you want to use the gui
# color_gui(False)

# or just the function itself
# color_plot('Bfile','Vfile',Epoch,period)
