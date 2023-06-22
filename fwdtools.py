import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
import sunpy.map
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation,SkyCoord
from sunpy.time import parse_time
from sunpy.coordinates import frames, sun
from matplotlib.patches import Ellipse

def sfu2tb(sfumap,lofarmap):
    #Convert solar flux units to brightness temperature
    #This is mostly for the GS model results
    b_maj,b_min,b_ang=plot_beam(lofarmap)
    #beam semimajor/semiminor axes in radians
    sigmax=np.abs(b_maj*3600/(2*np.sqrt(2*np.log(2))))/3600.*np.pi/180.
    sigmay=np.abs(b_min*3600/(2*np.sqrt(2*np.log(2))))/3600.*np.pi/180.
    #Boltzmann constant
    k=1.380649e-23 # m2 kg s-2 K-1
    #Conversion factor between sfu and Jy
    sfu2si=1.e-22 # 1 sfu = 10−22 W m−2 Hz−1
    wavelength=3.e8/(lofarmap.meta['wavelnth']*1.e6)
    omega=(np.pi*sigmax*sigmay)/(4*np.log(2))
    tb=(sfumap.data*1.e-22*wavelength*wavelength)/(2.*k*omega)
    return sunpy.map.Map(tb,sfumap.meta)
    
def plot_beam(lofarmap):
    # Get size and shape of LOFAR beam
    solar_PA = sun.P(sunpy.time.parse_time(lofarmap.meta['date-obs'])).degree
    b_maj = lofarmap.meta['BMAJ']
    b_min  = lofarmap.meta['BMIN']
    b_ang = (lofarmap.meta['BPA']+solar_PA) # should consider the beam for the data
    return [b_maj, b_min, b_ang] # Output in degrees

def get_plotbeam(lofar_imagemap,beampos=(50, 50)):
    #Calculate and prepare for plotting the fitted Gaussian beam
    #time=parse_time(lofar_imagemap.meta['DATE-OBS']).datetime
    #solar_PA = sun.P(time).degree
    b_maj, b_min, b_ang = plot_beam(lofar_imagemap)
    pixarc = lofar_imagemap.meta['cdelt1']
    plotbeam = Ellipse(beampos, b_maj*3600.0/pixarc, b_min*3600.0/pixarc, b_ang, color='white', linewidth=1)
    return plotbeam

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def get_beam(meta,scaled=True):
    time=parse_time(meta['DATE-OBS']).datetime
    solar_PA = sun.P(time).degree
    b_maj =  meta['BMAJ']
    if scaled is True:
        b_maj =  b_maj/meta['CDELT1']
    b_min  = meta['BMIN']
    if scaled is True:
        b_min=b_min/meta['CDELT2']
    b_ang = meta['BPA']+90.-solar_PA # should consider the beam for the data
    return [b_maj,b_min,b_ang]
    
def makeGaussian2d(x_center=0, y_center=0, theta=0, sigma_x = 10, sigma_y=10, x_size=640, y_size=640):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame 

    theta = 2*np.pi*theta/360.
    x = np.arange(0,x_size, 1, float)
    y = np.arange(0,y_size, 1, float)
    y = y[:,np.newaxis]
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center

    # rotation
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*x0 -np.sin(theta)*y0
    b0=np.sin(theta)*x0 +np.cos(theta)*y0
    gaussian=np.exp(-(((a-a0)**2)/(2*(sx**2)) + ((b-b0)**2) /(2*(sy**2))))
    
    return gaussian/np.sum(gaussian)


def load_imaging_file(filename):
    hdul = fits.open(filename)
    header=hdul[0].header
    image=hdul[0].data[0,0,:,:]
    return image,header
    
    
def load_lofar_image(filename,scalefactor=None):
    image,header=load_imaging_file(filename)
    if scalefactor is not None:
        newshape=(int(image.shape[0]/scalefactor),int(image.shape[1]/scalefactor))
        miniimage=rebin(image,newshape)
        image=miniimage
    return image,header
    
def get_lofar_map(lofar_imagefile,bottom_left=None,top_right=None):
    """
    ========================================================================================
    Function returns a Helioprojective Map from observations in the RA-DEC coordinate system
    Written by Laura Hayes. The output is brightness temperature in Kelvin
    ========================================================================================
    How to create a `~sunpy.map.Map` in Helioprojective Coordinate Frame from radio observations
    in GCRS (RA-DEC).
    In this example a LOFAR FITS file (created with LOFAR's `Default Pre-Processing Pipeline (DPPP) and
    WSClean Imager <https://support.astron.nl/LOFARImagingCookbook/dppp.html>`__) is read in,
    the WCS header information is then used to make a new header with the information in Helioprojective,
    and a `~sunpy.map.Map` is made.
    The LOFAR example file has a WCS in celestial coordinates i.e. Right Ascension and
    Declination (RA-DEC). For this example, we are assuming that the definition of LOFAR's
    coordinate system for this observation is exactly the same as Astropy's ~astropy.coordinates.GCRS.
    For many solar studies we may want to plot this data in some Sun-centered coordinate frame,
    such as `~sunpy.coordinates.frames.Helioprojective`. In this example we read the data and
    header information from the LOFAR FITS file and then create a new header with updated WCS
    information to create a `~sunpy.map.Map` with a HPC coordinate frame. We will make use of the
    `astropy.coordinates` and `sunpy.coordinates` submodules together with `~sunpy.map.make_fitswcs_header`
    to create a new header and generate a `~sunpy.map.Map`.
    """
    ##############################################################################
    # We will first begin be reading in the header and data from the FITS file.
    hdu = fits.open(lofar_imagefile)
    header = hdu[0].header
    #####################################################################################
    # The data in this file is in a datacube structure to hold difference frequencies and
    # polarizations.
    data = hdu[0].data[0, 0, :, :]
    #data = hdu[0].data[:, :]
    ###############################################################################
    # Lets pull out the observation time and wavelength from the header, we will use
    # these to create our new header.
    obstime = Time(header['date-obs'])
    frequency = header['crval3']*u.Hz
    ###############################################################################
    # To create a new `~sunpy.map.Map` header we need convert the reference coordinate
    # in RA-DEC (that is in the header) to Helioprojective. To do this we will first create
    # an `astropy.coordinates.SkyCoord` of the reference coordinate from the header information.
    # We will need the location of the observer (i.e. where the observation was taken).
    # We first establish the location on Earth from which the observation takes place, in this
    # case LOFAR observations are taken from Exloo in the Netherlands, which we define in lat and lon.
    # We can convert this to a SkyCoord in GCRSat the observation time.
    lofar_loc = EarthLocation(lat=52.905329712*u.deg, lon=6.867996528*u.deg)
    lofar_gcrs = SkyCoord(lofar_loc.get_gcrs(obstime))
    ##########################################################################
    # We can then define the reference coordinate in terms of RA-DEC from the header information.
    # Here we are using the ``obsgeoloc`` keyword argument to take into account that the observer is not
    # at the center of the Earth (i.e. the GCRS origin). The distance here is the Sun-observer distance.
    reference_coord = SkyCoord(header['crval1']*u.Unit(header['cunit1']),
                               header['crval2']*u.Unit(header['cunit2']),
                               frame='gcrs',
                               obstime=obstime,
                               obsgeoloc=lofar_gcrs.cartesian,
                               obsgeovel=lofar_gcrs.velocity.to_cartesian(),
                               distance=lofar_gcrs.hcrs.distance)
    ##########################################################################
    # Now we can convert the ``reference_coord`` to the HPC coordinate frame.
    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=lofar_gcrs))
    ##########################################################################
    # Now we need to get the other parameters from the header that will be used
    # to create the new header - here we can get the cdelt1 and cdelt2 which are
    # the spatial scales of the data axes.
    cdelt1 = (np.abs(header['cdelt1'])*u.deg).to(u.arcsec)
    cdelt2 = (np.abs(header['cdelt2'])*u.deg).to(u.arcsec)
    naxis1=header['naxis1']
    naxis2=header['naxis2']
    ##################################################################################
    # Finally, we need to specify the orientation of the HPC coordinate grid because
    # GCRS north is not in the same direction as HPC north.
    P1 = sun.P(obstime)
    ##########################################################################
    # Now we can use this information to create a new header using the helper
    # function `~sunpy.map.make_fitswcs_header()`. This will create a MetaDict
    # which we contain all the necessay WCS information to create a `~sunpy.map.Map`.
    new_header = sunpy.map.make_fitswcs_header(data, reference_coord_arcsec,
                                               reference_pixel=u.Quantity([header['crpix1']-1,
                                                                           header['crpix2']-1]*u.pixel),
                                               scale=u.Quantity([cdelt1, cdelt2]*u.arcsec/u.pix),
                                               rotation_angle=-P1,
                                               wavelength=frequency.to(u.MHz).round(2),
                                               observatory=header['TELESCOP'])
    
    #Convert the CLEAN output brightness (Jy/PSF or Jy/beam) to Tb (Kelvin)
    b_maj=header['BMAJ']
    b_min=header['BMIN']
    bpa=header['BPA']
    beamArea = (b_maj/180*np.pi)*(b_min/180*np.pi)*np.pi/(4*np.log(2))
    #Details of the conversion in, i.e. F. G. Mertens et al. 2020 (arXiv:2002.07196v1)
    dataTb = data*(300/(frequency/1e6))**2/2/(1.38e-23)/1e26/beamArea
    # speed of light 3e8, MHz 1e6
    
    ##########################################################################
    # Lets create a `~sunpy.map.Map`.
    lofar_map = sunpy.map.Map(dataTb, new_header)
    lofar_map.meta['BMAJ']=b_maj
    lofar_map.meta['BMIN']=b_min
    lofar_map.meta['BPA']=bpa
    ##########################################################################
    # We can now rotate the image so that solar north is pointing up and create
    # a submap in the field of view of interest.
    lofar_map_rotate = lofar_map.rotate()
    if bottom_left is None:
        bottom_left=(-4000.,-4000.)
    if top_right is None:
        top_right=(4000.,4000.)
    bl = SkyCoord(bottom_left[0]*u.arcsec, bottom_left[1]*u.arcsec, frame=lofar_map_rotate.coordinate_frame)
    tr = SkyCoord(top_right[0]*u.arcsec, top_right[1]*u.arcsec, frame=lofar_map_rotate.coordinate_frame)
    lofar_submap = lofar_map_rotate.submap(bl, top_right=tr)
    ##########################################################################
    
    return lofar_submap

def plot_lofar_map(lofar_submap,savename=None):
    # Now lets plot this map, and overplot some contours.
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection=lofar_submap)
    lofar_submap.plot(cmap='plasma')
    lofar_submap.draw_limb()
    #lofar_submap.draw_grid()
    #lofar_submap.draw_contours(np.arange(30, 100, 5)*u.percent)
    if savename is None: savename='lofar_map.png'
    plt.savefig(savename,bbox_inches='tight')
    
    
def plot_lofar_maps(inmaps):
    #files=glob.glob(indir+'*.fits')
    for inmap in inmaps:
        plot_lofar_map(inmap)
        
def get_forward_map(forward_imagefile,bottom_left=None,top_right=None):
    """
    ========================================================================================
    Function returns a Helioprojective Map from FORWARD outputs.
    Kamen Kozarev, based on code written by Laura Hayes.
    ========================================================================================
    How to create a `~sunpy.map.Map` in Helioprojective Coordinate Frame from FORWARD model.
    In this example we read the data and header information from the FORWARD SAV file and then create 
    a new header with updated WCS information to create a `~sunpy.map.Map` with a HPC coordinate frame. 
    We will make use of the `astropy.coordinates` and `sunpy.coordinates` submodules together with 
    `~sunpy.map.make_fitswcs_header` to create a new header and generate a `~sunpy.map.Map`.
    """
    ##############################################################################
    # We will first begin be reading in the header and data from the SAV file.
    hdul = readsav(forward_imagefile)
    
    #####################################################################################
    # The data in this file is in a datacube structure
    data=np.array(hdul['quantmap'].DATA[0])
    ###############################################################################
    # Lets pull out the observation time and quantity, we will use
    # these to create our new header.
    # Now we need to get the other parameters from the header that will be used
    # to create the new header - here we can get the cdelt1 and cdelt2 which are
    # the spatial scales of the data axes.

    pxrsun=hdul['quantmap'][0][4]
    obstime=str(hdul['quantmap'][0][5]).split('\'')[1]+'T12:00:00'
    quantity=str(hdul['quantmap'][0][6]).split('!')[0].split('\'')[1]
    try:
        units=str(hdul['quantmap'][0][12]).split('\'')[1]
    except:
        units=''
    rsunasec=950.
    asecpx=rsunasec*pxrsun
    cdelt1 = asecpx
    cdelt2 = asecpx
    naxis1=hdul['gridinputs'][0][22]
    naxis2=hdul['gridinputs'][0][24]
    crpix1 = int(naxis1/2)
    crpix2 = int(naxis2/2)
    crval1=0.
    crval2=0.
    
    ###############################################################################
    # To create a new `~sunpy.map.Map` header we need convert the reference coordinate
    # to Helioprojective. To do this we will first create
    # an `astropy.coordinates.SkyCoord` of the reference coordinate from the header information.
    # We will need the location of the observer (i.e. where the observation was taken).
    reference_coord = SkyCoord(crval1*u.arcsec,crval2*u.arcsec,frame='helioprojective',obstime=obstime)
    ##########################################################################

    ##########################################################################
    # Now we can use this information to create a new header using the helper
    # function `~sunpy.map.make_fitswcs_header()`. This will create a MetaDict
    # which we contain all the necessay WCS information to create a `~sunpy.map.Map`.
    new_header = sunpy.map.make_fitswcs_header(data, reference_coord,
                                               reference_pixel=u.Quantity([crpix1,
                                                                           crpix1]*u.pixel),
                                               scale=u.Quantity([cdelt1, cdelt2]*u.arcsec/u.pix),
                                               rotation_angle=0.*u.degree,
                                               observatory='PSIMAS/FORWARD',instrument=quantity)
    ##########################################################################
    # Lets create a `~sunpy.map.Map`.
    forward_map = sunpy.map.Map(data, new_header)
    ##########################################################################
    ##########################################################################
    # We can now rotate the image so that solar north is pointing up and create
    # a submap in the field of view of interest.
    forward_map_rotate = forward_map.rotate()
    #if bottom_left is None:
    #    bottom_left=(-4000.,-4000.)
    #if top_right is None:
    #    top_right=(4000.,4000.)
    if bottom_left is not None:
        bl = SkyCoord(bottom_left[0]*u.arcsec, bottom_left[1]*u.arcsec, frame=forward_map_rotate.coordinate_frame)
    if top_right is not None:
        tr = SkyCoord(top_right[0]*u.arcsec, top_right[1]*u.arcsec, frame=forward_map_rotate.coordinate_frame)
    if bottom_left and top_right:
        forward_submap = forward_map_rotate.submap(bl, top_right=tr)
    else:
        forward_submap = forward_map_rotate
    ##########################################################################
    
    return forward_submap



    