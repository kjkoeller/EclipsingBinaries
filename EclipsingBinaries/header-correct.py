#!/usr/bin/python3
#
#########################################################################################
#                                                                                       #
# Program to search image FITS file headers to make sure the RA and Dec coordinates     #
# are formatted correctly for IRAF to read (RA -> HH:MM:SS.s, Dec -> +DD:MM:SS.s).      #
# Original header values are written into the RA_ORIG and DEC_ORIG keywords.  It will   #
# also correct any errors in the Julian Date (JD), Heliocentric Julian Date (HJD,       #
# HJD_UTC), and add a Barycentric Julian Date (BJD_TDB) into the image headers.  It     #
# will also check to make sure the EPOCH and EQUINOX keywords are in the headers.  If   #
# they are not add them with either the default value of (EPOCH -> 2000.0,              #
# EQUINOX -> J2000.0).  It will also determine various other important observational    #
# values like effective airmass (EAIRMASS), and Sidereal Time (ST) if they do not       #
# exist.                                                                                #
#                                                                                       #
#########################################################################################
#
"""
Created By: Robert C. Berrington
Created On: 08/03/2024
Last Edited By: Kyle Koeller
Last Edits ON: 08/17/2024
"""
import argparse
from astropy.io import fits
from astropy.time import Time
# from astropy import time
from astropy import coordinates as coords
from astropy import units as u # contains important time and physical units
import glob  # Needed to allow windows to expand unix style filename wildcards
import re
# import io
#
# First We need to parse the command line for settings
#
parser = argparse.ArgumentParser(prog="HJD_correct.py",
                                description="Corrects (RA, Dec) values to have :s and adds JDs, HJDs, BJDs and effecitve airmass to image headers",
                                epilog="\nThis is a program in the testing phase.  Please use with caution.  To report any issues or bugs, please contact rberring@bsu.edu",
                                add_help=True)
parser.add_argument("-f", "--file", "--filename",
                    dest='wildcard_list',
                    metavar="<filename(s)>",
                    help="input FITS filename(s)",
                    nargs='+',
                    default=['temp.fits'])
parser.add_argument("-d", "--delimiter",
                    dest='delimiter',
                    help="character to delimit the sexigesimal angle measures.",
                    metavar="<delimiter>",
                    nargs='?',
                    default=[':'],
                    const=[':'],
                    type=str)
parser.add_argument("-eq", "--equinox",
                    dest='specified_equinox',
                    nargs='?',
                    default='J2000.0',
                    const='J2000.0',
                    help="Equinox of the equitorial coordinates (RA,Dec).")
parser.add_argument("-ep", "--epoch",
                    dest='specified_epoch',
                    nargs='?',
                    default=2000.0,
                    const=2000.0,
                    help="Epoch of the equitorial coordinates (RA,Dec) in Julian years.")
parser.add_argument("-ol", "--observat", "--observatory-location",
                    dest='observat',
                    nargs='*',
                    default=['BSU'],
                    help='Set the default observatory location.  Image header will take presidence.')
parser.add_argument("-de", "--debug",
                    dest='debug',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Print out additional debugging information.")
parser.add_argument("-v", "--verbose",
                    dest='verbose',
                    action='count',
                    default=0,
                    help='Set verbosity output. Multiple entries increment verbosity level.')
parser.add_argument("--JD",
                    dest='JD_flag',
                    action=argparse.BooleanOptionalAction,
                    default=True, # default is to calculate JDs.
                    help="Calculate of JDs, and enter into the image headers.")
parser.add_argument("--HJD",
                    dest='HJD_flag',
                    action=argparse.BooleanOptionalAction,
                    default=True, # default is to calculate HJDs.
                    help="Calculate HJDs, and enter into the image headers.")
parser.add_argument("--BJD",
                    dest='BJD_flag',
                    action=argparse.BooleanOptionalAction, 
                    default=True, # default is to calculate BJDs.
                    help="Calculate BJDs, and enter into the image headers.")
parser.add_argument("-X", "--airmass", "--eairmass", "--effective-airmass",
                    dest='eairmass_flag',
                    action=argparse.BooleanOptionalAction,
                    default=True, # default is to calculate BJDs.
                    help="Calculate the effective airmass, and enter into the image headers.")
parser.add_argument("-st", "--sidereal", "--sidereal-time",
                    dest='sidereal_flag',
                    action=argparse.BooleanOptionalAction,
                    default=True, # default is to calculate sidereal times.
                    help="Calculate the current sidereal time, and enter into the image headers.")
parser.add_argument("-fp", "--filter_parse",
                    dest='filter_parse',
                    action=argparse.BooleanOptionalAction,
                    default=False, # default is not to parse filter names for spaces and replace with underscores.
                    help="Parse filter names for spaces and replace with underscores.")
parser.add_argument("-lf", "--logfile",
                    dest='log_flag',
                    action=argparse.BooleanOptionalAction, 
                    default=False, # default is to not open a logfile.
                    help="Write out operations into a log file.")
parser.add_argument("-lfn", "--logfilename",
                    dest='logfilename',
                    metavar="<logfilename>",
                    help="Output logfile name.",
                    nargs='*',
                    default=['FITS_correction.log'])
parser.add_argument("-lc", "--logfile_comment",
                    dest='header_comment',
                    metavar="<#>",
                    help="Comment character to mark the header of the logfile.",
                    nargs='*',
                    default=['#'])
parser.add_argument("-ld", "--logfile_delimiter",
                    dest='log_file_delimiter',
                    metavar="< >",
                    help="Character to delimit the values of the logfile.",
                    nargs='*',
                    default=[' '])
args = parser.parse_args()
#
# Expand out any wildcards included on the command line and append to a list of images
# to operate on.
#
max_filename_length = 0
filename_list=[]
for wildcard in args.wildcard_list:
    filename=glob.glob(wildcard)
    for file in filename:
        current_filename_length = len(file) # measure the length of the filename.
        filename_list.append(file)
        if max_filename_length < current_filename_length:
            max_filename_length = current_filename_length # store the maximum filename length.
#
# Lets fix the inability of windows to unix style file expansions with wildcards.
#
if args.debug==True or args.verbose >= 3:
    print('Files specified at the command line',args.wildcard_list)
#
# Show the list of images to be operated on.
#
if args.debug == True or args.verbose >= 2:
    print('List of images to operate on:',filename_list)
#
# setup the header strings for the log file.  These will be written to the log file later.
#
if args.log_flag:
    header_comment   = args.header_comment[0]
    header_filename  = ' Filename '
    header_JD        = (26 * ' ') + ' JD' + (27 * ' ')
    header_HJD       = (26 * ' ') + 'HJD' + (27 * ' ')
    header_BJD       = (26 * ' ') + 'BJD' + (27 * ' ')
    header_types     = (4 * ' ') + 'original' + (11 * ' ') + 'corrected' + (11 * ' ') + 'deltat' + (7 * ' ')
    header_blank     = (53 * ' ')
    header_units     = (44 * ' ') + '[sec]' + (7 * ' ')
    header_BJDHJD    = (6 * ' ') + 'BJD_TDB' + (9 * ' ') + 'BJD_TDB-HJD' + (4 * ' ')
    header_Bunits    = (25 * ' ') + '[sec]' + (7 * ' ')
    if max_filename_length > 10: # Then the file name length is longer than the 'Filename' header.
        header_line1 = header_comment + header_filename + ((max_filename_length - 10) * ' ') + '|'
        header_line2 = header_comment + (max_filename_length * ' ') + '|'
        header_line3 = header_comment + (max_filename_length * ' ') + '|'
    else: # longest file name is shorter than the 'Filename' header.
        header_line1 = header_comment + header_filename + '|'
        header_line2 = header_comment + (10 * ' ') + '|'
        header_line3 = header_comment + (10 * ' ') + '|'
    
    if args.JD_flag:
        header_line1 += header_JD + '|'
        header_line2 += header_types + '|'
        header_line3 += header_units + '|'
    if args.HJD_flag:
        header_line1 += header_HJD + '|'
        header_line2 += header_types + '|'
        header_line3 += header_units + '|'
    if args.BJD_flag:
        if args.HJD_flag: # then HJD was calculated and we need to see how HJD and BJD differ
            header_line1 += header_BJDHJD
            header_line2 += header_blank
            header_line3 += header_Bunits
        else: # HJD was never calculated, and not difference between BJD and HJD can be determined
            header_line1 += header_BJD
            header_line2 += header_blank
            header_line3 += header_blank
    header_line1 += '\n'
    header_line2 += '\n'
    header_line3 += '\n'
    change_log_delimiter = args.log_file_delimiter[0]
#
# Setup a database of known observatory sites that are not located in the Astropy known sites.
#
# First entry is the Ball State University Observatory (BSUO)
#
# IRAF entry is as follows:
#
# Cooper Science Observatory position
# Submitted by Robert C. Berrington 05/07/2013.
# Position taken from Google Earth.
# observatory = "bsu"
#       name = "Ball State University Observatory"
#       longitude = 85:24:40.62
#       latitude = 40:11:59.61
#       altitude = 289.56
#       timezone = 4
#
BSU_lat = coords.Angle('40:11:59.61 degrees')   # +=N -=Sf
BSU_long = coords.Angle('-85:24:40.62 degrees') # +=E -=W
BSU_alt = 289.56 * u.m
BSU_datum = 'WGS84'
BSU_Timezone = -4
#
# Let's open the log file for keeping track of changes made.
#
logfilename = args.logfilename[0]
delimiter = args.delimiter[0]
#
# Write the header for the log file.
#
if args.log_flag:
    change_log = open(file=logfilename, mode="w+t")
    if args.debug == True or args.verbose >= 1:
        print('Opening Logfile:', logfilename)
    if args.debug == True or args.verbose >= 2:
        print('Writing header to logfile:',logfilename)
        print(header_line1,"\n")
        print(header_line2,"\n")
        print(header_line3,"\n")
    change_log.write(header_line1)
    change_log.write(header_line2)
    change_log.write(header_line3)
# 
# Start operating on the list of files to be updated.  The list is in the variable filename_list, and
# filename contains the current image operated on.
#
for filename in filename_list:
    if args.verbose >= 1 or args.debug == True:
        print('Current open image file:',filename)
    
    current_image = fits.open(filename, mode='update')

    if args.log_flag:
        if max_filename_length >= 12:
            change_log_line = filename + change_log_delimiter + ((max_filename_length + 2 - len(change_log_delimiter) - len(filename)) * ' ')
        else:
            change_log_line = filename + change_log_delimiter + ((12 - len(change_log_delimiter) - len(filename)) * ' ')

    if 'EPOCH' in current_image[0].header:             # Test to see if the EPOCH keyword exists
        image_epoch = current_image[0].header['EPOCH'] # Read epoch for RA,Dec from image header
    else:                                              # Then it does not exist and set.
        image_epoch = args.specified_epoch
        current_image[0].header['EPOCH'] = (image_epoch,'Epoch of image coordinates')

    if 'EQUINOX' in current_image[0].header:               # Test to see if the EQUINOX keyword exists
        image_equinox = current_image[0].header['EQUINOX'] # Read epoch for RA,Dec from image header
    else:                                                  # Then it does not exist and set.
        image_equinox = args.specified_equinox
        current_image[0].header['EQUINOX'] = (image_equinox,'Equinox of image coordinates')
    
    if args.filter_parse: # if filter_parse is set to true remove the space in the filter keyword
        if 'FILT_ORG' in current_image[0].header: # then we have already corrected FILT_ORG.  Skip.
            if args.debug == True or args.verbose >= 1:
                print('We have already corrected the FLITERS parameter.  Using FILT_ORG.')
            FILTER_image = current_image[0].header['FILT_ORG']
        elif 'FILTER' in current_image[0].header:     # Test for the FILTER keyword
            FILTER_image = current_image[0].header['FILTER']
        elif 'FILTERS' in current_image[0].header:  # Test for the FILTERS keyword
                FILTER_image = current_image[0].header['FILTERS']
        else:
            print('*WARNING* no recognized FILTER keyword present in header')
            # args.filter_parse == False
    else:
        if args.debug == True or args.verbose >= 1:
            print('Skipping filter keyword fix.')

    if args.filter_parse:
        # We do not need to test to see if the FILTER or FLITERS keyword exists.  That was done above.
        if 'FILT_ORG' not in current_image[0].header:
            if ' ' in FILTER_image:
                current_image[0].header['FILT_ORG'] = (FILTER_image,'original format of image FILTER keyword')
                FILTER_reformatted = re.sub(r' ', '_', FILTER_image, count=2)
                current_image[0].header['FILTER'] = (FILTER_reformatted,'Image filter')
                current_image[0].header['FILTERS'] = (FILTER_reformatted,'Image filter')
                if args.verbose >= 1 or args.debug == True:
                    print ('FILTER_new  =', FILTER_reformatted, 'saved to header')
                    if args.verbose >= 2 or args.debug == True:
                        print('FILTER changed', FILTER_image,'->',FILTER_reformatted)
            else:
                if args.debug == True or args.verbose >= 2:
                    print('*WARNING* FILTER or FILTERS keyword does not contain a space.  Skipping...')
    #
    # Store original values from the image header for RA and DEC keywords from the image header.
    #
    if 'RA' in current_image[0].header:                  # Test to see if the RA keyword exists
        RA_image_angle = current_image[0].header['RA']   # Read the RA keyword from the image header
    else:                                                # Then it does not exist and set.
        print('*WARNING* RA keyword header does not exist.')

    if 'DEC' in current_image[0].header:                 # Test to see if the DEC keyword exists
        DEC_image_angle = current_image[0].header['DEC'] # Read the DEC keyword from the image header
    else:                                                # Then it does not exist and set.
        print('*WARNING* DEC keyword header does not exist.')
    #
    # Print out the original values for comparison if debug or verbosity are set.
    #
    if args.verbose > 0 or args.debug == True:
        print ('RA =', RA_image_angle,
               'DEC =', DEC_image_angle, 
               'EPOCH =', image_epoch,
               'EQUINOX =', image_equinox)
    #
    # First lets check RA.  I might want to change this algorithm to take advantage of the astropy angle
    # unit type.
    #
    if delimiter in RA_image_angle:
        if args.debug:
            print('Image:', filename, 'has RA is in the correct format.')
    else:
        RA_reformatted_angle = re.sub(r' ', delimiter, RA_image_angle, count=2)
        if args.verbose > 0 or args.debug == True:
            print ('RA_new  = ', RA_reformatted_angle, 'saved to header')
        current_image[0].header['RA_ORIG'] = (RA_image_angle,'original format of image RA coordinate')
        current_image[0].header['RA'] = RA_reformatted_angle
    #    
    # Now Lets check DEC.  I might want to change this algorithm to take advantage of the astropy angle
    # unit type.
    #
    if delimiter in DEC_image_angle:
        if args.debug:
            print('Image:', filename, 'has DEC in the correct format.')
    else:
        DEC_reformatted_angle = re.sub(r' ', delimiter, DEC_image_angle, count=2)
        if args.verbose > 0 or args.debug == True:
            print ('DEC_new =', DEC_reformatted_angle, 'saved to header')
        current_image[0].header['DEC_ORIG'] = (DEC_image_angle,'original format of image DEC coordinate')
        current_image[0].header['DEC'] = DEC_reformatted_angle
    #
    # Print out the values saved to the header if debug or verbosity are set.
    #
    if args.verbose > 0 or args.debug == True:
        print ('RA =',current_image[0].header['RA'],
               'DEC =',current_image[0].header['DEC'], 
               'EPOCH =', image_epoch,
               'EQUINOX =', image_equinox)
    #
    # if either HJD (default: True), BJD (default: True), Sidereal time (default: True), or
    # effective airmass (default: True) are asked to be calculated, then determine the observatory location and
    # set for later use.
    #
    if args.HJD_flag == True or args.BJD_flag == True or args.sidereal_flag == True or args.eairmass_flag == True:
        if 'OBSERVAT' in current_image[0].header:
            observer_at = current_image[0].header['OBSERVAT']
        else:
            observer_at = args.observat[0]
            current_image[0].header['OBSERVAT'] = observer_at
            print('*WARNING* OBSERVAT keyword must be set for file:', filename)
            print('Using value:', observer_at)
        
        if (observer_at == 'sara-kp') or (observer_at == 'SARA-KP'):
            observatory_location = coords.EarthLocation.of_site('kpno')
            if args.debug:
                print('Using observatory location', observer_at)
        elif (observer_at == 'sara-n') or (observer_at == 'SARA-N'):
            observatory_location = coords.EarthLocation.of_site('kpno')
            if args.debug:
                print('Using observatory location', observer_at)
        elif (observer_at == 'sara-ct') or (observer_at == 'SARA-CT'):
            observatory_location = coords.EarthLocation.of_site('ctio')
            if args.debug:
                print('Using observatory location', observer_at)
        elif (observer_at == 'sara-s') or (observer_at == 'SARA-S'):
            observatory_location = coords.EarthLocation.of_site('ctio')
            if args.debug:
                print('Using observatory location', observer_at)
        elif (observer_at == 'sara-rm') or (observer_at == 'SARA-RM'):
            observatory_location = coords.EarthLocation.of_site('Roque de los Muchachos')
            if args.debug:
                print('Using observatory location', observer_at)
        elif (observer_at == 'bsu') or (observer_at == 'BSU'):
            observatory_location = coords.EarthLocation.from_geodetic(lon=BSU_long,
                                                                      lat=BSU_lat,
                                                                      height=BSU_alt,
                                                                      ellipsoid=BSU_datum)
            if args.debug:
                print('Using observatory location', observer_at)
        elif (observer_at == 'bsuo') or (observer_at == 'BSUO'):
            observatory_location = coords.EarthLocation.from_geodetic(lon=BSU_long,
                                                                      lat=BSU_lat,
                                                                      height=BSU_alt,
                                                                      ellipsoid=BSU_datum)
            if args.debug:
                print('Using observatory location', observer_at)
        else:
            print('WARNING: Unknown observatory location')
            print('Defaulting to the Ball State University Observatory.')
            observatory_location = coords.EarthLocation.from_geodetic(lon=BSU_long,
                                                                      lat=BSU_lat,
                                                                      height=BSU_alt,
                                                                      ellipsoid=BSU_datum)
        if args.debug:
            print('OBSERVAT =', observer_at)
            print('Location set', observer_at, 'to', observatory_location)
    
        date = Time(current_image[0].header['DATE-OBS'], scale='utc', format='fits', location=observatory_location)
    else: # then neither HJD, BJD, EAIRMASS, or ST are calculated, and we do not need observer location.
        date = Time(current_image[0].header['DATE-OBS'], scale='utc', format='fits')
    #
    # Read exposure time from header. If keyword is not present assume exposure time = 0 sec.
    #
    if 'EXPTIME' in current_image[0].header:
        exp_time = current_image[0].header['EXPTIME'] * u.second
    elif 'EXP_TIME' in current_image[0].header:
        exp_time = current_image[0].header['EXP_TIME'] * u.second
    else:
        print('*WARNING* no exposure time set. Assume exposure time = 0 sec.')
        exp_time = 0 * u.second
    #
    # Determine start, mid and end exposure times, and if debug or verbosity set print out times.
    #
    half_exp_time = exp_time / 2.0
    date_at_half_exptime = date + half_exp_time # correct time to the midpoint of the exposure.
    date_at_end = date + exp_time
    if args.debug == True or args.verbose >= 2:
        print ('Exposure times for current image:', filename)
        print ('date begin =', date.fits)
        print ('date mid.  =', date_at_half_exptime.fits)
        print ('date end   =', date_at_end.fits)
        print ('exptime    =', exp_time, 'half exptime =', half_exp_time)
   
    delta_t_half_exp = date_at_half_exptime - date
    if args.debug == True or args.verbose >= 2:
        print ('JD      =', date.jd, 'JD +1/2 =', date_at_half_exptime.jd)
        print ('delta_t =', delta_t_half_exp.jd, 'sec =', (delta_t_half_exp.jd * 86400.0))
    #
    # Calculate the Julian Date (JD).  This will require positional information like
    # date and time of observation that is read above.
    #
    if args.JD_flag:
        if 'JD_START' in current_image[0].header:  # We have already run the script.  Save the original value.
            JD_original = current_image[0].header['JD_START'] # Then store the value for later use
        elif 'JD' in current_image[0].header:
            JD_original = current_image[0].header['JD'] # Then store the value for later use
            if 'JD_ORIG' in current_image[0].header:  # Then we have already run the script on this image.
                if args.debug == True or args.verbose >= 3:
                    print('*NOTE* JD correction has already been run on file:', filename)
                    print('Not saving JD_ORIG keyword and value to header.')
            else:  # Then we have not run the script.
                current_image[0].header['JD_ORIG'] = (JD_original, 'Original JD value in image.')
        else:  #  JD does not exist.  Using mid exposure time for JD time.
            JD_original = date.jd # then store the JD and mid exposure for later use.
            if 'JD_ORIG' not in current_image[0].header:
                current_image[0].header['JD_ORIG'] = (date.jd, 'Original JD from header.')
            else:
                if args.debug == True or args.verbose >= 3:
                    print('*WARNING* JD_ORIG already exists.  Not altering original value.')
        current_image[0].header['JD'] = (date_at_half_exptime.jd, 'Julian Date at mid exposure.')
        current_image[0].header['JD_MID'] = (date_at_half_exptime.jd, 'Julian Date at mid exposure.')
        current_image[0].header['JD_START'] = (date.jd, 'Julian Date at exposure start.')
        current_image[0].header['JD_END'] = (date_at_end.jd, 'Julian Date at exposure end.')
        JD = date_at_half_exptime
        delta_t_JD = JD.jd - JD_original
        if args.debug:
           print('JD corrected by', delta_t_JD,'sec =',(delta_t_JD * 86400.0))
        if args.log_flag:
            change_log_line += str(JD_original) + change_log_delimiter
            change_log_line += str(JD.jd) + change_log_delimiter
            change_log_line += str(delta_t_JD * 86400.0) + change_log_delimiter
    else:
        #
        # calculate JD was set to False, and do not calculate.
        #
        if args.verbose >= 1 or args.debug == True:
            print('Skipping JD calculation')
    #
    # Now calculate the HJDs and enter into the headers.  This will require extracting
    # positional information like observatory location, RA and Dec of target.
    #
    if image_epoch == 2000.0 or image_equinox == 'J2000.0':
        target_object = coords.SkyCoord(current_image[0].header['RA'],
                                        current_image[0].header['DEC'],
                                        unit=(u.hourangle, u.deg),
                                        obstime=date_at_half_exptime,
                                        frame='icrs')
    else:  # Not sure if this will work so it remains untested and not currently supported.
        target_object = coords.SkyCoord(current_image[0].header['RA'],
                                        current_image[0].header['DEC'],
                                        unit=(u.hourangle, u.deg),
                                        obstime=date_at_half_exptime,
                                        frame=image_equinox)

    if args.HJD_flag:
        HJD_correction = date_at_half_exptime.light_travel_time(target_object, 'heliocentric')
        if 'HJD' in current_image[0].header: # then HJDs are already in the headers as store.
            HJD_original = current_image[0].header['HJD']
            if 'HJD_ORIG' in current_image[0].header: # Then we have already run the HJD correction.
                if args.debug == True or args.verbose >= 3:
                    print('*NOTE* HJD correction has already been run on file:', filename)
                    print('Not saving HJD_ORIG keyword and value to header.')
            else: # we have not run the HJD correction.
                current_image[0].header['HJD_ORIG'] = (HJD_original, 'Original HJD value in header.')
        else: # Then HJDs do not exist
            if args.debug == True or args.verbose >= 1:
                print('*WARNING* HJDs do not exist in file:', filename)
            HJD_original = date.jd + HJD_correction.jd
            if 'HJD_ORIG' not in current_image[0].header:
                current_image[0].header['HJD_ORIG'] = (HJD_original.jd, 'HJD value from exp start.')
        HJD = date_at_half_exptime + HJD_correction
        if args.debug:
            print('JD',date_at_half_exptime.jd,'+ correction',HJD_correction.jd, '= HJD:',HJD.jd)
        current_image[0].header['HJD'] = (HJD.jd, 'HJD_UTC at mid exposure')
        current_image[0].header['HJD_UTC'] = (HJD.jd, 'HJD_UTC at mid exposure')
        delta_t_HJD = HJD.jd - HJD_original
        if args.debug:
           print('HJD corrected by', delta_t_HJD, 'sec =', (delta_t_HJD * 86400.0))
        if args.log_flag:
            change_log_line += str(HJD_original) + change_log_delimiter
            change_log_line += str(HJD.jd) + change_log_delimiter
            change_log_line += str(delta_t_HJD * 86400.0) + change_log_delimiter
    else:
        #
        # calculate HJD was set to False, and do not calculate.
        #
        if args.verbose >= 1 or args.debug == True:
            print('Skipping HJD calculation')
    #
    # Now calculate the BJDs and enter into the headers.  This will require extracting
    # positional information like observatory location, RA and Dec of target.
    #
    if args.BJD_flag:
        BJD_correction = date_at_half_exptime.light_travel_time(target_object)
        BJD_UTC = date_at_half_exptime.utc + BJD_correction
        BJD_TDB = date_at_half_exptime.tdb + BJD_correction
        if 'BJD' in current_image[0].header: # Then BJDs calculated and store for later use.
            BJD_original = current_image[0].header['BJD']
            if args.debug:
                print('Extracting BJD from header as original BJD.')
        elif 'BJD_TDB' in current_image[0].header: # Then BJDs calculated and store for later use.
            BJD_original = current_image[0].header['BJD_TDB']
            if args.debug:
                print('Extracting BJD_TDB from header as original BJD.')
        elif 'BJD_UTC' in current_image[0].header: # Then BJDs calculated and store for later use.
            BJD_original = current_image[0].header['BJD_UTC']
            if args.debug:
                print('Extracting BJD_UTC from header as original BJD.')
        if args.debug:
            print('JD',date_at_half_exptime.jd,' + correction',BJD_correction.jd,'= BJD:',BJD_TDB.jd)
        current_image[0].header['BJD_UTC'] = (BJD_UTC.jd, 'BJD_UTC at mid exposure')
        current_image[0].header['BJD_TDB'] = (BJD_TDB.jd, 'BJD_TDB at mid exposure')
        if args.HJD_flag: # HJDs are determined and use to see difference
            delta_t_BJDHJD = BJD_TDB.jd - HJD.jd
            if args.debug:
                print('BJD_TDB - HJD =', delta_t_BJDHJD, ': sec =', (delta_t_BJDHJD * 86400.0))
        else: # HJDs are not determined and use BJD_UTC to see difference
            delta_t_BJD = BJD_TDB.jd - BJD_UTC.jd
            if args.debug:
                print('BJD_TDB - BJD_UTC =', delta_t_BJD, ': sec =', (delta_t_BJD * 86400.0))
        if args.log_flag:
            change_log_line += str(BJD_TDB.jd) + change_log_delimiter
            change_log_line += str(delta_t_BJDHJD * 86400.0) # This is the last line added to the change log.  No need for a delimiter.
    else:
        #
        # calculate BJD was set to False, and do not calculate.
        #
        if args.verbose >= 1 or args.debug == True:
            print('Skipping BJD calculation')
    #
    # Calculate the Sideral time and enter in to the header if the sidereal flag option is set.
    # This requires information like observatory location and time of exposure information.
    #
    if args.sidereal_flag:
        if args.debug == True or args.verbose >= 2:
            print('Calculating Sidereal time.')
        
        if 'ST_ORIG' in current_image[0].header:   # Then we have already run the sidereal calculation on this image.  Skip.
            if args.debug == True or args.verbose >=1:
                print('Sidereal Time calculation has already been run.  Usnig ST_ORIG as original value.')
            sidereal_time_original = current_image[0].header['ST_ORIG']
        else:  # Then this is our first time running this program.
            if 'SIDEREAL' in current_image[0].header:  # Then the SIDEREAL keyword exist.
                sidereal_time_original = current_image[0].header['SIDEREAL']
            elif 'ST' in current_image[0].header:      # Then the ST keyword exists
                sidereal_time_original = current_image[0].header['ST']
            else:
                # sidereal_time_original is None
                if args.debug == True or args.verbose >=2:
                    print('No sidereal time keyword exists in image header of file:',filename)

        mean_sidereal_time = Time.sidereal_time(date_at_half_exptime, kind='mean', longitude=observatory_location, model='IAU2006')
        apparent_sidereal_time = Time.sidereal_time(date_at_half_exptime, kind='apparent', longitude=observatory_location, model='IAU2006A')

        if sidereal_time_original is not None:  # Then the original sidereal time from the image header existed, write to keyword ST_ORIG
            current_image[0].header['ST_ORIG'] = (sidereal_time_original, 'Original sidereal time at exp start.')
        current_image[0].header['SIDEREAL']    = (apparent_sidereal_time.to_string(sep=':'), 'Local app sidereal time at exp midpt [IAU2006A]')
        current_image[0].header['MEAN_ST']     = (mean_sidereal_time.to_string(sep=':'), 'local mean sidereal time at exp midpt [IAU2006]')
        current_image[0].header['APP_ST']      = (apparent_sidereal_time.to_string(sep=':'), 'local app sidereal time at exp midpt [IAU2006A]')
        current_image[0].header['ST']          = (apparent_sidereal_time.to_string(sep=':'), 'local app sidereal time at exp midpt [IAU2006A]')
        if args.debug == True or args.verbose >= 2:
            print('Wrote LMST:', mean_sidereal_time, 'and LAST:', apparent_sidereal_time, 'to file:', filename)
    else:
        if args.verbose >= 1 or args.debug == True:
            print('Skipping sidereal time calculation.')
    #
    # Calculate the effictive airmass of the object at the time of mid exposure. This will
    # require obervatory location, date and time of obseration, and object location.
    #
    if args.eairmass_flag:
        if args.debug == True or args.verbose >= 1:
            print('Calculating Effictve Airmass.')
        horizon_coordinates = target_object.transform_to(coords.AltAz(obstime=date_at_half_exptime, location=observatory_location))
        secz = horizon_coordinates.secz

        seczminusone = (secz - 1.0)
        eairmass = secz - 0.0018167 * seczminusone - 0.002875 * seczminusone * seczminusone - 0.0008083 * seczminusone * seczminusone * seczminusone
        # Check to see if SECZ key header already exists in the header.
        if 'SECZ' in current_image[0].header:  # SECZ is present in the header
            secz_original = current_image[0].header['SECZ']
            if args.debug == True or args.verbose >= 3:
                print('in image SECZ:', secz_original, 'calculated SECZ:', secz)
            current_image[0].header['SECZ_ORG'] = (float(secz_original), 'Orignial value of SecZ in image header.')
            current_image[0].header['SECZ'] = (float(secz), 'SecZ for airmass estimation.')
        else:                                  # SECZ is not present in the header
            if args.debug == True or args.verbose >= 3:
                print('Calculated SECZ:', secz)
            current_image[0].header['SECZ'] = (float(secz), 'SecZ for airmass estimation.')
        # Determine if the EAIRMASS keyword is present.
        if 'EAIRMASS' in current_image[0].header:  # Then the EAIRMASS has already been determined.
            if args.verbose >= 1 or args.debug == True:
                print('*WARNING* Effective airmass has already been determined.')
            eairmass_original = current_image[0].header['EAIRMASS']
            if args.debug == True or args.verbose >= 1:
                print('EAIRMASS:', eairmass_original,'->', eairmass, 'delta_ea:', eairmass_original - eairmass)
        else:                                      # Then we have not calculated the effective airmass.
            if args.verbose >= 3 or args.debug == True:
                print("keyword EAIRMASS does not exist.  Writing EAIRMASS:", eairmass)
        current_image[0].header['EAIRMASS'] = (float(eairmass), 'Airmass at mid exposure.')
        if args.debug == True or args.verbose >= 2:
            print('Wrote EAIRMASS:', eairmass)
    else:
        if args.debug == True or args.verbose >= 1:
            print('Skipping effective airmass calculation.')
    #
    # Close the current image, if verbosity or degug is set then announce successful closing.
    #
    if args.verbose >= 1 or args.debug == True:
        current_image.close(verbose = True)
    else:
        current_image.close()
    #
    # terminate the line for this image and write to log file is log file option set.
    #
    if args.log_flag:
        change_log_line += '\n'
        change_log.write(change_log_line)
#
# Close the log file, and if debug or verbosity set (>= 2) then test to make sure it was
# closed successfully.
#
if args.log_flag:
    change_log.close()
    if change_log.closed:
        if args.debug == True or args.verbose >= 2:
            print('Closed file:', logfilename)
    else:
        if args.debug == True or args.verbose >= 2:
            print('*WARNING* Failed to close file:', logfilename)
#
# announce completion of the script.
#
if args.debug == True or args.verbose >= 1:
    print('Finished!')
