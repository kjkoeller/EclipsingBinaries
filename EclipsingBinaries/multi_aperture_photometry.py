"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 04/19/2026
"""

# Python imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import warnings
from tqdm import tqdm

# Astropy imports
import ccdproc as ccdp
from astropy.coordinates import SkyCoord
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.wcs import WCS
import astropy.units as u
from astropy import wcs

# Suppress FITS header standards-compliance warnings that are not actionable
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)

# Use non-interactive backend so plots can be saved without a display
matplotlib.use('Agg')


def main(path="", pipeline=False, radec_list=None, obj_name="", write_callback=None, cancel_event=None):
    """
    Entry point for multi-aperture photometry. Iterates over each filter and its
    corresponding RADEC file, dispatching to multiple_AP for the actual photometry.

    Parameters
    ----------
    path : str
        Path to the folder containing the reduced images.
    pipeline : bool
        If True, suppresses prompts and expects all inputs to be provided programmatically.
    radec_list : list
        List of RADEC file paths, one per filter in the same order as filt_list.
    obj_name : str
        Name of the target object, used for output file naming.
    write_callback : callable
        Function to route log messages to a GUI. Falls back to print if None.
    cancel_event : threading.Event
        Event flag that allows external cancellation of the running task.
    """
    # Filters expected in the image collection, matched positionally with radec_list
    filt_list = ["Empty/B", "Empty/V", "Empty/R"]

    def log(message):
        if write_callback:
            write_callback(message)
        else:
            print(message)

    try:
        images_path = Path(path)
        files = ccdp.ImageFileCollection(images_path)

        for filt, radec_file in zip(filt_list, radec_list):
            # Respect cancellation between filters so the task exits cleanly
            if cancel_event and cancel_event.is_set():
                log("Multi-Aperture Photometry was canceled.")
                return

            image_list = files.files_filtered(imagetyp='LIGHT', filter=filt)
            log(f"Processing {len(image_list)} images for {filt} filter.")

            multiple_AP(
                image_list=image_list,
                path=images_path,
                filt=filt,
                pipeline=pipeline,
                obj_name=obj_name,
                radec_file=radec_file,
                write_callback=write_callback,
                cancel_event=cancel_event
            )

    except Exception as e:
        log(f"An error occurred in Multi-Aperture Photometry: {e}")


def multiple_AP(image_list, path, filt, pipeline=False, obj_name="", radec_file="",
                write_callback=None, cancel_event=None):
    """
    Perform aperture photometry on all images for a single filter. Comparison stars
    that fall outside the image bounds are automatically removed and the process
    restarts from the first image until a stable set of comparison stars is found.

    Parameters
    ----------
    image_list : list
        List of image filenames to process.
    path : Path
        Path to the folder containing the images.
    filt : str
        Filter label (e.g. 'Empty/B').
    pipeline : bool
        If True, suppresses user prompts.
    obj_name : str
        Target object name used for output file naming.
    radec_file : str
        Path to the RADEC file for the current filter.
    write_callback : callable
        Function to route log messages to a GUI.
    cancel_event : threading.Event
        Event flag that allows external cancellation of the running task.
    """

    def log(message, pbar=None):
        """
        Route a log message appropriately depending on context.
        In terminal mode the message is written through tqdm so it appears
        above the progress bar rather than interleaving with it.
        In GUI mode the write_callback is used directly and tqdm is bypassed.
        """
        if write_callback:
            write_callback(message)
        elif pbar is not None:
            pbar.write(message)
        else:
            tqdm.write(message)
    
    if cancel_event and cancel_event.is_set():
        log("Task canceled during photometry.")
        return

    # Aperture and annulus sizes in pixels
    aperture_radius = 20
    annulus_radii = (30, 50)

    # Read noise from the detector, used in flux error calculations
    read_noise = 10.83  # electrons, sourced from FITS headers

    try:
        # ---------------------------------------------------------------------------
        # Load the RADEC file
        # Row 0 is the target star; all subsequent rows are comparison stars.
        # Columns: RA (h), Dec (deg), <unused>, <unused>, catalog magnitude
        # ---------------------------------------------------------------------------
        df = pd.read_csv(radec_file, skiprows=7, sep=",", header=None)

        all_ra  = df[0].values
        all_dec = df[1].values
        all_mag = df[4].values

        # Separate the target star from the comparison stars
        target_ra  = all_ra[0]
        target_dec = all_dec[0]
        comp_ra_all  = all_ra[1:]
        comp_dec_all = all_dec[1:]
        comp_mag_all = all_mag[1:]

        # ---------------------------------------------------------------------------
        # Build the initial set of valid comparison star indices.
        # Stars with a catalog magnitude of 99.999 (no valid magnitude) are excluded
        # immediately so the magnitude and flux arrays stay in sync throughout all
        # subsequent processing and potential restarts.
        # ---------------------------------------------------------------------------
        valid_mag_mask = np.array([(m != 99.999 and not pd.isna(m)) for m in comp_mag_all])
        valid_comp_indices = list(np.where(valid_mag_mask)[0])

        # Inform the user of any stars excluded due to missing catalog magnitudes
        for idx in np.where(~valid_mag_mask)[0]:
            log(
                f"[EXCLUDED] Comparison star #{idx + 1} "
                f"(RA={comp_ra_all[idx]}, Dec={comp_dec_all[idx]}) "
                f"removed before processing: magnitude is 99.999 (no valid catalog magnitude)."
            )

        # ---------------------------------------------------------------------------
        # Main processing loop. If a comparison star is found to be outside the
        # bounds of any image, it is removed and the loop restarts from image 1
        # with the reduced set. This continues until all remaining comparison stars
        # are within bounds for every image in the list.
        # ---------------------------------------------------------------------------
        restart = True
        while restart:
            restart = False

            if not valid_comp_indices:
                log("ERROR: All comparison stars have been removed. Cannot continue photometry.")
                return

            # Slice coordinate and magnitude arrays down to currently valid stars
            cur_comp_ra  = comp_ra_all[valid_comp_indices]
            cur_comp_dec = comp_dec_all[valid_comp_indices]
            cur_comp_mag = comp_mag_all[valid_comp_indices]

            # Build a clean float Series of catalog magnitudes for the active comp stars
            magnitudes_comp = pd.Series(cur_comp_mag.astype(float)).reset_index(drop=True)

            # Accumulators for per-image results
            magnitudes = []
            mag_err    = []
            hjd        = []
            bjd        = []

            # Use tqdm in terminal mode only; the GUI receives progress via write_callback
            use_tqdm  = not write_callback
            image_iter = (
                tqdm(image_list, desc=f"Processing {filt} images") if use_tqdm else image_list
            )

            for icount, image_file in enumerate(image_iter):
                pbar = image_iter if use_tqdm else None

                # Check for external cancellation at the start of each image
                if cancel_event and cancel_event.is_set():
                    log("Task canceled during photometry.", pbar)
                    if use_tqdm:
                        image_iter.close()
                    return

                log(f"Processing {image_file}", pbar)

                # Load the image data and header
                image_data, header = fits.getdata(path / image_file, header=True)
                img_height, img_width = image_data.shape

                # Build the WCS from the image header for sky-to-pixel conversion
                wcs_ = WCS(header)

                # Convert target RA/Dec to pixel coordinates
                target_sky   = SkyCoord(target_ra, target_dec, unit=(u.h, u.deg), frame='icrs')
                target_pixel = wcs_.world_to_pixel(target_sky)
                target_position = (float(target_pixel[0]), float(target_pixel[1]))

                # Convert all active comparison star RA/Dec values to pixel coordinates
                comp_sky   = SkyCoord(cur_comp_ra, cur_comp_dec, unit=(u.h, u.deg), frame='icrs')
                comp_pixels = wcs_.world_to_pixel(comp_sky)
                comp_x = np.array(comp_pixels[0], dtype=float)
                comp_y = np.array(comp_pixels[1], dtype=float)

                # ---------------------------------------------------------------------------
                # Bounds check: verify every comparison star's aperture and annulus fit
                # entirely within the image. The outer annulus radius is used as the margin
                # so that background estimation is never computed from partial data.
                # ---------------------------------------------------------------------------
                margin = annulus_radii[1]
                out_of_bounds_idx = None

                for i, (cx, cy) in enumerate(zip(comp_x, comp_y)):
                    x_in_bounds = margin <= cx < (img_width  - margin)
                    y_in_bounds = margin <= cy < (img_height - margin)
                    if not x_in_bounds or not y_in_bounds:
                        out_of_bounds_idx = i
                        break

                if out_of_bounds_idx is not None:
                    # Record which original star (1-based, relative to the RADEC file) was removed
                    orig_idx = valid_comp_indices[out_of_bounds_idx]
                    log(
                        f"\n[REMOVED] Comparison star #{orig_idx + 1} "
                        f"(RA={cur_comp_ra[out_of_bounds_idx]}, Dec={cur_comp_dec[out_of_bounds_idx]}) "
                        f"is out of bounds in '{image_file}'.\n"
                        f"  Pixel position: x={comp_x[out_of_bounds_idx]:.1f}, "
                        f"y={comp_y[out_of_bounds_idx]:.1f} | "
                        f"Image size: {img_width}x{img_height} | "
                        f"Required margin: {margin}px\n"
                        f"  Restarting photometry without this star...",
                        pbar
                    )
                    valid_comp_indices.pop(out_of_bounds_idx)
                    if use_tqdm:
                        image_iter.close()
                    restart = True
                    break  # Exit image loop and re-enter the while loop

                # Build the list of (x, y) positions for all active comparison stars
                comparison_positions = list(zip(comp_x, comp_y))

                # Record the timestamps from the header for this image
                hjd.append(header['HJD-OBS'])
                bjd.append(header['BJD-OBS'])

                # ---------------------------------------------------------------------------
                # Define apertures and annuli for the target and each comparison star
                # ---------------------------------------------------------------------------
                target_aperture  = CircularAperture(target_position, r=aperture_radius)
                target_annulus   = CircularAnnulus(target_position, *annulus_radii)

                comparison_aperture = [
                    CircularAperture(pos, r=aperture_radius) for pos in comparison_positions
                ]
                comparison_annulus = [
                    CircularAnnulus(pos, *annulus_radii) for pos in comparison_positions
                ]

                # ---------------------------------------------------------------------------
                # Perform aperture photometry for the target star
                # ---------------------------------------------------------------------------
                target_phot_table = aperture_photometry(image_data, target_aperture)

                # ---------------------------------------------------------------------------
                # Perform aperture photometry for each comparison star individually so
                # background and flux can be computed per-star
                # ---------------------------------------------------------------------------
                comparison_phot_table = []
                for comp_ap, comp_an in zip(comparison_aperture, comparison_annulus):
                    comparison_phot_table.append((
                        aperture_photometry(image_data, comp_ap),
                        aperture_photometry(image_data, comp_an)
                    ))

                # ---------------------------------------------------------------------------
                # Estimate the sky background for the target star via annulus statistics.
                # Fall back to zero if the mean is non-finite (e.g. star near the edge).
                # ---------------------------------------------------------------------------
                target_bkg_mean = ApertureStats(image_data, target_annulus).mean
                if np.isnan(target_bkg_mean) or np.isinf(target_bkg_mean):
                    target_bkg_mean = 0.0

                # Estimate sky background for each comparison star in the same way
                comparison_bkg_mean = []
                for annulus in comparison_annulus:
                    stats = ApertureStats(image_data, annulus)
                    if np.isnan(stats.mean) or np.isinf(stats.mean):
                        comparison_bkg_mean.append(0.0)
                    else:
                        comparison_bkg_mean.append(stats.mean)

                # ---------------------------------------------------------------------------
                # Extract aperture sums to plain Python floats immediately after photometry.
                # astropy Table columns can carry shape metadata that causes ambiguous
                # behaviour in downstream arithmetic if left as Column objects.
                # ---------------------------------------------------------------------------

                # Background-subtracted flux for the target
                target_aperture_sum = float(target_phot_table['aperture_sum'][0])
                target_bkg  = target_bkg_mean * target_aperture.area
                target_flx  = target_aperture_sum - float(target_bkg)

                # Poisson + read-noise flux error for the target
                target_flux_err = np.sqrt(target_aperture_sum + target_aperture.area * read_noise ** 2)

                # Background-subtracted flux for each comparison star
                comparison_flx = [
                    float(phot[0]['aperture_sum'][0]) - float(bkg_mean * aperture.area)
                    for phot, bkg_mean, aperture in zip(
                        comparison_phot_table, comparison_bkg_mean, comparison_aperture
                    )
                ]

                # Poisson + read-noise flux error for each comparison star as a clean 1D float array
                comp_flux_err = np.array([
                    float(np.sqrt(float(phot[0]['aperture_sum'][0]) + aperture.area * read_noise ** 2))
                    for phot, aperture in zip(comparison_phot_table, comparison_aperture)
                ])

                # Total flux across all active comparison stars, used in several calculations below
                total_comp_flx = sum(comparison_flx)

                # ---------------------------------------------------------------------------
                # Relative flux for each comparison star: its own flux divided by the sum
                # of all other comparison stars' fluxes (i.e. excluding itself).
                # This gives a check-star light curve for each comparison star.
                # ---------------------------------------------------------------------------
                rel_flux_comps = [flx / (total_comp_flx - flx) for flx in comparison_flx]

                # ---------------------------------------------------------------------------
                # Target magnitude using the ensemble comparison method.
                # The first term converts the catalog magnitudes of the comparison stars
                # into a reference magnitude for the ensemble. The second term applies
                # the differential flux ratio between target and ensemble.
                # ---------------------------------------------------------------------------
                target_magnitude = (
                    (-np.log(sum(2.512 ** -magnitudes_comp)) / np.log(2.512)) -
                    (2.5 * np.log10(target_flx / total_comp_flx))
                )

                # Propagated magnitude error from flux uncertainties of target and ensemble
                target_magnitude_error = 2.5 * np.log10(1 + np.sqrt(
                    (target_flux_err ** 2 / target_flx ** 2) +
                    (np.sum(comp_flux_err ** 2) / total_comp_flx ** 2)
                ))

                magnitudes.append(float(target_magnitude))
                mag_err.append(float(target_magnitude_error))

        # ---------------------------------------------------------------------------
        # Plot the resulting light curve with error bars and save it to disk
        # ---------------------------------------------------------------------------
        _, ax = plt.subplots(figsize=(11, 8))
        ax.errorbar(hjd, magnitudes, yerr=mag_err, fmt='o', label="Source_AMag_T1")

        fontsize = 14
        ax.set_xlabel('HJD', fontsize=fontsize)
        ax.set_ylabel('Source_AMag_T1', fontsize=fontsize)
        ax.invert_yaxis()  # Magnitudes increase downward by convention
        ax.grid()
        ax.legend(loc="upper right", fontsize=fontsize).set_draggable(True)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        filter_letter = filt.split("/")
        plt.savefig(path / f"{obj_name}_{filter_letter[1]}_figure.jpg")
        plt.close()

        # ---------------------------------------------------------------------------
        # Save the light curve data to a CSV file
        # ---------------------------------------------------------------------------
        light_curve_data = pd.DataFrame({
            'HJD': hjd,
            'BJD': bjd,
            'Source_AMag_T1': magnitudes,
            'Source_AMag_T1_Error': mag_err
        })

        output_file = path / f"{obj_name}_{filter_letter[1]}_data.csv"
        light_curve_data.to_csv(output_file, index=False)
        log(f"Saved light curve data for {filt} to {output_file}")

    except Exception as e:
        log(f"Error during photometry for {filt}: {e}")


def im_plot(image_data, target_aperture, comparison_apertures, target_annulus, comparison_annuli):
    """
    Diagnostic plot of a single image with all apertures and annuli overlaid.
    Useful for visually verifying that aperture positions are correct before
    running the full photometry pipeline.

    Parameters
    ----------
    image_data : array
        Pixel data from the FITS image.
    target_aperture : CircularAperture
        Aperture centred on the target star.
    comparison_apertures : list of CircularAperture
        Apertures centred on each comparison star.
    target_annulus : CircularAnnulus
        Background annulus for the target star.
    comparison_annuli : list of CircularAnnulus
        Background annuli for each comparison star.
    """
    plt.figure(figsize=(8, 8))

    # Display the image with a percentile stretch to avoid saturation dominating the scale
    plt.imshow(
        image_data,
        cmap='gray',
        origin='lower',
        vmin=np.percentile(image_data, 5),
        vmax=np.percentile(image_data, 95)
    )
    plt.colorbar(label='Counts')

    lw    = 1.5  # Line width for aperture outlines
    alpha = 1.0  # Opacity for aperture outlines

    # Target star aperture and annulus in green
    target_aperture.plot(color='darkgreen', lw=lw, alpha=alpha)
    target_annulus.plot(color='darkgreen', lw=lw, alpha=alpha)

    # Comparison star apertures and annuli in red
    for comp_ap, comp_an in zip(comparison_apertures, comparison_annuli):
        comp_ap.plot(color='red', lw=lw, alpha=alpha)
        comp_an.plot(color='red', lw=lw, alpha=alpha)

    plt.pause(1)
    plt.show()


if __name__ == '__main__':
    main()
