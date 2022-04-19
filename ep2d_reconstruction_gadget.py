from __future__ import division

import ismrmrd
import ismrmrd.xsd
import math
import numpy as np
import scipy as sp
import scipy.interpolate as interp
import scipy.signal
import os

import sys
from ismrmrdtools import coils, transform, show, sense
import scipy.signal
import matplotlib.pyplot as plt

# # # #
import gadgetron
import logging
import time
import matplotlib.pyplot as plt
import nibabel as nib

from gadgetron.types.image_array import ImageArray

from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # #    AIDING FUNCTIONS   # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_first_index_of_non_empty_header(header):
    # if the data is under-sampled, the corresponding acquisition Header will be filled with 0
    # in order to catch valuable information, we need to catch an non-empty header
    # using the following lines

    print(np.shape(header))
    dims = np.shape(header)
    for ii in range(0, dims[0]):
        # print(header[ii].scan_counter)
        if header[ii].scan_counter > 0:
            break
    print(ii)
    return ii


def send_reconstructed_images(connection, acq, data_array):
    # the function creates an new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisition Header
    # fill additional fields
    # send the reconstructed image and ImageHeader to the next gadget

    print("Sending reconstructed images.")

    dims = data_array.shape

    base_header = ismrmrd.ImageHeader()  # header
    base_header.version = 1
    ndims_image = (dims[0], dims[1], dims[2], dims[3])  # [RO,ENY,SLC,REP]
    print(ndims_image)
    base_header.channels = ndims_image[3]  # number of channels
    base_header.matrix_size = (data_array.shape[0], data_array.shape[1], data_array.shape[2])
    base_header.position = acq.position
    base_header.read_dir = acq.read_dir
    base_header.phase_dir = acq.phase_dir
    base_header.slice_dir = acq.slice_dir
    base_header.patient_table_position = acq.patient_table_position
    base_header.acquisition_time_stamp = acq.acquisition_time_stamp
    base_header.image_index = 0
    base_header.image_series_index = 0
    base_header.data_type = ismrmrd.DATATYPE_CXFLOAT
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    base_header.repetition = acq.idx.repetition
    #  Include free user parameters here!
    base_header.CMx = acq.user_float[0]
    base_header.CMy = acq.user_float[1]
    base_header.CMz = acq.user_float[2]

    R = np.zeros((dims[0], dims[1], dims[2], dims[3]))
    for slc in range(0, dims[2]):
        for rep in range(0, dims[3]):
            R = data_array[:, :, slc, rep]
            base_header.image_type = ismrmrd.IMTYPE_COMPLEX
            base_header.slice = slc
            print(R.shape)
            image_array = ismrmrd.image.Image.from_array(R, headers=base_header)
            connection.send(image_array)
            print("Images successfully sent!")

    # base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    # base_header.slice = acq.idx.slice
    # print(data_array.shape)
    # image_array = ismrmrd.image.Image.from_array(data_array, headers=base_header)
    # connection.send(image_array)
    # print("Images successfully sent!")


def save_reconstructed_image(image_array, path, fname):
    image = nib.Nifti1Image(image_array, np.eye(4))
    nib.save(image, os.path.join(path, fname))

    return image


def EPI_trapezoid_regridding(parameters, kspace):
    # # # # # # # # # # # # # # # # # # # # # # # #
    #      Calculate the gridding operators       #
    # # # # # # # # # # # # # # # # # # # # # # # #

    nky = parameters["N_phase_encode"]
    nkx = parameters["N_phase_recon"]
    nsamp = parameters["readout"]
    tup = parameters["rampUpTime"]
    tdown = parameters["rampDownTime"]
    tdelay = parameters["acqDelayTime"]
    tflat = parameters["flatTopTime"]
    tdwell = parameters["dwellTime"]

    # Temporary trajectory for a symmetric readout
    nK = nsamp
    k = np.zeros(nK)

    # Some timings
    totTime = tup + tflat + tdown
    readTime = tdwell * nsamp

    balanced_ = 1
    # Fix the tdelay for balanced acquisitions
    if balanced_ == 1:
        tdelay = 0.5 * (totTime - readTime)

    # Some areas
    totArea = 0.5 * tup + tflat + 0.5 * tdown
    readArea = 0.5 * tup + tflat + 0.5 * tdown

    if tup > 0.0:
        readArea = readArea - (0.5 * tdelay * tdelay) / tup
    if tdown > 0.0:
        readArea = readArea - 0.5 * (totTime - (tdelay + readTime)) * (totTime - (tdelay + readTime)) / tdown

    # Pre-phase is set so that k=0 is halfway through the readout time
    prePhaseArea = 0.5 * totArea

    # The scale is set so that the readout area corresponds to the number of encoded points
    scale = nky / readArea

    # k = np.zeros(nsamp)
    for i in range(nK):
        t = (i + 1.0) * tdwell + tdelay
        if t <= tup:
            # on the ramp up
            k[i] = 0.5 / tup * t * t
            # print("Condition 1")

        elif tup < t <= (tup + tflat):
            # on the flat top
            k[i] = 0.5 * tup + (t - tup)
            # print("Condition 2")

        else:
            # on the ramp down
            v = tup + tflat + tdown - t
            k[i] = 0.5 * tup + tflat + 0.5 * tdown - 0.5 / tdown * v * v
            # print("Condition 3")

    # Initialise the k-space trajectory arrays
    trajectoryPos_ = np.zeros(nsamp)
    trajectoryNeg_ = np.zeros(nsamp)

    # Fill the positive and negative trajectories
    for i in range(nsamp):
        trajectoryPos_[i] = scale * (k[i] - prePhaseArea)
        trajectoryNeg_[i] = scale * (-1.0 * k[i] + totArea - prePhaseArea)

    # Compute the reconstruction operator
    Km = math.floor(nky / 2.0)
    Ne = 2 * Km + 1

    # Resize the reconstruction operator
    s = (nkx, nsamp)
    Mpos_ = np.array(np.zeros(s), dtype="complex_")
    Mneg_ = np.array(np.zeros(s), dtype="complex_")

    # Evenly spaced k-spaced locations
    keven = np.linspace(-Km, Km, Ne)

    # Image domain locations [-0.5, ..., 0.5]
    x = np.linspace(-0.5, (nkx - 1.) / (2. * nkx), nkx)

    # DFT operator
    # Going from k-space to image space, we use the IFFT sign convention
    s = (nkx, Ne)
    F = np.array(np.zeros(s), dtype="complex_")
    fftscale = 1.0 / np.sqrt(Ne)

    for p in range(nkx):
        for q in range(Ne):
            F[p, q] = fftscale * np.exp(complex(0.0, 1.0 * 2 * np.pi * keven[q] * x[p]))

    # Forward operators
    s = (nsamp, Ne)
    Qp = np.zeros(s)
    Qn = np.zeros(s)

    for p in range(nsamp):
        for q in range(Ne):
            Qp[p, q] = np.sinc(trajectoryPos_[p] - keven[q])
            Qn[p, q] = np.sinc(trajectoryNeg_[p] - keven[q])

    # Reconstruction operators
    s = (nkx, nsamp)
    Mp = Mn = np.zeros(s)
    Mp = np.dot(F, np.linalg.pinv(Qp))
    Mn = np.dot(F, np.linalg.pinv(Qn))

    # Compute the off-centre distance in the RO direction
    roOffCentreDistance = np.dot(parameters["position"], parameters["read_dir"])

    my_keven = np.linspace(0, nsamp - 1, nsamp)

    # Find the offset
    trajectoryPosArma = trajectoryPos_

    n = np.where(trajectoryPosArma == 0)
    for i in n:
        n = int(i) + 1
    my_keven = my_keven - (n - 1)

    # Scale it: we have to find the maximum k-trajectory (absolute) increment
    Delta_k = np.abs(trajectoryPosArma[1:nsamp] - trajectoryPosArma[0:nsamp - 1])
    my_keven = my_keven * np.max(Delta_k[:])

    # Off-centre corrections:
    s = (nsamp, 1)
    myExponent = np.zeros(s)

    for i in range(len(trajectoryPosArma)):
        myExponent[i] = np.imag(2 * np.pi * roOffCentreDistance / parameters["FOV_1"] * trajectoryPosArma[i] -
                                my_keven[i])

    offCentreCorrN = np.exp(myExponent)

    for i in range(len(trajectoryPosArma)):
        myExponent[i] = np.imag(2 * np.pi * roOffCentreDistance / parameters["FOV_1"] * trajectoryNeg_[i] +
                                my_keven[i])

    offCentreCorrP = np.exp(myExponent)

    # Finally, combine the off-centre correction with the reconstruction operator:
    Mp = Mp * np.diag(offCentreCorrP)
    Mn = Mn * np.diag(offCentreCorrN)

    for p in range(parameters["N_phase_recon"]):
        for q in range(nsamp):
            Mpos_[p, q] = Mp[p, q]
            Mneg_[p, q] = Mn[p, q]

    mat_is_reverse = np.array(parameters["is_reversed_acq"])
    mat_is_reverse = np.reshape(mat_is_reverse, (parameters["N_phase_encode"], parameters["N_slices"],
                                                 parameters["N_repetition"]), order='F')

    kspace_corr = np.array(np.zeros((int(kspace.shape[0] / 2), kspace.shape[1], kspace.shape[2], kspace.shape[3],
                                     kspace.shape[4])), dtype="complex_")

    for dim5 in range(kspace.shape[4]):
        for dim2 in range(kspace.shape[1]):
            for dim4 in range(kspace.shape[3]):
                if mat_is_reverse[dim2, dim4, dim5] == 1:
                    kspace_corr[:, dim2, :, dim4, dim5] = (np.dot(Mneg_, np.squeeze(kspace[:, dim2, :, dim4, dim5])))
                else:
                    kspace_corr[:, dim2, :, dim4, dim5] = (np.dot(Mpos_, np.squeeze(kspace[:, dim2, :, dim4, dim5])))

    return kspace_corr


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def epi_2d_recon(connection):
    # The dataset might include navigator measurements. We'll find the acquisitions with navigator flags
    # and never pass them down the chain. They contain no image data, and will not be missed.

    header = connection.header
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    rNx = enc.reconSpace.matrixSize.x
    rNy = enc.reconSpace.matrixSize.y
    rNz = enc.reconSpace.matrixSize.z

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z
    rFOVx = enc.reconSpace.fieldOfView_mm.x
    rFOVy = enc.reconSpace.fieldOfView_mm.y
    rFOVz = enc.reconSpace.fieldOfView_mm.z

    # Number of Slices, Reps, Contrasts, etc.
    ncoils = header.acquisitionSystemInformation.receiverChannels
    print("Number of Coils:", ncoils)

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1
    print("Number of Slices:", nslices)

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1
    print("Number of Repetitions:", nreps)

    if enc.encodingLimits.contrast is not None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1
    print("Number of Contrasts:", ncontrasts)

    # Initialise a storage array
    eNy_ = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
    kspace = np.zeros((eNx, eNy_, ncoils, nslices, nreps), dtype=np.complex64)

    is_reversed_acq = []
    for acquisition in connection:
        if acquisition.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA):
            # print("Found navigator acquisition.")
            navigation_data = acquisition.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)
            continue
        if acquisition.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
            # print("Found parallel calibration acquisition.")
            parallel_calibration = acquisition.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
            continue
        if acquisition.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
            # print("Found phase correction acquisition.")
            phase_corr_data = acquisition.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)

        else:
            # print("Found image acquisition.")

            # Stuff into the buffer
            rep = acquisition.idx.repetition
            # contrast = acq.idx.contrast
            slice = acquisition.idx.slice
            y = acquisition.idx.kspace_encode_step_1
            z = acquisition.idx.kspace_encode_step_2
            kspace[:, y, :, slice, rep] = acquisition.data.T

            if acquisition.isFlagSet(ismrmrd.ACQ_IS_REVERSE):
                rev = acquisition.isFlagSet(ismrmrd.ACQ_IS_REVERSE)
                # print(int(is_reverse_acq))
                is_reversed_acq.append(int(rev))
            else:
                is_reversed_acq.append(int(False))

    print("Removed all navigation acquisitions- proceeding to reconstruction.")

    # Reconstruct images
    # images = np.zeros((eNx, eNy_, ncoils, nslices, nreps), dtype=np.complex64)
    #
    # H = transform.transform_kspace_to_image(kspace, dim=[1])
    # im = np.abs(np.squeeze(np.sum(H, axis=2)))

    # # Show an image
    # for rep in range(nreps):
    #     for slice in range(nslices):
    #         show.imshow(np.squeeze(np.abs(im[:, :, slice, rep])), cmap="gray")

    # # # # # # # # # # # # # # # # # #
    #           REGRIDDING            #
    # # # # # # # # # # # # # # # # # #

    print("The images can be improved by regridding taking into account the ramp time.")

    # tup = tdown = tflat = tdelay = nsamp = nnav = etl = tdwell = 0

    for o in enc.trajectoryDescription.userParameterDouble:
        if o.name == 'dwellTime':
            # tdwell = o.value
            parameters = {o.name: o.value}

    # tup = tdown = tflat = tdelay = nsamp = nnav = etl = 0
    for o in enc.trajectoryDescription.userParameterLong:
        if o.name == 'rampUpTime':
            # tup = o.value
            parameters.update({o.name: o.value})

        if o.name == 'rampDownTime':
            # tdown = o.value
            parameters.update({o.name: o.value})

        if o.name == 'flatTopTime':
            # tflat = o.value
            parameters.update({o.name: o.value})

        if o.name == 'acqDelayTime':
            # tdelay = o.value
            parameters.update({o.name: o.value})

        if o.name == 'numSamples':
            # nsamp = o.value
            parameters.update({o.name: o.value})

        if o.name == 'numberOfNavigators':
            # nnav = o.value
            parameters.update({o.name: o.value})

        if o.name == 'etl':
            # etl = o.value
            parameters.update({o.name: o.value})

    # print(tup, tdown, tflat, tdelay, nsamp, nnav, etl, tdwell)

    parameters.update({"readout": eNx})
    parameters.update({"N_phase_encode": enc.encodingLimits.kspace_encoding_step_1.maximum + 1})
    parameters.update({"N_phase_recon": int(enc.encodedSpace.matrixSize.x / 2)})
    parameters.update({"N_slices": nslices})
    parameters.update({"N_repetition": nreps})
    parameters.update({"read_dir": acquisition.read_dir[:]})
    parameters.update({"position": acquisition.position[:]})
    parameters.update({"FOV_1": eFOVx})
    parameters.update({"is_reversed_acq": is_reversed_acq})

    kspace_corr = EPI_trapezoid_regridding(parameters, kspace)

    # show.imshow(np.squeeze(np.abs(kspace_corr[:, :, 0, 0, 0])))

    # Reconstruct images
    images_corr = np.zeros((int(eNx / 2), eNy_, ncoils, nslices, nreps), dtype=np.complex64)

    H = transform.transform_kspace_to_image(kspace_corr, dim=[1])
    epi_im = np.abs(np.squeeze(np.sum(H, axis=2)))

    send_reconstructed_images(connection, acquisition, epi_im)

    # # Show an image
    # for rep in range(nreps):
    #     for slice in range(nslices):
    #         show.imshow(np.squeeze(np.abs(epi_im[:, :, slc, rep])), cmap="gray")

    show.imshow(np.squeeze(np.abs(epi_im[:, :, 0, 0])), cmap="gray")

    logging.info(f"Python reconstruction done. Duration: {(time.time() - start):.2f} s")

