import numpy as np
import scipy.io
from glob import glob
from scipy.io.wavfile import read, write
from scipy import signal
from os.path import basename
import nnresample
import sys
import pdb
import json
from collections import defaultdict

"""
Input formats:
- sounds in .wav files -> 44.1kHz, variable length

Output formats:
- spatialized sounds in .wav files -> 44.1kHz, stereo, length of original sound + HRIR length - 1 (I think; it's not cut off)

"""



DEBUG = False
out_path = "/nobackup/scratch/Sat/francl/bkgd_textures_out_sparse_sampled_same_texture_expanded_set_44.1kHz_anechoic/"
env_paths = sorted(glob(
    "/om/user/francl/Room_Simulator_20181115_Rebuild/Anechoic_HRIRdist140-5deg_elev_az_room5x5y5z_materials26wall26floor26ciel"))
# -> Glob containing 5 paths, one for each room configuration / environment
# But the path above doesn't make sense then bc it already specifies one room (room5x5y5z, materials26wall26floor26ciel)
# Earlier commit: env_paths = sorted(glob("/om/user/francl/Room_Simulator_20181115_Rebuild/HRIRdist140-5deg_elev_az_*"))
anechoic_HRTF = True if any(["Anechoic" in path for path in env_paths]) else False
# They're using anechoic HRTFs here! Room size and listener position doesn't matter then, right?
# Env folder contains 504 HRIRs then.
# But the noise should also be rendered at the same position as the signal, otherwise it isn't realisitic.
version = int(sys.argv[1])  # 0-4, see slurm script. This is run once for each room configuration
hrir_path = env_paths[version]
hrir_name = basename(hrir_path)
walls = hrir_name.split("_")[4] if anechoic_HRTF else hrir_name.split("_")[3]
materials = hrir_name.split("_")[5] if anechoic_HRTF else hrir_name.split("_")[4]
# They only get wall size and materials once, i.e., for this room

# out_path = "./pure_tones_out/"
rate_out = 44100  # Desired sample rate of "spatialized" WAV
rate_in = 44100  # Sample Rate for Stimuli

background_file_paths = glob(
    "/nobackup/scratch/Sat/francl/STSstep_v0p2/_synths/*/*.wav")  # File path to background sounds
# hrirs_files = sorted(glob("./JayDesloge/HRIR_elev_az_44100Hz_vary_loc/*{}x{}y{}z*.wav".format(x,y,z)))
MAXVAL = 32767.0
scaling = 0.1
NUM_BKGDS = 50000  # Number of backgrounds to generate; has nothing to do with there being 50000 texture samples...
NUM_POSITIONS = 504

array_write = False

# Resample stims if necessary
# Store in dictionary: key = stim_key (?), value = list of tuples (stim_resampled, file_path)
# -> Dict has 50 keys (sound classes), each key has 1000 values (synthesized textures)
resampled_stim_dict = defaultdict(list)
for i, bg_file_path in enumerate(background_file_paths):
    stim_rate, stim = read(bg_file_path)

    if stim.dtype != np.float32:
        raise ValueError("Stim not float32 format! Normalization incorrect.")
    stim_name = basename(bg_file_path).split(".")[0]
    stim_key = "".join(stim_name.split('_')[:3])
    # Changes stim sampling rate to match HRIR
    # nroutsamples = round(len(stim) * rate_out/rate_in)
    # stim_resampled = signal.resample(stim,
    #                                 nroutsamples).astype(np.float32,casting='unsafe',)
    if rate_out != stim_rate:
        stim_resampled = nnresample.resample(stim, rate_out, stim_rate).astype(np.float32)
        pdb.set_trace()
    else:
        stim_resampled = stim
    resampled_stim_dict[stim_key].append((stim_resampled, background_file_paths[i]))
    if not i % 100:  # Every 100 files print:
        print(f"Resampling...{i} complete")

hrirs_files = sorted(glob(f"{hrir_path}/*.wav"))

# Get relative spherical coords for naming purposes later, order is important...
# Assuming hrirs_files has 504 elements, locs contains the spherical coordinates of the HRIRs, e.g. ['0e0a', '0e5a', ...]
# hrirs_files has at least 1008 elements, as it contains left and right channel HRIRs one after the other
locs = list(set([filename.split("_")[-2] for filename in hrirs_files]))



resampled_stim_array_keys = list(resampled_stim_dict.keys())
nr_src_classes = len(resampled_stim_dict)
# nr_srcs_per_class = {src_class: len(textures) for src_class, textures in resampled_stim_dict.items()}
noise_index_array = []
# assuming a left and right channel for every source
for _ in range(NUM_BKGDS):
    # Iterate 50000 times
    num_sampled_pos = np.random.randint(3, 8)
    # -> random number between 3 and 8  (e.g. 5)
    pos_idx = np.random.choice(NUM_POSITIONS, num_sampled_pos, replace=False)
    # -> transform to 3-8 random position indices between 0 and 503 (e.g. [5, 37, 102, 200, 499])

    source_class_key = np.random.randint(0, nr_src_classes)
    # - get random index for source class (0 - nr_src_classes) (e.g. 42)
    num_sources_in_class = len(resampled_stim_dict[resampled_stim_array_keys[source_class_key]])
    # -> determine amount of sources in this class (e.g. 1000)
    source_idxs = np.random.choice(num_sources_in_class, NUM_POSITIONS, replace=False)
    # -> sample 504 random indices from 0 to 999
    selected_vals = np.full(pos_idx.shape + (2,), source_class_key)
    # -> create array of shape (5, 2) filled with 42
    selected_vals[:, 1] = source_idxs[pos_idx]
    # -> replace second column with 3-8 random indices from 0 to 999
    # -> e.g. [[42, 5], [42, 37], [42, 102], [42, 200], [42, 499]]
    #

    mask = np.full(source_idxs.shape + (2,), np.nan)
    # -> create array of shape (504, 2) filled with NaN
    mask[pos_idx] = selected_vals
    # -> replace 3-8 random indices with the values from selected_vals
    # These contain the source class and the index of the source in the class, and indirectly which position to use
    noise_index_array.append(mask)
    # -> append the array to the list
    # -> list contains 50000 arrays of shape (504, 2) sparsely filled with source class and source index information
    # Information contained: Which synthesized background textures to use at what position (0-503), i.e. with which HRIR to convolve

# gets filesnames for left and right channel HRIR
metadata_dict = {}
for i, noise_indices in enumerate(noise_index_array):
    # Iterate 50000 times
    total_waveform = []
    total_waveform_labels = []
    hrirs_files = sorted(glob(f"{hrir_path}/*{locs[i % len(locs)]}*.wav"))
    #
    for l, r, (src_class_idx, texture_idx) in zip(hrirs_files[::2], hrirs_files[1::2], noise_indices):
        # Looks like they iterate 504 times, as they use anechoic HRTFs which don't depend on room size and listener position
        # This matches up with the 504 positions in each of the lists the noise_index_array

        # index[0] is the source class index, index[1] is index of texture in the class

        if np.isnan(src_class_idx):
            continue  # -> if mask value is NaN, skip this hrir position
        else:  # -> if it has a value, get the source class key and the noise index
            source_class_key = resampled_stim_array_keys[int(src_class_idx)]
            # -> get key of the source class to retrieve list of textures
            noise_index = int(texture_idx)
        name_list = basename(r).split("_")  # basename(r) is the HRIR file name, then split by "_"
        elev = name_list[0].split("e")[0]  # "0e_0a_0x0y0z_
        azim = name_list[1].split("a")[0]
        channel = name_list[3].split(".")[0]
        # Reads in HRIRs
        hrir_r = read(r)[1].astype(np.float32)
        hrir_l = read(l)[1].astype(np.float32)
        # Grab correct noise sample
        noise, stim_name = resampled_stim_dict[source_class_key][noise_index]
        # "spatializes" the sound, float64 return value. VERY loud. Do not play.
        conv_stim_r = signal.convolve(noise, hrir_r).astype(np.float32)
        conv_stim_l = signal.convolve(noise, hrir_l).astype(np.float32)
        # Testing code
        if DEBUG:
            name = f"{stim_name}_{elev}elev_{azim}azim_convolved.mat"
            name_with_path = out_path + name
            scipy.io.savemat(name_with_path, mdict={'arr': conv_stim_r})
        total_waveform.append([conv_stim_l, conv_stim_r])
        total_waveform_labels.append([azim, elev, stim_name])
    summed_waveform = np.sum(np.array(total_waveform), axis=0)
    # -> Sum the 3-8 spatialized waveforms
    ###This is where you stopped### Need to test array summation and listen to waveforms
    # Rescale to not blow out headphones/eardrums
    max_val = np.max(summed_waveform)
    rescaled_summed_waveform = summed_waveform / max_val * scaling
    if array_write:
        name = f"noise_{i}_spatialized_{rate_out}sr.npy"
        name_with_path = out_path + name
        print(name)
        np.save(name_with_path, resampled_stim_array)
    else:
        # converts to proper format for scipy wavwriter
        name = f"noise_{i}_spatialized_{locs[i % len(locs)]}_{walls}_{materials}.wav"
        metadata_dict[name] = total_waveform_labels
        name_with_path = out_path + name
        out_stim = np.array(rescaled_summed_waveform, dtype=np.float32).T
        print(name)
        write(name_with_path, rate_out, out_stim)

    # Indented one, otherwise it doesn't make sense...
    # Saves azim, elev, and texture name for each sound in the spatialized background
    json_name = out_path + f"label_meatadata_{locs[i % len(locs)]}_{walls}_{materials}.json"
    with open(json_name, 'w') as fp:
        json.dump(metadata_dict, fp)
