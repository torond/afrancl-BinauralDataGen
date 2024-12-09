import numpy as np
from hanning_window import hanning_window
import scipy.io
from glob import glob
from scipy.io.wavfile import read, write
from scipy import signal
from os.path import basename
import sys
import pdb
import json

"""
Inputs:
- out_path: folder path for output files in .wav format containing spatialized sounds
- in_path: folder path for input files in .wav format containing sounds to be spatialized
- env_paths: list / glob of folder paths for HRIRs -> one path per room

Parameters:
- metadata_dict_mode: boolean flag to append metadata to preexisting json dicts

Input information / parameters:


Input formats:
- sounds in .wav files -> 44.1kHz, variable length
- HRIRs in .wav files?

Output formats:
- spatialized sounds in .wav files -> 44.1kHz, stereo, length of original sound + HRIR length - 1 (I think; it's not cut off)

"""



DEUBUG = False
out_path = "/nobackup/scratch/Wed/francl/logspaced_bandpassed_fluctuating_noise_0.5octv_out_convolved_anechoic_oldHRIRdist140_no_hanning/testset/"
# out_path = "./pure_tones_out/"
rate_out = 44100  # Desired sample rate of "spatialized" WAV
rate_in = 44100  # Sample Rate for Stimuli

# appends metadata to preexisting json dicts
metadata_dict_mode = False
# stim_files = glob("/om/user/francl/SoundLocalization/noise_samples_1octvs_jittered/*.wav")
# stim_paths = "/scratch/Wed/francl/nsynth-valid-upsampled"
stim_paths = "/nobackup/scratch/Wed/francl/bandpassed_fluctuating_noise_0.5octv_fullspec_jittered_phase_logspace_new_2020_05_20/testset"
# stim_paths = "/om/user/gahlm/sorted_sounds_dataset/sorted_stimuli_specfilt"    #File path to stimuli
stim_files = glob(stim_paths + "/audio/*.wav") if metadata_dict_mode else glob(stim_paths + "/*.wav")

# env_paths = sorted(glob("/om/user/francl/Room_Simulator_20181115_Rebuild/HRIRdist140-5deg_elev_az_room*"))
# env_paths = sorted(glob("/om/user/francl/Room_Simulator_20181115_Rebuild_2_mic_version/2MicIRdist140-5deg_elev_az_room*"))
env_paths = sorted(glob(
    "/om/user/francl/Room_Simulator_20181115_Rebuild/Anechoic_HRIRdist140-5deg_elev_az_room5x5y5z_materials26wall26floor26ciel"))
ramp_dur_ms = 10
filter_str = ''

# zero padding options
zero_pad = True
padding_samples = 4000

# jitters ITD if true
vary_itd_flag = False

# stim_files = glob("./pure_tones/*.wav")    #File path to stimuliu
# scales loudness of sounds DO NOT CHANGE
scaling = 0.1

# scales probability of rendering any given position for a sound
# Nsynth
# prob_gen = 0.05
# Use for 1oct white noise
# prob_gen =0.017
# broadband whitenoise
# prob_gen = 0.2
# Natural Stim case
# prob_gen =0.05
# Use for anechoic pure tones or natural stim
prob_gen = 0.25
# Use for natural stim anechoic testset
# prob_gen = 0.125
# whitenoise anechoic
# prob_gen=0.5
# I think this was used in a previous anechoic case
# prob_gen =32.00
version = int(sys.argv[1])

if metadata_dict_mode:
    json_filename = glob(stim_paths + "/*.json")
    assert len(json_filename) == 1, "Only one JSON file supported"
    with open(json_filename[0], 'r') as f:
        json_dict = json.load(f)
json_dict_out = {}

# slice array to parrallelize in slurm
low_idx = 110 * version
high_idx = min(110 * (version + 1), len(stim_files))
stim_files_slice = stim_files[low_idx:high_idx]


def vary_itd(left_stim, right_stim, diff):
    left_roll = np.roll(left_stim, diff)
    return left_roll, right_stim


# For each stimulus file (.wav)
#   For each Room
#       For each HRIR
#           Open HRIR
#           Convolve stimulus with HRIR, or skip if randomly


# Needs to have saved: HRIRs in separate folders for each room
# Path: "hrirtype?_elev_azim_roomgeom_roommaterials/elev_azim_headloc_channelnr.filetype"
# Example: "HRIRdist140-5deg_elev_az_room5x5y5z_materials26wall26floor26ciel/0e_0a_1x2y3z(?)_0.wav"


# Output name (spatialized signal): "/audio/{stim_name}_{elev}elev_{azim}ax_{head_location}_{room_geometry}_{room_materials}.wav"


for s in stim_files_slice:
    # Iterates over all stimuli (455 I think)
    stim_name = basename(s).split(".wav")[0]
    stim_rate, stim = read(s)

    msg = ("The sampling rate {}kHz does not match"
           "the declared value {}kHz".format(stim_rate, rate_in))
    assert stim_rate == rate_in, msg
    if len(stim.shape) > 1:
        print("{} is stereo audio".format(stim_name))
        continue

    # Zeros pad stimulus to avoid sound onset always being at the start of wave
    # files
    hann_stim = hanning_window(stim, ramp_dur_ms, stim_rate)
    stim = hann_stim.astype(np.float32)

    # Changes stim sampling rate to match HRIR
    # nroutsamples = round(len(stim) * rate_out/rate_in)
    # stim_resampled = signal.resample(stim,nroutsamples)
    # gets filesnames for left and right channel HRIR
    for env in env_paths:  # env is one room
        # Iterate through rooms
        hrirs_files = sorted(glob(f'{env}/{filter_str}*.wav'))
        # This should be 72 for 5 degree bins  # What should be 72? o.O
        num_positions = len(hrirs_files) / (36 * 7 * 2)  # calculate nr of listener positions (from file)
        class_balancing_factor = 1 if num_positions < 1 else 4.0 / num_positions  # Hardcoded 4 positions in smallest room
        # This means though that now calculating stuff for *one* room
        env_name_list = basename(env).split("_")  # basename is name of file. "?_?_?_roomgeom_roommaterials"
        room_geometry = env_name_list[3]  # "room5x5y5z" -> Only used as metadata when saving again
        room_materials = env_name_list[4]  # "materials26wall26floor26ciel" -> Only metadata

        # Loop over HRIRs for this room; apparently HRIRs are sorted in pairs of left and right so we can skip every second; lol
        for i, (l, r) in enumerate(zip(hrirs_files[::2], hrirs_files[1::2])):
            if np.random.random() > prob_gen * class_balancing_factor:  # Randomly skip positions
                continue
            name_list = basename(r).split("_")  # "elev_azim_headloc_channelnr.filetype" "0e_0a_headloc_0.wav"
            elev = name_list[0].split("e")[0]
            azim = name_list[1].split("a")[0]
            head_location = name_list[2]
            channel = name_list[3].split(".")[0]
            # Reads in HRIRs
            # Expensive I/O is done maximally 2*71064*nr_sounds times
            # But! In the end only for the selected locations
            # Still, ca. 1000 times per location
            # I can just keep it in memory I guess, it's "only" 45GB...
            # Otherwise, base nested loop on HRIRs and not on stimuli:
            # For each training_coordinate in training_coordinates:
            #   For each stimulus in stimuli:
            #     Apply HRIR to stimulus w/ prob, otherwise skip
            # -> How to integrate background gen?
            # Idea: Two passes
            # 1. Pass: Generate coordinates for stimuli and background noises
            # 2. Pass: Load HRIRs and apply to all sounds specified in the first pass
            # Pre-generate a list of background coordinate sets (each has 3-8 positions)
            hrir_r = read(r)[1].astype(np.float32)
            hrir_l = read(l)[1].astype(np.float32)
            # "spatializes" the sound, float64 return value. VERY loud. Do not play.
            conv_stim_r = signal.fftconvolve(stim, hrir_r)
            conv_stim_l = signal.fftconvolve(stim, hrir_l)
            if vary_itd_flag:
                if not zero_pad:
                    raise NotImplementedError("Vary ITD only supported with zero padding")
                diff = np.random.randint(-25, 25)
                conv_stim_l, conv_stim_r = vary_itd(conv_stim_l, conv_stim_r, diff)
            # Testing code
            if DEUBUG:
                name = "{}_{}elev_{}azim_convolved.mat".format(stim_name, elev, azim)
                name_with_path = out_path + name
                scipy.io.savemat(name_with_path, mdict={'arr': conv_stim_r})
            # Rescale to not blow out headphones/eardrums
            max_val = max(np.max(conv_stim_r), np.max(conv_stim_l))
            rescaled_conv_stim_r = conv_stim_r / max_val * scaling
            rescaled_conv_stim_l = conv_stim_l / max_val * scaling
            # converts to proper format for scipy wavwriter
            out_stim = np.array([rescaled_conv_stim_l, rescaled_conv_stim_r], dtype=np.float32).T
            if metadata_dict_mode:
                spatial_dict = {'elev': int(elev),
                                'azim': int(azim),
                                'head_location': head_location,
                                'room_geometry': room_geometry,
                                'room_materials': room_materials}
                name = f"/audio/{stim_name}_{elev}elev_{azim}ax_{head_location}_{room_geometry}_{room_materials}.wav"
                json_dict_out[name] = {**json_dict[stim_name], **spatial_dict}
            else:
                name = "/{}_{}elev_{}ax_{}_{}_{}.wav".format(stim_name, elev, azim, head_location, room_geometry,
                                                             room_materials)
            if not i % 1000:
                print(name)
            name_with_path = out_path + name
            write(name_with_path, rate_out, out_stim)

if metadata_dict_mode:
    json_filename = out_path + "/examples{}.json".format(version)
    with open(json_filename, 'w') as f:
        json.dump(json_dict_out, f)
