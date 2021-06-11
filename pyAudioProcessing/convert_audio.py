#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:23:55 2021

@author: jsingh
"""
# Imports

import os
import glob

from pydub import AudioSegment


# Functions

def convert_from_m4a(file_path):
    """
    Converts m4a audio into wav format.
    """
    try:
        track = AudioSegment.from_file(file_path, "m4a")
        file_handle = track.export(
            file_path.replace(".m4a", ".wav"), format='wav'
        )
    except FileNotFoundError:
        print("{} does not appear to be valid. Please check.")
    except Exception as e:
        print(e)


def convert_from_mp3(file_path):
    """
    Converts mp3 audio into wav format.
    """
    try:
        track = AudioSegment.from_file(file_path, "mp3")
        file_handle = track.export(
            file_path.replace(".mp3", ".wav"), format='wav'
        )
    except FileNotFoundError:
        print("{} does not appear to be valid. Please check.")
    except Exception as e:
        print(e)


def convert_from_mp4(file_path):
    """
    Converts mp4 audio into wav format.
    """
    try:
        track = AudioSegment.from_file(file_path, "mp4")
        file_handle = track.export(
            file_path.replace(".mp4", ".wav"), format='wav'
        )
    except FileNotFoundError:
        print("{} does not appear to be valid. Please check.")
    except Exception as e:
        print(e)


def convert_from_aac(file_path):
    """
    Converts aac audio into wav format.
    """
    try:
        track = AudioSegment.from_file(file_path, "aac")
        file_handle = track.export(
            file_path.replace(".aac", ".wav"), format='wav'
        )
    except FileNotFoundError:
        print("{} does not appear to be valid. Please check.")
    except Exception as e:
        print(e)


def convert_files_to_wav(dir_path, audio_format="m4a"):
    """
    Converts all the audio files in the input directory path
    with the extension specified by audio_format input
    into .wav audio files, and saves them in the same directory.
    """
    # Read contents of dir
    # Only select files with the mentioned extension
    files = glob.glob(os.path.join(dir_path, "*." + audio_format))

    # Convert to wav
    # The wav files save in the same dir as specified by dir_path
    if audio_format == "m4a":
        cntr = 0
        for aud_file in files:
            convert_from_m4a(aud_file)
            cntr += 1
        print(
            "{} {} files converted to .wav and saved in {}".format(
                cntr, audio_format, dir_path
            )
        )
    elif audio_format == "mp3":
        cntr = 0
        for aud_file in files:
            convert_from_mp3(aud_file)
            cntr += 1
        print(
            "{} {} files converted to .wav and saved in {}".format(
                cntr, audio_format, dir_path
            )
        )
    elif audio_format == "aac":
        cntr = 0
        for aud_file in files:
            convert_from_aac(aud_file)
            cntr += 1
        print(
            "{} {} files converted to .wav and saved in {}".format(
                cntr, audio_format, dir_path
            )
        )
    elif audio_format == "mp4":
        cntr = 0
        for aud_file in files:
            convert_from_mp4(aud_file)
            cntr += 1
        print(
            "{} {} files converted to .wav and saved in {}".format(
                cntr, audio_format, dir_path
            )
        )
    else:
        print(
            "File format {} is not in supported types (mp3, mp4, m4a, aac)".format(
                audio_format
            )
        )
    
    
