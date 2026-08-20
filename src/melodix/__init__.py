"""Melodix: optical music recognition for drum notation.

Converts drum sheet music into multi-track MIDI, synthesized audio, and
interactive sync metadata. The pipeline runs in four stages, each in its own
subpackage: :mod:`melodix.geometry` measures the page, :mod:`melodix.vision`
recognises the symbols on it, reconstruction turns those into MIDI, and
synthesis renders audio.
"""
