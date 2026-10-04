"""
Step 0 -- Ground-truth direction mapping (shared dependency for direction
classifier calibration/validation, Step 2, and results reporting, Step 6).

Sourced from docs/Fall_Prevention_Actuator_Input_Pipeline.md, Part 3, Table 2
(Fall Type Reference). Single source of truth so this mapping isn't re-derived
ad hoc in multiple places.
"""

TASK_TO_TRUE_DIRECTION = {
    20: "forward",   # F01 -- forward fall trying to sit down
    21: "backward",  # F02 -- backward fall trying to sit down
    22: "lateral",   # F03 -- lateral fall trying to sit down
    23: "forward",   # F04 -- forward fall trying to get up
    24: "lateral",   # F05 -- lateral fall trying to get up
    25: "forward",   # F06 -- forward fall while sitting, fainting
    26: "lateral",   # F07 -- lateral fall while sitting, fainting
    27: "backward",  # F08 -- backward fall while sitting, fainting
    28: "forward",   # F09 -- vertical (forward) fall while walking, fainting
    29: "forward",   # F10 -- fall while walking, hands used to dampen, fainting
    30: "forward",   # F11 -- forward fall while walking, trip
    31: "forward",   # F12 -- forward fall while jogging, trip
    32: "forward",   # F13 -- forward fall while walking, slip
    33: "lateral",   # F14 -- lateral fall while walking, slip
    34: "backward",  # F15 -- backward fall while walking, slip
}

TASK_TO_FALL_ID = {
    20: "F01", 21: "F02", 22: "F03", 23: "F04", 24: "F05",
    25: "F06", 26: "F07", 27: "F08", 28: "F09", 29: "F10",
    30: "F11", 31: "F12", 32: "F13", 33: "F14", 34: "F15",
}
