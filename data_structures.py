import os
import numpy as np
from dataclasses import dataclass, make_dataclass
from scipy.signal import resample, envelope
from typing import Literal
from util import *
from scipy.signal import butter, filtfilt, find_peaks, peak_widths
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import pandas as pd

# gait parameters to be calculated later
GAIT_PARAMETERS = [
    "stride_length",
    "stride_time",
    "stride_velocity",
    "base_of_support",
    "double_support_time",
    "swing_time",
]

# create dataclass for storing gait parameters
parameters = [(parameter_name, float, np.nan) for parameter_name in GAIT_PARAMETERS]
Parameters = make_dataclass(cls_name="Parameters", fields=parameters)

gait_analysis_properties = {
    "stereo": {
        "fps": 100,
        "max_gap_interpolation": 0.25,
        "filter_cutoff": 2,
        "filter_order": 4,
        "heel_thr": 0.8,
        "toe_thr": 0.8,
        "stride_time_min": 0.65,
        "stride_time_max": 2.5,
        "swing_time_min": 0.3,
        "stance_time_min": 0.3,
    },
    "qualisys": {
        "fps": 100,
        "max_gap_interpolation": 0.25,
        "filter_cutoff": 7,
        "filter_order": 4,
        "heel_thr": 0.8,
        "toe_thr": 0.8,
        "stride_time_min": 0.65,
        "stride_time_max": 2.5,
        "swing_time_min": 0.3,
        "stance_time_min": 0.3,
    },
}

stereo_properties = gait_analysis_properties["stereo"]
qualisys_properties = gait_analysis_properties["qualisys"]

CREATE_DEBUG_FIGS = False


@dataclass
class GaitEvent:
    """
    Dataclass to store data related to a single Gait Event.

    Attributes:
        type: Initial Contact (IC) or Final Contact (FC)
        frame: point of time (frame) at which the event occured
        side: left / right side at which the event occured
        perspective: perspective (front/back) of segment to which the event belong to
        segment: indicates whether the segment contains straight walking or turning action
        segment_index: index of segment (starting from 0 at the beginning of recording). Incremented with perspective change
        relative_position: position of the event within the segment, relative to the segment's end (1=segment start in time, 0=segment end)
    """

    def __init__(
        self,
        type: Literal["IC", "FC"] = np.nan,
        frame: int = np.nan,
        side: Literal["left", "right"] = np.nan,
        perspective: Literal["front", "back"] = np.nan,
        segment: Literal["straight", "turn"] = np.nan,
        segment_index: int = np.nan,
        relative_position: float = np.nan,
    ):
        self.type = type
        self.frame = int(frame)
        self.side = side
        self.perspective = perspective
        self.segment = segment
        self.segment_index = segment_index
        self.relative_position = relative_position

    def __repr__(self):
        return f"{self.type:<10}{self.side:<10}{self.frame:<10}{self.perspective:<10}{self.segment:<10}{self.segment_index:<10}{self.relative_position:<10}"

    def __str__(self):
        return f"{self.type:<10}{self.side:<10}{self.frame:<10}{self.perspective:<10}{self.segment:<10}{self.segment_index:<10}{self.relative_position:<10}"


@dataclass
class Flags:
    """
    Dataclass to store the result of various checks performed on GaitCycle objects.

    Attributes:
        complete_cycle: all gait events are available
        types_ok: all gait events are available & have the correct type (IC/FC)
        sides_ok: all gait events are available & have the correct sides (right/left)
        perspectives_ok: at least IC0, IC2 are available & have the same perspective (front/back)
        segments_ok: at least IC0, IC2 are available & have the same segment type (straight, turn) and same segment_index
        valid: GC has not been deemed useless, there are no critical mistakes with the GEs
    """

    def __init__(
        self,
    ):
        self.complete_cycle = True
        self.types_ok = True
        self.sides_ok = True
        self.perspectives_ok = True
        self.segments_ok = True
        self.valid = True

    def __str__(self):
        string = f"valid: {self.valid:<5}persp: {self.perspectives_ok:<5}segments: {self.segments_ok:<5}complete cycle: {self.complete_cycle:<5}types: {self.types_ok:<5}sides: {self.sides_ok:<5}"
        return string

    def __repr__(self):
        string = f"valid: {self.valid:<5}persp: {self.perspectives_ok:<5}segments: {self.segments_ok:<5}complete cycle: {self.complete_cycle:<5}types: {self.types_ok:<5}sides: {self.sides_ok:<5}"
        return string


class GaitCycle:
    """
    Class to representing a Gait Cycle. Calculates, checks and stores the propeprties of a GC.

    Attributes:
        possible_gait_events: GEs which might belong to this GC. They will be evaluated and if fit, the final GEs will be picked from this list.
        keypoints: 3D keypoint (preprocessed) data in (num_kpts, dims, frames) shape.
        labels: labels describing each keypoint
        fps: recording frequency in Hz.
        IC0: 1st IC type ipsilateral GE, regarded as the start of GC
        IC2: last IC type ipsilateral GE, regarded as end of GC
        IC1, FC0, FC1: rest of the GEs of the segment
        ipsi: ipsilateral side (left/right)
        contra: contralateral side
        perspective: perspective of the segment to which the GEs belong
        segment: whether the segment is straight/turn type
        segment_index: index of segment, starting from 0 at the beginning of recording. Changes with perspective
        relative_position: relative position of the start of GC compared to the beginning of current segment
        gait_events: the GEs that have been selected and verified to be part of this GC
        flags: Flags type container, storing the results of GE checks. Indicates the overall quality of GC
        parameters: Parameters type container, storing the results of calculated gait-cycle parameters

    """

    def __init__(self, input_gait_events: list, keypoint_data: np.ndarray, labels: list, fps):

        self.possible_gait_events = sorted(input_gait_events, key=lambda event: event.frame)
        self.keypoints = keypoint_data
        self.labels = labels
        self.fps = fps

        # IC0, IC2 are determined at the Recording level (checks are also performed there)
        self.IC0 = self.possible_gait_events[0] if len(self.possible_gait_events) >= 2 else None
        self.IC2 = self.possible_gait_events[-1] if len(self.possible_gait_events) >= 2 else None

        self.ipsi = self.IC0.side
        self.contra = "left" if self.ipsi == "right" else "right"
        self.perspective = self.IC0.perspective
        self.segment = self.IC0.segment
        self.segment_index = self.IC0.segment_index
        self.relative_position = self.IC0.relative_position

        # GE priorities (based on parameters): IC0-IC2, IC1, FC0-FC1 -> select IC1 first, then FC0, FC1
        self.IC1 = self.select_gait_event(type="IC", side=self.contra)
        self.FC0 = self.select_gait_event(
            type="FC", side=self.contra, prev_event=self.IC0, next_event=self.IC1
        )
        self.FC1 = self.select_gait_event(
            type="FC", side=self.ipsi, prev_event=self.IC1, next_event=self.IC2
        )

        # get rid of not found / non-existent GEs
        self.gait_events = [ge for ge in [self.IC0, self.FC0, self.IC1, self.FC1, self.IC2] if ge]

        self.flags = Flags()
        self.set_flags()

        self.parameters = Parameters()
        if None not in [self.IC0, self.IC2]:
            self.calculate_stride_parameters()

            if self.flags.valid:
                if None not in [self.IC1]:
                    self.calculate_base_of_support()
                    if None not in [self.FC0, self.FC1]:
                        self.calculate_double_support_time()
                        self.calculate_swing_time()

    def __str__(self):
        return f"{self.perspective:<10}{self.ipsi:<10}{self.flags.valid:<10}{[ge.frame for ge in self.gait_events]}"

    def __repr__(self):
        return f"{self.perspective:<10}{self.ipsi:<10}{self.contra:<10}{self.flags.valid:<10}{[ge.frame for ge in self.gait_events]}"

    def calculate_stride_parameters(self) -> None:  # IC0, IC2
        """
        Computes stride_time, stride_lenght and stride_velocity.
            Gait events needed: IC0, IC2

        Sets:
            parameters.stride_time
            parameters.stride_length
            parameters.stride_velocity
        """
        self.parameters.stride_time = (self.IC2.frame - self.IC0.frame) / self.fps

        # IC0_heel = self.keypoints[self.labels.index(f"{self.ipsi}_heel"), :, self.IC0.frame]
        # IC2_heel = self.keypoints[self.labels.index(f"{self.contra}_heel"), :, self.IC2.frame]
        IC0_ankle = self.keypoints[self.labels.index(f"{self.ipsi}_ankle"), :, self.IC0.frame]
        IC2_ankle = self.keypoints[self.labels.index(f"{self.ipsi}_ankle"), :, self.IC2.frame]
        self.parameters.stride_length = np.linalg.norm(IC2_ankle - IC0_ankle)

        self.parameters.stride_velocity = (
            self.parameters.stride_length / self.parameters.stride_time
            if self.parameters.stride_time >= 0
            else np.nan
        )

    def calculate_swing_time(self) -> None:  # FC1, IC2
        """
        Computes swing_time.
            Gait events needed: FC1, IC2

        Sets:
            parameters.swing_time
        """
        if self.fps > 0:
            self.parameters.swing_time = (self.IC2.frame - self.FC1.frame) / self.fps

    def calculate_double_support_time(self) -> None:  # FC0, FC1, IC0, IC1
        """
        Computes double_support_time.
            Gait events needed: FC0, FC1, IC0, IC1

        Sets:
            parameters.double_support_time
        """
        if self.fps > 0:
            self.parameters.double_support_time = (
                (self.FC0.frame - self.IC0.frame) + (self.FC1.frame - self.IC1.frame)
            ) / self.fps

    def calculate_base_of_support(self) -> None:  # IC0, IC1, IC2
        """
        Computes base_of_support.
            Gait events needed: IC0, IC1, IC2

        Sets:
            parameters.base_of_support
        """

        # ankles
        IC1_ankle = self.keypoints[self.labels.index(f"{self.contra}_ankle"), :, self.IC1.frame]
        IC0_ankle = self.keypoints[self.labels.index(f"{self.ipsi}_ankle"), :, self.IC0.frame]
        IC2_ankle = self.keypoints[self.labels.index(f"{self.ipsi}_ankle"), :, self.IC2.frame]

        line_of_progression = IC2_ankle - IC0_ankle
        IC0_IC1_line = IC0_ankle - IC1_ankle

        # perpendicular vector to both, magnitude proportional to area of parallelorgram
        parallelogram_area = np.linalg.norm(np.cross(line_of_progression, IC0_IC1_line))

        # contra foot distance from line of progression (IC2_IC0_line)
        distance_from_IC2_IC0_line = parallelogram_area / np.linalg.norm(line_of_progression)
        self.parameters.base_of_support = distance_from_IC2_IC0_line

    def set_flags(self) -> None:
        """
        Checks the perspectives (front/back), segments (straight/turn), types (IC/FC), sides (left/right) of gait cycle.

        Sets:
            flags.perspectives_ok
            flags.segments_ok
            flags.types_ok
            flags.sides_ok
            flags.complete_cycle
            flags.valid
        """

        # check if perspectives are all the same (front/back)
        self.flags.perspectives_ok = False not in [
            ge.perspective == self.perspective for ge in self.gait_events
        ]
        # check if segments are all the same (straight/turn)
        self.flags.segments_ok = False not in [
            ge.segment == self.segment and ge.segment_index == self.segment_index
            for ge in self.gait_events
        ]

        self.flags.types_ok = False not in [
            self.IC0.type == "IC" if self.IC0 else True,
            self.FC0.type == "FC" if self.FC0 else True,
            self.IC1.type == "IC" if self.IC1 else True,
            self.FC1.type == "FC" if self.FC1 else True,
            self.IC2.type == "IC" if self.IC2 else True,
        ]
        self.flags.sides_ok = False not in [
            self.IC0.side == self.ipsi if self.IC0 else True,
            self.FC0.side == self.contra if self.FC0 else True,
            self.IC1.side == self.contra if self.IC1 else True,
            self.FC1.side == self.ipsi if self.FC1 else True,
            self.IC2.side == self.ipsi if self.IC2 else True,
        ]

        # if we have all gait events
        if None not in [self.IC0, self.IC1, self.IC2, self.FC0, self.FC1]:
            # get rid of the rest (extra events)
            self.gait_events = sorted(
                [self.IC0, self.IC1, self.IC2, self.FC0, self.FC1], key=lambda event: event.frame
            )
        else:
            self.flags.complete_cycle = False

        if (
            not self.flags.types_ok
            or not self.flags.sides_ok
            or not self.flags.perspectives_ok
            or not self.flags.segments_ok
        ):
            self.flags.valid = False

    def select_gait_event(
        self,
        type: Literal["IC", "FC"],
        side: Literal["left", "right"],
        prev_event: GaitEvent = None,
        next_event: GaitEvent = None,
    ) -> GaitEvent | None:
        """
        Select the appropriate gait event from the possible_gait_events belonging to this gait cycle.

        Args:
            type: type of gait event
            side: side of gait event
            prev_event: previoius GE event (based on frame). Selected GE can only come after this
            next_event: next GE event. Selected one must come before this.

        Returns:
            selected_gait_event: the first event (ranked by frame) that satisfies the criteria, otherwise None
        """

        frame_min = prev_event.frame if prev_event else self.IC0.frame
        frame_max = next_event.frame if next_event else self.IC2.frame

        specified_gait_events = [
            ge
            for ge in self.possible_gait_events
            if (
                ge.type == type
                and ge.side == side
                and ge.perspective == self.perspective
                and ge.segment == self.segment
                and frame_min <= ge.frame <= frame_max
            )
        ]
        selected_gait_event = next(iter(specified_gait_events), None)
        return selected_gait_event


class Recording:
    """
    Class representing the functionality and data of a single recording system (stereo/qualisys).
    Ascertains segment types, indices and perspectives. Performs GE detection and arranges GEs into GCs.

    Attributes:
        keypoints: 3D (prefiltered) keypoint data with shape (num_kpts, dims, frames)
        labels: labels describing each keypoint
        recording_type: type of the system used to obtain the keypoint data (stereo/qualisys
        gait_analysis_properties: properties used for GE detection and preprocessing
        fps: sampling frequency in Hz
        heel_thr: velocity threshold below which the heel keypoint is considered stationary
        toe_thr: velocity threshold below which the big_toe keypoint is considered stationary
        max_stride_time: maximum time below which a stride sequence is considered valid
        min_swing_time: minimal time above which a swing phase is considered valid
        min_stance_time: minaimal time above which a stance phase is considered valid
        debug_folder: folder that should containe the figures and graphs used mainly for debugging
        turn_mask: mask indicating whether a given point in time belongs to a straight or turn segment
        perspective: mask indicating whether the given point in time belongs to a front of back perspective
        segment_indices: mask indicating the index of segments
        segment_boundaries: dictionary linking segment indices to the start and end points of each segment
        initial_gait_events: GEs detected by the first pass of a 2-stage detection algorithm
        initial_gait_cycles: GCs constructed from the initial_gait_events
        velocity_estimate: stride velocity estimate caluclated using initial_gait_cycles
        gait_events: final GEs detected by 2nd stage of the 2-stage algorithm
        all_gait_cycles: all the GCs detected (valid and invalid)
        gait_cycles: only the valid GCs (the ones which passed all the checks)
    """

    def __init__(
        self,
        keypoints: np.ndarray,
        labels,
        recording_type: Literal["stereo", "qualisys"],
        gait_analysis_properties: dict,
        debug_folder: str = None,
    ):
        self.keypoints = keypoints
        self.labels = labels
        self.recording_type = recording_type

        # gait analysis properties
        gait_analysis_properties = gait_analysis_properties[self.recording_type]
        self.fps = gait_analysis_properties["fps"]

        self.heel_thr = gait_analysis_properties["heel_thr"]
        self.toe_thr = gait_analysis_properties["toe_thr"]

        self.max_stride_time = gait_analysis_properties["stride_time_max"] * self.fps
        self.min_swing_time = gait_analysis_properties["swing_time_min"] * self.fps
        self.min_stance_time = gait_analysis_properties["stance_time_min"] * self.fps

        if not os.path.exists(debug_folder) and CREATE_DEBUG_FIGS:
            os.mkdir(debug_folder)
        self.debug_folder = debug_folder

        self.turn_mask, self.perspective, self.segment_indices, self.segment_boundaries = (
            self.get_turns_and_perspective_and_segments(relative_height=0.7)
        )

        # if the global-frame coordinate axes are flipped, flop them back
        if self.perspective[0] != "front":
            self.keypoints[:, 1, :] = self.keypoints[:, 1, :] * (-1)
            self.turn_mask, self.perspective, self.segment_indices, self.segment_boundaries = (
                self.get_turns_and_perspective_and_segments(relative_height=0.7)
            )

        self.initial_gait_events = self.get_gait_events(velocity=1, create_debug_fig=False)
        self.initial_gait_cycles = self.get_gait_cycles(self.initial_gait_events)
        self.velocity_estimate = np.nanmean(
            [gc.parameters.stride_velocity for gc in self.initial_gait_cycles if gc.flags.valid]
        )
        self.gait_events = self.get_gait_events(
            velocity=self.velocity_estimate, create_debug_fig=CREATE_DEBUG_FIGS
        )
        # make distinction between valid and all-gait cycles
        self.all_gait_cycles = self.get_gait_cycles(self.gait_events)
        self.gait_cycles = [gc for gc in self.all_gait_cycles if gc.flags.valid]
        if self.debug_folder:
            self.create_gait_cycles_figure()

    def get_gait_cycles(self, gait_events: list):
        gait_cycles = []
        for event in gait_events:
            if event.type == "IC":
                next_ipsi_IC = next(
                    iter(
                        [
                            ge
                            for ge in gait_events
                            if (
                                ge.type == "IC"
                                and ge.side == event.side
                                and ge.frame > event.frame
                                and ge.frame <= event.frame + self.max_stride_time
                                and ge.perspective == event.perspective
                                and ge.segment == event.segment
                            )
                        ]
                    ),
                    None,
                )
                if next_ipsi_IC:
                    possible_events = [
                        ge
                        for ge in gait_events
                        if (ge.frame >= event.frame and ge.frame <= next_ipsi_IC.frame)
                    ]
                    current_gait_cycle = GaitCycle(
                        possible_events, self.keypoints, self.labels, self.fps
                    )
                    gait_cycles.append(current_gait_cycle)

        return gait_cycles

    def get_turns_and_perspective_and_segments(
        self,
        min_peak_height: float = 0.25,
        relative_height: float = 0.5,
        min_walk_duration: float = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Identifies turning segments (straight/turn) and perspective (front / back) based on shoulder coordinates using peak detection method.
        Also assigns numeric indices to segments (0,1,2...) and records segment boundaries based on perspective.

        Args:
            min_peak_height:    only peaks above this height will be considered
            relative_height:    height at which the bases of a peak will be taken. It is MEASURED FROM APEX
            min_walk_duration:  min elapsed time (in seconds) between turns (1 turn = 180 deg)

        Returns:
            turn_mask: boolean array indicating straight turning segments (1) and straight segments (0).

            perspective: bollean array indicating frontal facing (1) and back facing (0) segments.

            segment_indices: array containing the segment index for every time point.

            segment_boundaries: dict linking segent indices to segment bondaries. (idx 1-> 0-1500)
        """

        # min distance from peak to peak (2 seconds of straight movement) in samples
        min_peak_distance = min_walk_duration * self.fps

        # get the y coord of shoulders (for qualisys x-y plane is horizontal, y coord was forwards/backwards movement)
        right_shoulder_position = self.keypoints[self.labels.index("right_shoulder"), 1, :]
        left_shoulder_position = self.keypoints[self.labels.index("left_shoulder"), 1, :]

        # get rate of change in positional difference
        shoulder_diff_change = np.abs(
            np.diff(
                right_shoulder_position - left_shoulder_position,
                prepend=(right_shoulder_position - left_shoulder_position)[0],
            )
        )
        # Interpolate NaNs in shoulder_diff
        if np.any(np.isnan(shoulder_diff_change)):
            nans = np.isnan(shoulder_diff_change)
            shoulder_diff_change[nans] = np.interp(
                np.flatnonzero(nans), np.flatnonzero(~nans), shoulder_diff_change[~nans]
            )
        # normalize
        shoulder_diff_change = shoulder_diff_change / np.nanmax(shoulder_diff_change)

        numeric_perspective = np.where(left_shoulder_position > right_shoulder_position, 1, 0)
        perspective_change = np.diff(a=numeric_perspective, prepend=[numeric_perspective[0]])

        # search for peaks in shoulder difference change by looking at position crossover of shoulders
        init_peak_indices = np.flatnonzero(perspective_change)
        init_peaks = shoulder_diff_change[init_peak_indices]

        # assume most peaks are detected normally, pick average representation as median
        median_peak_height = np.median(init_peaks)

        # if there is diff larger than 20% between median and any other peak, consider it outlier
        if True in [abs(peak - median_peak_height) > 0.7 for peak in init_peaks]:
            # clip signal with median
            shoulder_diff_change = shoulder_diff_change.clip(0, median_peak_height)

            # normalize again since maximum changed
            shoulder_diff_change = shoulder_diff_change / np.nanmax(shoulder_diff_change)

            # calculate upper envelope of signal to get rid of noisy parts (one big peak divided into several smaller)
            shoulder_diff_change = envelope(
                z=shoulder_diff_change, bp_in=(None, 100), squared=True
            )[0]

        # find peaks in corrected signal
        peaks, peak_properties = find_peaks(
            shoulder_diff_change, height=min_peak_height, distance=min_peak_distance
        )
        widths, width_heights, left_ips, right_ips = peak_widths(
            x=shoulder_diff_change, peaks=peaks, rel_height=relative_height
        )

        # create mask to indicate straight walking segments (straight=False, turn=True)
        turn_mask = np.zeros_like(right_shoulder_position)
        for left_base_idx, right_base_idx in zip(
            np.floor(left_ips).astype(int), np.ceil(right_ips).astype(int)
        ):
            turn_mask[left_base_idx:right_base_idx] = 1

        # create mask to indicate front vs back perspectives (front=True, back=False)
        perspective = np.where(left_shoulder_position > right_shoulder_position, "front", "back")
        turn_mask = np.where(turn_mask == 1, "turn", "straight")

        # assign indices to straight segments
        segment_idx = 0
        segment_idices = np.empty_like(perspective_change)
        for frame_idx, change in enumerate(perspective_change):
            if change != 0:
                segment_idx += 1
            segment_idices[frame_idx] = segment_idx

        # record boundary indices of segments
        segment_boundary_indices = np.concat(
            ([0], np.flatnonzero(perspective_change), [len(perspective_change)])
        )
        segment_boundaries = {}
        for idx, boundary in enumerate(segment_boundary_indices):
            if idx < len(segment_boundary_indices) - 1:
                segment_boundaries[idx] = [boundary, segment_boundary_indices[idx + 1]]

        # PLOT
        if self.debug_folder and CREATE_DEBUG_FIGS:

            plt.close("all")
            fig, axs = plt.subplots(4, 1, figsize=(14, 6))

            axs[0].plot(shoulder_diff_change, label="shoulder diff")
            axs[0].plot(left_ips, width_heights, "o", label="turn start", markersize=4)
            axs[0].plot(right_ips, width_heights, "o", label="turn end", markersize=4)
            axs[0].plot(turn_mask, label="straight/turn")
            axs[0].set_title("turn mask")

            axs[1].plot(left_shoulder_position, label="left")
            axs[1].plot(right_shoulder_position, label="right")
            axs[1].set_title("shoulder position (horizontal)")

            axs[2].plot(perspective, label="perspective")
            axs[2].set_title("Perspective")

            axs[3].plot(segment_idices, label="segment indices")
            axs[3].set_title("segment indices")
            axs[3].set_xlabel("Time [samples]")

            for ax in axs[:-2]:
                ax.legend(loc="upper left")
                ax.grid()

            plt.tight_layout()

            out_filename = os.path.join(
                self.debug_folder, f"{self.recording_type}_turns_and_perspective.png"
            )
            plt.savefig(out_filename)

            print(f"TPS figure saved to: \n{out_filename}")

        return (turn_mask, perspective, segment_idices, segment_boundaries)

    def get_gait_events(self, velocity, create_debug_fig: bool = False):
        heel_thr = self.heel_thr * velocity
        ankle_thr = self.heel_thr * velocity
        big_toe_thr = self.toe_thr * velocity

        all_events = []

        for side in ["left", "right"]:

            ankle = self.keypoints[self.labels.index(f"{side}_ankle"), :, :]
            ankle_vel = (
                np.linalg.norm(np.diff(ankle, axis=1, prepend=ankle[:, [0]]), axis=0) * self.fps
            )
            ground_contact_thrs = (ankle_vel < ankle_thr).astype(int)

            # remove events from turn segments
            ground_contact_straight = np.where(self.turn_mask == "straight", ground_contact_thrs, 1)

            contact_diff_straight = np.diff(
                ground_contact_straight, prepend=ground_contact_straight[0]
            )

            # remove too short swing periods
            ground_conact_no_short_swing = np.copy(ground_contact_straight)
            starts = np.nonzero(contact_diff_straight == -1)[0]
            ends = np.nonzero(contact_diff_straight == 1)[0]

            for start_frame, end_frame in zip(starts, ends):
                if end_frame - start_frame < self.min_swing_time:
                    ground_conact_no_short_swing[start_frame:end_frame] = 1

            contact_diff_no_short_swing = np.diff(
                ground_conact_no_short_swing, prepend=ground_conact_no_short_swing[0]
            )

            # remove too short contact periods
            ground_contact_no_short_contact = np.copy(ground_conact_no_short_swing)
            starts = np.nonzero(contact_diff_no_short_swing == 1)[0]
            ends = np.nonzero(contact_diff_no_short_swing == -1)[0]
            for idx, [start_frame, end_frame] in enumerate(zip(starts, ends)):
                if idx + 1 < ends.shape[0]:
                    if ends[idx + 1] - start_frame < self.min_stance_time:
                        ground_contact_no_short_contact[start_frame : ends[idx + 1]] = 0

            contact_diff_no_short_contact = np.diff(
                ground_contact_no_short_contact, prepend=ground_contact_no_short_contact[0]
            )

            IC_frames = np.nonzero(contact_diff_no_short_contact == 1)[0]
            FC_frames = np.nonzero(contact_diff_no_short_contact == -1)[0]

            IC_events = [
                GaitEvent(
                    type="IC",
                    frame=frame,
                    side=side,
                    perspective=self.perspective[frame],
                    segment=self.turn_mask[frame],
                    segment_index=self.segment_indices[frame],
                    relative_position=self.calculate_relative_position(frame_idx=frame),
                )
                for frame in IC_frames
            ]

            FC_events = [
                GaitEvent(
                    type="FC",
                    frame=frame,
                    side=side,
                    perspective=self.perspective[frame],
                    segment=self.turn_mask[frame],
                    segment_index=self.segment_indices[frame],
                    relative_position=self.calculate_relative_position(frame_idx=frame),
                )
                for frame in FC_frames
                if frame < len(self.turn_mask)
            ]

            all_events.extend(IC_events + FC_events)

            # PLOT
            if create_debug_fig:
                plt.close("all")
                fig, axs = plt.subplots(4, 1, figsize=(14, 7))

                axs[0].plot(ankle_vel, label="velocity")
                axs[0].plot(ground_contact_thrs * ankle_thr, label="ground contact at thr")

                axs[1].plot(ground_contact_thrs + 6, label="raw", color="#ff8800ff")
                axs[1].plot(ground_contact_straight + 4, label="no turns", color="#ffd000ff")
                axs[1].plot(
                    ground_conact_no_short_swing + 2, label="no short swings", color="#d9e717"
                )
                axs[1].plot(
                    ground_contact_no_short_contact, label="no short contacts", color="#6dc916"
                )

                axs[2].plot(contact_diff_straight + 4, label="no turns", color="#ffd000ff")
                axs[2].plot(
                    contact_diff_no_short_swing + 2, label="no short swings", color="#d9e717"
                )
                axs[2].plot(
                    contact_diff_no_short_contact, label="no short contact", color="#6dc916"
                )

                axs[3].plot(self.perspective, label="perspective")

                # fmt: off
                axs[0].set_title("ankle velocity")
                axs[1].set_title("ground contact")
                axs[2].set_title("ground contact diff (GEs)")

                axs[0].set_yticks([0, 3, ankle_thr])
                axs[0].set_ylim([0,3])
                # fmt: on
                for ax in axs:
                    ax.legend(loc="upper left")
                    ax.grid()

                plt.tight_layout()
                out_filename = os.path.join(
                    self.debug_folder, f"{self.recording_type}_gait_events_{side}.png"
                )
                plt.savefig(out_filename)
                plt.close("all")
                print(f"GE figure saved to: \n{out_filename}")
        all_events_sorted = sorted(all_events, key=lambda event: event.frame)

        return all_events_sorted

    def create_gait_cycles_figure(self):
        if self.debug_folder and CREATE_DEBUG_FIGS:
            plt.close("all")
            fig, axs = plt.subplots(3, 1, figsize=(14, 5))

            # show GEs of all gait cycles
            for gc in self.all_gait_cycles:
                alpha = 1 if gc.flags.valid else 0.25
                if gc.IC0:
                    axs[0].vlines(gc.IC0.frame, 0, 0.9, "c", linewidth=3, alpha=alpha)
                if gc.IC2:
                    axs[0].vlines(gc.IC0.frame, 1.1, 1.9, "m", linewidth=3, alpha=alpha)
                if gc.IC1:
                    axs[0].vlines(gc.IC0.frame, 2.1, 2.9, "r", linewidth=3, alpha=alpha)
                if gc.FC1:
                    axs[0].vlines(gc.IC0.frame, 3.1, 3.9, "b", linewidth=3, alpha=alpha)
                if gc.FC0:
                    axs[0].vlines(gc.IC0.frame, 4.1, 4.9, "g", linewidth=3, alpha=alpha)

            axs[0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5], ["IC0", "IC2", "IC1", "FC1", "FC0"])
            axs[0].set_xlim([0, len(self.perspective)])
            axs[0].set_title("gait cycle composition")

            # show flags of all gait cycles
            for gc in self.all_gait_cycles:
                if gc.flags.valid:
                    axs[1].vlines(gc.IC0.frame, 0, 0.9, "gray", linewidth=3)
                if gc.flags.perspectives_ok:
                    axs[1].vlines(gc.IC0.frame, 1.1, 1.9, "gray", linewidth=3)
                if gc.flags.segments_ok:
                    axs[1].vlines(gc.IC0.frame, 2.1, 2.9, "gray", linewidth=3)
                if gc.flags.types_ok:
                    axs[1].vlines(gc.IC0.frame, 3.1, 3.9, "gray", linewidth=3)
                if gc.flags.sides_ok:
                    axs[1].vlines(gc.IC0.frame, 4.1, 4.9, "gray", linewidth=3)

            axs[1].set_yticks(
                [0.5, 1.5, 2.5, 3.5, 4.5], ["valid", "persp", "segments", "types", "sides"]
            )
            axs[1].set_xlim([0, len(self.perspective)])
            axs[1].set_title("gait cycle flags")

            axs[2].plot(self.perspective)
            axs[2].plot(self.turn_mask)
            axs[2].set_xlim([0, len(self.perspective)])

            plt.tight_layout()
            # plt.show()

            out_filename = os.path.join(self.debug_folder, f"{self.recording_type}_gait_cycles.png")
            plt.savefig(out_filename)
            print(f"Gait cycles figure saved to: \n{out_filename}")

    def calculate_relative_position(self, frame_idx: int) -> float:
        """
        Calculates the relative (compared to segment start) position of GE within the segment it belongs to.

        Args:
            frame_idx: frame of the GE

        Returns:
            relative_position: position of GE's frame within the segment defined by self.segment_boundaries
        """
        for segment_idx, (start, end) in self.segment_boundaries.items():
            if start < frame_idx < end:
                segment_length = end - start
                relative_position = (frame_idx - start) / segment_length
                return relative_position

    def draw_keypoints(self):

        class KeypointsDrawer:
            def __init__(self, recording: Recording):
                self.recording = recording
                self.n_frames = self.recording.keypoints.shape[2]
                self.labels = self.recording.labels
                self.frame_idx = 0

                self.stop = False
                self.fig = plt.figure(figsize=(15, 10))
                self.ax = self.fig.add_subplot(111, projection="3d")
                self.fig.canvas.mpl_connect("key_press_event", self.on_key)
                self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
                self.plot_frame(self.frame_idx)
                plt.show()

            def plot_frame(self, idx):
                self.ax.clear()

                for kpt_idx, keypoint in enumerate(self.recording.keypoints):
                    x, y, z = keypoint[:, idx]
                    c = "b"
                    marker = ""
                    alpha = 0.2
                    if kpt_idx in [
                        self.labels.index("left_ankle"),
                        self.labels.index("right_ankle"),
                    ]:
                        c = "g"
                        alpha = 0.8

                    self.ax.scatter(x, y, z, c=c, s=15, alpha=alpha)

                self.ax.scatter(
                    3.09417443,
                    0.19429838,
                    -(-1.11145841),
                    marker="D",
                    c="r",
                    label="stereo camera",
                )
                self.ax.scatter(0, 0, 0, c="r", label="origin")

                self.ax.plot([0, 0], [0, 0.45], "blue", zs=[0, 0])
                self.ax.plot([0, 0.75], [0, 0], "blue", zs=[0, 0])
                self.ax.plot([0.75, 0.75], [0, 0.45], "blue", zs=[0, 0])
                self.ax.plot([0.75, 0], [0.45, 0.45], "blue", zs=[0, 0])

                self.ax.set_xlabel("x")
                self.ax.set_ylabel("y")
                self.ax.set_zlabel("z")
                self.ax.set_title(f"Frame {idx}")
                self.ax.set_xlim(-4, 4)
                self.ax.set_ylim(-0.75, 1.25)
                self.ax.set_zlim(0, 2)
                self.ax.set_aspect("equal")

                self.ax.legend()
                self.fig.canvas.draw_idle()

            def on_key(self, key_event):
                match key_event.key:
                    case "up":
                        self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
                    case "down":
                        self.frame_idx = max(self.frame_idx - 1, 0)
                    case "right":
                        self.frame_idx = min(self.frame_idx + 5, self.n_frames - 5)
                    case "left":
                        self.frame_idx = max(self.frame_idx - 5, 0)

                self.plot_frame(self.frame_idx)

            def on_scroll(self, mouse_event):
                step = int(mouse_event.step * 10)

                match mouse_event.button:
                    case "up":
                        self.frame_idx = min(self.frame_idx + step, self.n_frames - step)
                    case "down":
                        self.frame_idx = max(self.frame_idx + step, 0)

                self.plot_frame(self.frame_idx)

        self.keypoint_artist = KeypointsDrawer(self)


class RecordingPair:
    """
    Class representing the data and methods needed for comparison of 2 different types of recordings of the same participant.

    Attributes:
        folder_path: path to the folder containing all the input data (qualisys_keypoints, stereo_keypoins, calibraition img)
        qualisys_file_path: path to the file contatining keypoints obtained by qualisys system
        stereo_file_path: path to the file contatining keypoints obtained by stereo system
        debug_folder_path: path to the folder that should containe the figures and graphs used mainly for debugging
        qualisys: object of type Recording that contains data from qualisys
        stereo: object of type Recording that contains data from stereo system
        sync_pairs: list of GC object pairs that identifies the same GC in the stereo and qualisys data
        sync_shift: amount of frames by which the stereo system's data has to be shifted so the recordings' starts align
        statistics_results: dictionary contating the statistical results for each gait parameter obtaied by both system
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.qualisys_file_path = os.path.join(folder_path, "qualisys.npz")
        self.stereo_file_path = os.path.join(folder_path, "stereo.npz")
        self.debug_folder_path = os.path.join(folder_path, "Results")

        qualisys_keypoints, qualisys_labels = load_keypoints_and_labels(self.qualisys_file_path)
        stereo_keypoints, stereo_labels = load_keypoints_and_labels(self.stereo_file_path)

        self.qualisys = Recording(
            keypoints=qualisys_keypoints,
            labels=qualisys_labels,
            recording_type="qualisys",
            gait_analysis_properties=gait_analysis_properties,
            debug_folder=self.debug_folder_path,
        )
        self.stereo = Recording(
            keypoints=stereo_keypoints,
            labels=stereo_labels,
            recording_type="stereo",
            gait_analysis_properties=gait_analysis_properties,
            debug_folder=self.debug_folder_path,
        )
        self.get_sync_pairs()
        if CREATE_DEBUG_FIGS:
            self.create_sync_figure()

        self.statistics_results = self.create_result_container()
        self.get_statistics_results()
        self.save_statistics_results()

    def get_sync_pairs(self):

        gc_pairs = []
        window = 0.10

        stereo_first_turn = np.where(self.stereo.turn_mask == "turn")[0][0]
        qualisys_first_turn = np.where(self.qualisys.turn_mask == "turn")[0][0]
        shift = stereo_first_turn - qualisys_first_turn

        qualisys_gcs = np.copy(self.qualisys.gait_cycles).tolist()
        for gc in self.stereo.gait_cycles:
            potential_pairs = [
                p for p in qualisys_gcs if p.segment_index == gc.segment_index and p.ipsi == gc.ipsi
            ]
            choosen_pair = None
            # there are potential pair found
            if potential_pairs:

                if gc.segment_index != 0:
                    # rank potential pairs based on relative positions (position within segment, from segment end), put them in ascending order
                    # choose closest one if its within acceptable distance
                    potential_pair_ranks = sorted(
                        [
                            [p, abs(gc.relative_position - p.relative_position)]
                            for p in potential_pairs
                            if abs(gc.relative_position - p.relative_position) < window
                        ],
                        key=lambda x: x[1],
                    )

                    choosen_pair = potential_pair_ranks[0][0] if potential_pair_ranks else None

                else:
                    # rank potential pairs based on absolute positions (relative to first segment end), put them in ascending order, choose 1st one...
                    # Needed since recordings start at different times, first segments have different lengths which throws off relative calculaiton.
                    stereo_segment_end = self.stereo.segment_boundaries[0][1]
                    stereo_absolute_position = stereo_segment_end - gc.IC0.frame

                    qualisys_segment_end = self.qualisys.segment_boundaries[0][1]
                    potential_pair_ranks = sorted(
                        [
                            [
                                p,
                                abs(
                                    stereo_absolute_position - (qualisys_segment_end - p.IC0.frame)
                                ),
                            ]
                            for p in potential_pairs
                            if abs(stereo_absolute_position - (qualisys_segment_end - p.IC0.frame))
                            < 100
                        ],
                        key=lambda x: x[1],
                    )

                    choosen_pair = potential_pair_ranks[0][0] if potential_pair_ranks else None

                # pair was found
                if choosen_pair:
                    gc_pairs.append([gc, choosen_pair])
                    # remove choosen pair from pool potential pairs of the rest
                    qualisys_gcs.remove(choosen_pair)

            self.sync_pairs = np.asarray(gc_pairs)
            self.sync_shift = shift

    def create_sync_figure(self):
        plt.close("all")
        fig, axs = plt.subplots(2, 1, figsize=(14, 3))

        # plot turn masks
        axs[0].plot(
            self.stereo.turn_mask[self.sync_shift :], color="#000000", alpha=0.9, linestyle="-"
        )
        axs[1].plot(self.qualisys.turn_mask, color="#000000", alpha=0.9, linestyle="-")

        # plot valid GCs for both recording types
        for stereo_gc, qualisys_gc in zip(self.stereo.gait_cycles, self.qualisys.gait_cycles):
            axs[0].vlines(
                x=stereo_gc.IC0.frame - self.sync_shift, ymin=0, ymax=1, color="#138BDB76"
            )
            axs[1].vlines(x=qualisys_gc.IC0.frame, ymin=0, ymax=1, color="#138BDB76")

        # plot pairs
        colors = plt.get_cmap("prism", int(len(self.sync_pairs)))
        for color_idx, (stereo_gc, qualisys_gc) in enumerate(self.sync_pairs):
            axs[0].vlines(
                x=stereo_gc.IC0.frame - self.sync_shift,
                ymin=0,
                ymax=1,
                color=colors(color_idx),
                linewidth=4,
            )
            axs[1].vlines(
                x=qualisys_gc.IC0.frame, ymin=0, ymax=1, color=colors(color_idx), linewidth=4
            )

        axs[0].set_title("stereo")
        axs[1].set_title("qualisys")
        axs[0].set_xlim([0, self.qualisys.keypoints.shape[2]])
        axs[1].set_xlim([0, self.qualisys.keypoints.shape[2]])
        axs[1].set_xlabel("Time [samples]")

        plt.tight_layout()
        out_filename = os.path.join(self.debug_folder_path, f"sync.png")
        plt.savefig(out_filename)
        print(f"Sync figure saved to: \n{out_filename}")

    def sync_display(self):

        plt.close("all")
        fig, axs = plt.subplots(2, 1, figsize=(14, 4))

        # plot turn masks
        axs[0].plot(
            self.stereo.turn_mask[self.sync_shift :], color="#000000", alpha=0.9, linestyle="-"
        )
        axs[1].plot(self.qualisys.turn_mask, color="#000000", alpha=0.9, linestyle="-")

        # plot valid GCs for both recording types
        for stereo_gc, qualisys_gc in zip(self.stereo.gait_cycles, self.qualisys.gait_cycles):
            axs[0].vlines(
                x=stereo_gc.IC0.frame - self.sync_shift, ymin=0, ymax=1, color="#138BDB76"
            )
            axs[1].vlines(x=qualisys_gc.IC0.frame, ymin=0, ymax=1, color="#138BDB76")

        # plot pairs
        colors = plt.get_cmap("prism", int(len(self.sync_pairs)))
        for color_idx, (stereo_gc, qualisys_gc) in enumerate(self.sync_pairs):
            axs[0].vlines(
                x=stereo_gc.IC0.frame - self.sync_shift,
                ymin=0,
                ymax=1,
                color=colors(color_idx),
                linewidth=3.5,
            )
            axs[1].vlines(
                x=qualisys_gc.IC0.frame, ymin=0, ymax=1, color=colors(color_idx), linewidth=3.5
            )

        axs[0].set_title("stereo")
        axs[1].set_title("qualisys")
        axs[0].set_xlim([0, self.qualisys.keypoints.shape[2]])
        axs[1].set_xlim([0, self.qualisys.keypoints.shape[2]])
        axs[0].set_ylim([0, 2])
        axs[1].set_ylim([0, 2])
        axs[1].set_xlabel("Time [samples]")

        plt.tight_layout()

        def annotate_point(event):
            axes = event.inaxes
            # mouse event registered with one subplot, figure out which one
            if axes:
                if axes == axs[0]:
                    draw_ax = axs[0]
                    objects = self.stereo.gait_cycles
                    shift = self.sync_shift
                elif axes == axs[1]:
                    draw_ax = axs[1]
                    objects = self.qualisys.gait_cycles
                    shift = 0

                # set annotation style
                annotation = draw_ax.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"),
                )

                # find closest line/object
                cursor_x, cursor_y = event.xdata, event.ydata
                closest_obj = [
                    o
                    for o in objects
                    if abs(cursor_x - o.IC0.frame + shift) < 30 and abs(cursor_y - 0.5) < 0.5
                ]

                # if found, draw text
                if len(closest_obj) != 0:
                    closest_obj = closest_obj[0]
                    annotation.xy = [closest_obj.IC0.frame - shift, 1]
                    text = f"{closest_obj.IC0.frame:<5}{closest_obj.perspective:<7}{closest_obj.ipsi:<5}{'\n'}{round(closest_obj.IC0.relative_position,2)}"
                    annotation.set_text(text)
                    fig.canvas.draw_idle()

                # not found, erase every text
                else:
                    for annotation in draw_ax.texts:
                        annotation.remove()
                        fig.canvas.draw_idle()

        plt.connect("motion_notify_event", annotate_point)
        plt.show()

    def create_result_container(self):
        metrics = {"mean": np.nan, "std": np.nan, "cv": np.nan, "asymmetry": np.nan}

        # return statistics_results
        nested_dict = lambda: defaultdict(nested_dict)
        statistics_results = nested_dict()
        for perspective in ["front", "back", "all"]:
            for recording_type in ["qualisys", "stereo"]:
                for parameter in GAIT_PARAMETERS:
                    statistics_results[perspective][recording_type][parameter] = {}
        return statistics_results

    def get_statistics_results(self):
        """
        Calculates metrics for stereo and qualisys gait cycles present in self.sync_pairs for different perspectives.
        """
        for perspective in ["front", "back"]:
            for parameter in GAIT_PARAMETERS:
                for type_idx, recording_type in enumerate(["stereo", "qualisys"]):

                    values_left = [
                        getattr(gc.parameters, parameter, np.nan)
                        for gc in self.sync_pairs[:, type_idx]
                        if gc.perspective == perspective and gc.ipsi == "left"
                    ]
                    values_right = [
                        getattr(gc.parameters, parameter, np.nan)
                        for gc in self.sync_pairs[:, type_idx]
                        if gc.perspective == perspective and gc.ipsi == "right"
                    ]

                    values_all = np.concat((values_left, values_right))
                    mean = np.nanmean(values_all)
                    std = np.nanstd(values_all)
                    cv = 100 * std / mean if mean != 0 else np.nan
                    asymmetry = compute_asymmetry(values_left, values_right)

                    metrics = {"mean": mean, "std": std, "cv": cv, "asymmetry": asymmetry}

                    self.statistics_results["all"][recording_type][parameter] = metrics
                    self.statistics_results[perspective][recording_type][parameter] = metrics

    def print_statistics_results(self):
        stat = self.statistics_results
        rows = []
        for (
            (parameter_name, qualisys_front),
            stereo_front,
            qualisys_back,
            stereo_back,
            qualisys_all,
            stereo_all,
        ) in zip(
            stat["front"]["qualisys"].items(),
            stat["front"]["stereo"].values(),
            stat["back"]["qualisys"].values(),
            stat["back"]["stereo"].values(),
            stat["all"]["qualisys"].values(),
            stat["all"]["stereo"].values(),
        ):
            for (
                (metric_name, qualisys_front_value),
                stereo_front_value,
                qualisys_back_value,
                stereo_back_value,
                qualisys_all_value,
                stereo_all_value,
            ) in zip(
                qualisys_front.items(),
                stereo_front.values(),
                qualisys_back.values(),
                stereo_back.values(),
                qualisys_all.values(),
                stereo_all.values(),
            ):
                rows.append(
                    [
                        parameter_name,
                        metric_name,
                        qualisys_front_value,
                        stereo_front_value,
                        qualisys_back_value,
                        stereo_back_value,
                        qualisys_all_value,
                        stereo_all_value,
                    ]
                )
        pd.options.display.float_format = "{:,.1f}".format
        df = pd.DataFrame(
            rows,
            columns=[
                "PARAMETER",
                "METRIC",
                "Q FRONT",
                "S FRONT",
                "Q BACK",
                "S BACK",
                "Q ALL",
                "S ALL",
            ],
        )
        return df

    def save_statistics_results(self):
        statistics_save_path = os.path.join(self.folder_path, "", "stat_results.json")

        with open(statistics_save_path, "w") as f:
            json.dump(self.statistics_results, f)
        print(f"statistics results saved to: {statistics_save_path}")
