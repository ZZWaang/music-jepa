import os
import json
import h5py
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class JepaDataset(Dataset):
    def __init__(
        self,
        data_list_path,  # Path to the JSON file (e.g., train.json, val.json, or test.json)
        data_dir="data/maestro",
        datasets=None,
        num_samples_per_clip=None,
        num_step=2,
        num_frame_per_step=400,
        overlap_ratio=0.00,
        split="train",
        num_examples=None,
        randomly_sample=False,
        feature_dim=249,  # no MFCC: Mel only is 229 bins, MFCC 20 bins
        pianoroll_dim=88,
        normalize_features=True,  # per-track feature normalization
        midi_chans=1,  # whether to store both velocity and onset (binary) in separate channels
        spec_pad_val=0.0,  # value to use for padding spectrograms (if needed)
        return_midi_id=False,  # whether to return midi_id, start_frame, end_frame for each segment
        pr_and_pedal_h5_path="data/maestro/maestro_midi.h5"
    ):
        """
        Args:
            data_list_path (str): Path to the JSON file (e.g., train.json, val.json, or test.json).
            data_dir (str): Directory where the H5 files are stored.
            num_samples_per_clip (int): Number of clips to sample per example (for training). This is ignored if randomly_sample=False, in which case all segments from the sliding window are used.
            num_step (int): Number of steps to return.
            num_frame_per_step (int): Frames per step.
            overlap_ratio (float): Overlap ratio for sliding window.
            split (str): "train", "validation", or "test".
            pianoroll_dim (int): MIDI pianoroll dimension.
            normalize_features (bool): Normalize audio features per track.
            pr_and_pedal_h5_path (str): Path to the H5 file containing pianoroll and pedal data.
        """
        with open(data_list_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
            print(f"Loaded {len(self.examples)} examples from {data_list_path}")

        # filter the examples
        self.examples = [
            ex
            for ex in self.examples
            if datasets is None or self.filter_examples(ex, datasets)
        ]
        print(f"Filtered examples: {len(self.examples)}")
        if num_examples is not None:
            # randomly select num_examples
            np.random.seed(0)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_examples]
            print(f"Randomly selected examples: {len(self.examples)}")
            # print out all midi_ids
            selected_midi_ids = set([ex["midi_id"] for ex in self.examples])
            print(f"Selected midi_ids: {selected_midi_ids}")

        self.data_dir = data_dir
        self.num_samples_per_clip = num_samples_per_clip
        self.num_step = num_step
        self.num_frame_per_step = num_frame_per_step
        self.total_frames = num_step * num_frame_per_step
        self.overlap_ratio = overlap_ratio
        self.split = split.lower()
        self.randomly_sample = randomly_sample
        self.feature_dim = feature_dim
        self.pianoroll_dim = pianoroll_dim
        self.normalize_features = normalize_features
        self.midi_chans = midi_chans
        self.spec_pad_val = spec_pad_val
        self.track_norm_cache = {}
        self.return_midi_id = return_midi_id
        self.pr_and_pedal_h5_path = pr_and_pedal_h5_path

        # Open all H5 files and store them in a dictionary.
        self.h5fs = None

        # Precompute number of segments per example if not randomly sampling.
        if not self.randomly_sample:
            self.segments_per_example = self.precompute_segments_per_example(self.examples)

        # Print some information.
        modalities = ["audio features (mel bins and MFCCs)", "MIDI note values", "pedal values"]
        print(
            f"Loaded {len(self.examples)} examples from {data_list_path} for split: {self.split}"
        )
        print(f"Using modalities: {', '.join(modalities)}")

    def _get_h5(self, file_path: str):
        """Open H5 lazily inside each worker process."""
        if self.h5fs is None:
            self.h5fs = {}

        if file_path not in self.h5fs:
            f = h5py.File(file_path, "r")
            self.h5fs[file_path] = f
        return self.h5fs[file_path]

    # (optional but recommended) so dataset can be pickled even if something opened later
    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5fs"] = None  # never pickle open handles
        return state

    def __del__(self):
        # close file handles when worker exits
        h5fs = getattr(self, "h5fs", None)
        if isinstance(h5fs, dict):
            for f in h5fs.values():
                try:
                    f.close()
                except Exception:
                    pass

    def precompute_segments_per_example(self, examples):
        segments_per_example = []
        for ex in examples:
            num_frames = ex["num_frames"]
            if num_frames > self.total_frames:
                # Compute number of segments with a sliding window.
                step = int(self.total_frames * (1 - self.overlap_ratio))
                segments = math.ceil((num_frames - self.total_frames) / step) + 1
            else:
                segments = 1
            segments_per_example.append({"num_segments": segments})
        return segments_per_example

    def get_track_normalization_stats(self, file_path, example_index):
        cache_key = (file_path, example_index)
        if cache_key in self.track_norm_cache:
            return self.track_norm_cache[cache_key]

        h5f = self._get_h5(file_path)
        full_features = h5f["features"][str(example_index)][
            :self.feature_dim, :
        ].T
        full_features = torch.tensor(full_features, dtype=torch.float32)
        mean = full_features.mean(dim=0, keepdim=True)
        std = full_features.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-6)

        self.track_norm_cache[cache_key] = (mean, std)
        return mean, std

    def normalize_feature_tensor(self, features, mean, std):
        if features.numel() == 0:
            print("[Warning] Attempting to normalize empty features")
            return features
        return (features - mean) / std

    def filter_examples(self, ex, datasets):
        """
        Filter examples based on the provided datasets.

        Args:
            ex (dict): Example to filter.
            datasets: ["r1-pf1", "r2-pf1", "r3-pf1", "r0-pf1"].
        """
        if datasets is None:
            return True
        room_id = str(ex["room_id"])
        pedal_factor = str(ex["pedal_factor"])
        if f"r{room_id}-pf{pedal_factor}" in datasets:
            return True
        else:
            return False

    def __len__(self):
        if not self.randomly_sample:
            return sum([seg["num_segments"] for seg in self.segments_per_example])
        else:
            return len(self.examples) * self.num_samples_per_clip

    def fetch_segment(self, file_path, example_index, start_frame, end_frame):
        h5f = self._get_h5(file_path)
        pr_and_pedal_h5f = self._get_h5(self.pr_and_pedal_h5_path)

        # Slice the feature, MIDI, and pedal arrays.
        selected_feature = h5f["features"][str(example_index)][
            :self.feature_dim, start_frame:end_frame
        ].T  # [max_frame, feature_dim]
        selected_pedal_value = pr_and_pedal_h5f["pedal"][
            str(example_index)
        ][0, start_frame:end_frame]
        selected_midi_values = pr_and_pedal_h5f["midi_values"][
            str(example_index)
        ][:, start_frame:end_frame]

        selected_feature = torch.tensor(selected_feature, dtype=torch.float32)
        if self.normalize_features:
            mean, std = self.get_track_normalization_stats(file_path, example_index)
            selected_feature = self.normalize_feature_tensor(selected_feature, mean, std)

        selected_midi = torch.tensor(selected_midi_values.T, dtype=torch.float32)
        selected_pedal = torch.tensor(selected_pedal_value, dtype=torch.float32)

        if selected_feature.shape[0] < self.total_frames:
            (selected_feature, selected_midi, selected_pedal) = self.pad_data(
                selected_feature,
                selected_midi,
                selected_pedal
            )

        if self.midi_chans == 2:
            previous_frame_midi = pr_and_pedal_h5f["midi_values"][str(example_index)][:, start_frame - 1] if start_frame != 0 else None
            previous_frame_midi = torch.tensor(previous_frame_midi, dtype=torch.float32) if previous_frame_midi is not None else None
            selected_midi = self.convert_midi_to_2_channel(selected_midi, previous_frame_midi=previous_frame_midi)

        return selected_feature, selected_midi, selected_pedal


    def get_segment_info(self, idx):
        # Map global idx to a specific example and segment.
        if not self.randomly_sample:
            running = 0
            for example_idx, example in enumerate(self.segments_per_example):
                seg_count = example["num_segments"]
                if idx < running + seg_count:
                    ex_idx = example_idx
                    seg_idx = idx - running
                    break
                running += seg_count
            else:
                raise IndexError("Index out of range in sliding window mode.")
        else:
            ex_idx = idx // self.num_samples_per_clip
            seg_idx = None  # For training, we'll sample a random segment.

        # Get JSON info for this example.
        ex_info = self.examples[ex_idx]
        file_path = os.path.join(self.data_dir, ex_info["file_path"])
        example_index = ex_info["example_index"]
        num_frames = ex_info["num_frames"]

        # Determine start_frame and end_frame.
        if self.randomly_sample:
            if num_frames > self.total_frames:
                start_frame = np.random.randint(0, num_frames - self.total_frames)
            else:
                start_frame = 0
            end_frame = start_frame + self.total_frames
        else: # sliding window
            step = int(self.total_frames * (1 - self.overlap_ratio))
            start_frame = seg_idx * step
            end_frame = min(start_frame + self.total_frames, num_frames)

        midi_id = ex_info["midi_id"]
        return file_path, example_index, start_frame, end_frame, midi_id

    def __getitem__(self, idx):
        file_path, example_index, start_frame, end_frame, midi_id = self.get_segment_info(idx)

        # Fetch the segment.
        (
            selected_feature,
            selected_midi,
            selected_pedal
        ) = self.fetch_segment(file_path, example_index, start_frame, end_frame)
        if selected_feature.shape[0] == 0:
            print("[Warning] Empty feature detected!")
            return self.__getitem__(np.random.randint(0, len(self)))

        if self.return_midi_id:
            return (selected_feature, selected_midi, selected_pedal, midi_id, start_frame, end_frame)
        else:
            return (selected_feature, selected_midi, selected_pedal)

    def convert_midi_to_2_channel(self, selected_midi, previous_frame_midi=None):
        selected_midi_2_channel = torch.zeros(2, selected_midi.shape[0], self.pianoroll_dim, dtype=selected_midi.dtype)
        selected_midi_2_channel[0, :, :] = selected_midi

        prev = torch.zeros_like(selected_midi)
        prev[1:, :] = selected_midi[:-1, :]
        prev[0] = previous_frame_midi if previous_frame_midi is not None else 0
        onset = ((selected_midi > 0) & (selected_midi != prev)).to(selected_midi.dtype) * 127
        selected_midi_2_channel[1, :, :] = onset

        return selected_midi_2_channel

    def pad_data(self, selected_feature, selected_midi, selected_pedal):
        pad_length = self.total_frames - selected_feature.shape[0]
        if pad_length <= 0:
            return selected_feature, selected_midi, selected_pedal

        selected_feature = F.pad(selected_feature, (0, 0, 0, pad_length), "constant", self.spec_pad_val)
        selected_midi = F.pad(selected_midi, (0, 0, 0, pad_length), "constant", 0)
        selected_pedal = F.pad(selected_pedal, (0, pad_length), "constant", 0)

        return selected_feature, selected_midi, selected_pedal
