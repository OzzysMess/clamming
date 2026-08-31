"""Track the vertical displacement of the marked pole in a video.

Example:
	python video_pole_tracking.py input.mp4 --reference ref_pole.png

The default ROI is the approximate pole box supplied for this experiment.
Use --pixels-per-segment to convert the measured pixel displacement to cm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROI = (765, 190, 35, 330)  # x, y, width, height


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("video", type=Path, help="input video file")
	parser.add_argument(
		"--reference",
		type=Path,
		default=None,
		help="optional pole image; its best match sets the initial ROI",
	)
	parser.add_argument("--output", type=Path, default=Path("pole_displacement.csv"))
	parser.add_argument(
		"--plot",
		type=Path,
		default=None,
		help="output displacement plot; defaults to the CSV name with .png",
	)
	parser.add_argument("--pixels-per-segment", type=float, required=True)
	parser.add_argument("--segment-cm", type=float, default=4.0)
	parser.add_argument("--search-pixels", type=int, default=45)
	parser.add_argument(
		"--annotated-video",
		type=Path,
		default=None,
		help="output video showing the highlighted tracking region; enabled by default",
	)
	return parser.parse_args()


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int):
	x, y, roi_width, roi_height = roi
	x = max(0, min(x, width - 1))
	y = max(0, min(y, height - 1))
	roi_width = min(roi_width, width - x)
	roi_height = min(roi_height, height - y)
	return x, y, roi_width, roi_height


def locate_with_reference(frame: np.ndarray, reference_path: Path, fallback_roi):
	reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
	if reference is None:
		raise FileNotFoundError(f"Could not read reference image: {reference_path}")
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# The supplied reference is a full scene image, not a pole crop.
	if reference.shape[0] > gray_frame.shape[0] * 0.8 or reference.shape[1] > gray_frame.shape[1] * 0.8:
		return fallback_roi
	if reference.shape[0] > gray_frame.shape[0] or reference.shape[1] > gray_frame.shape[1]:
		raise ValueError("Reference image is larger than the first video frame")
	result = cv2.matchTemplate(gray_frame, reference, cv2.TM_CCOEFF_NORMED)
	_, score, _, top_left = cv2.minMaxLoc(result)
	if score < 0.35:
		print(f"Reference match is weak ({score:.2f}); using the supplied ROI")
		return fallback_roi
	return top_left[0], top_left[1], reference.shape[1], reference.shape[0]


def track(video_path: Path, reference_path, output_path: Path, pixels_per_segment: float,
		  segment_cm: float, search_pixels: int, annotated_path, plot_path):
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise OSError(f"Could not open video: {video_path}")

	frame_rate = capture.get(cv2.CAP_PROP_FPS)
	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	ok, frame = capture.read()
	if not ok:
		capture.release()
		raise OSError("The video contains no readable frames")

	frame_height, frame_width = frame.shape[:2]
	roi = clamp_roi(DEFAULT_ROI, frame_width, frame_height)
	if reference_path is not None:
		roi = clamp_roi(locate_with_reference(frame, reference_path, roi), frame_width, frame_height)

	x, y, roi_width, roi_height = roi
	template = cv2.cvtColor(frame[y:y + roi_height, x:x + roi_width], cv2.COLOR_BGR2GRAY)
	if template.size == 0 or roi_width < 3 or roi_height < 3:
		capture.release()
		raise ValueError(f"ROI is outside the frame: {roi}")

	if annotated_path is None:
		annotated_path = output_path.with_name(f"{output_path.stem}_tracked.mp4")
	if plot_path is None:
		plot_path = output_path.with_suffix(".png")
	annotated_path.parent.mkdir(parents=True, exist_ok=True)
	codec = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(annotated_path), codec, frame_rate, (frame_width, frame_height))
	if not writer.isOpened():
		capture.release()
		raise OSError(f"Could not create annotated video: {annotated_path}")

	initial_y = y
	rows: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 1.0)]
	frame_index = 1
	while True:
		ok, frame = capture.read()
		if not ok:
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		search_y = max(0, y - search_pixels)
		search_bottom = min(frame_height, y + roi_height + search_pixels)
		search_x = max(0, x - 5)
		search_right = min(frame_width, x + roi_width + 5)
		search_area = gray[search_y:search_bottom, search_x:search_right]
		result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
		_, score, _, best = cv2.minMaxLoc(result)
		x = search_x + best[0]
		y = search_y + best[1]
		displacement_pixels = float(initial_y - y)
		displacement_cm = displacement_pixels * segment_cm / pixels_per_segment
		timestamp = frame_index / frame_rate if frame_rate else float(frame_index)
		rows.append((timestamp, displacement_pixels, displacement_cm, score))
		overlay = frame.copy()
		cv2.rectangle(overlay, (x, y), (x + roi_width, y + roi_height), (0, 220, 0), -1)
		frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
		cv2.rectangle(frame, (x, y), (x + roi_width, y + roi_height), (0, 255, 0), 2)
		center_x = x + roi_width // 2
		cv2.line(frame, (center_x, y), (center_x, y + roi_height), (0, 0, 255), 2)
		label = f"pole ROI | y displacement: {displacement_cm:.2f} cm | match: {score:.2f}"
		cv2.putText(frame, label, (max(8, x - 180), max(25, y - 10)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
		writer.write(frame)
		frame_index += 1

	capture.release()
	if writer is not None:
		writer.release()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savetxt(
		output_path,
		np.asarray(rows),
		delimiter=",",
		header="time_s,displacement_pixels,displacement_cm,match_score",
		comments="",
	)
	data = np.asarray(rows)
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axis = plt.subplots(figsize=(11, 5.5))
	axis.plot(data[:, 0], data[:, 2], color="#087f8c", linewidth=1.2)
	axis.axhline(0, color="#555555", linewidth=0.8)
	axis.set_title("Marked pole vertical displacement")
	axis.set_xlabel("Time (s)")
	axis.set_ylabel("Displacement (cm)")
	axis.grid(True, alpha=0.25)
	fig.tight_layout()
	fig.savefig(str(plot_path), dpi=160)
	plt.close(fig)
	print(f"Tracked {len(rows)} frames (video reports {frame_count}); wrote {output_path}")
	print(f"Wrote highlighted video {annotated_path} and plot {plot_path}")


def main() -> None:
	args = parse_args()
	if args.pixels_per_segment <= 0:
		raise ValueError("--pixels-per-segment must be positive")
	track(
		args.video,
		args.reference,
		args.output,
		args.pixels_per_segment,
		args.segment_cm,
		args.search_pixels,
		args.annotated_video,
		args.plot,
	)


if __name__ == "__main__":
	main()

