import argparse
import csv
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import (
    TimeDomain,
    TimeQueryOptions,
)


def find_streams(provider):
    streams = provider.get_all_streams()
    left_sid = None
    right_sid = None
    imu_sid = None
    for s in streams:
        label = provider.get_label_from_stream_id(s)
        if label == "slam-front-left":
            left_sid = s
        elif label == "slam-front-right":
            right_sid = s
        elif label == "imu-right":
            imu_sid = s
    if left_sid is None:
        raise RuntimeError("Could not find slam-front-left")
    if right_sid is None:
        raise RuntimeError("Could not find slam-front-right")
    if imu_sid is None:
        raise RuntimeError("Could not find imu-right")
    print("[INFO] Using slam-front-left")
    print("[INFO] Using slam-front-right")
    print("[INFO] Using imu-right")
    return left_sid, right_sid, imu_sid


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def extract_camera(provider, stream_id, output_dir, image_extension=".png"):
    ensure_dir(output_dir)
    timestamps = provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
    exposures = []
    print(f"[INFO] Extracting {len(timestamps)} frames -> {output_dir}")
    for _, ts in enumerate(tqdm(timestamps)):
        img, metadata = provider.get_image_data_by_time_ns(
            stream_id,
            ts,
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )
        filename = f"{ts}{image_extension}"
        cv2.imwrite(str(output_dir / filename), img.to_numpy_array())
        exposure_ns = int(metadata.exposure_duration * 1e9)
        exposures.append((ts, exposure_ns))
    return timestamps, exposures


def write_image_csv(timestamps, cam_folder, image_extension=".png"):
    data_csv = cam_folder / "data.csv"
    with open(data_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "filename"])
        for ts in timestamps:
            writer.writerow([ts, f"{ts}{image_extension}"])


def write_exposure_csv(exposures, csv_path):
    with open(csv_path, "w") as f:
        f.write("#timestamp [ns],exposure time[ns]\n")
        for ts, exposure_ns in exposures:
            f.write(f"{ts},{exposure_ns}\n")


def write_imu_csv(provider, imu_stream_id, output_csv):
    timestamps = provider.get_timestamps_ns(imu_stream_id, TimeDomain.DEVICE_TIME)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "#timestamp [ns]",
                "w_RS_S_x [rad s^-1]",
                "w_RS_S_y [rad s^-1]",
                "w_RS_S_z [rad s^-1]",
                "a_RS_S_x [m s^-2]",
                "a_RS_S_y [m s^-2]",
                "a_RS_S_z [m s^-2]",
            ]
        )
        print(f"[INFO] Extracting {len(timestamps)} samples -> {output_csv}")
        for ts in tqdm(timestamps):
            imu = provider.get_imu_data_by_time_ns(
                imu_stream_id,
                ts,
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.CLOSEST,
            )
            if not (imu.accel_valid and imu.gyro_valid):
                continue
            gyro = imu.gyro_radsec
            accel = imu.accel_msec2
            writer.writerow(
                [
                    ts,
                    gyro[0],
                    gyro[1],
                    gyro[2],
                    accel[0],
                    accel[1],
                    accel[2],
                ]
            )


def convert_imu_to_dso(imu_csv, imu_txt):
    with open(imu_csv) as f_in, open(imu_txt, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            timestamp = parts[0]
            gyro = parts[1:4]
            accel = parts[4:7]
            f_out.write(" ".join([timestamp] + gyro + accel) + "\n")


def make_dso_times(data_csv, exposure_csv, out_txt):
    with open(data_csv) as f:
        data_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    with open(exposure_csv) as f:
        exp_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    assert len(data_lines) == len(exp_lines)
    with open(out_txt, "w") as f:
        f.write("# filename timestamp exposure\n")
        for dl, el in zip(data_lines, exp_lines):
            ts_ns, filename = dl.split(",")
            _, exposure_ns = el.split(",")
            ts_s = int(ts_ns) / 1e9
            if int(exposure_ns) < 0:
                exposure_ms = 0.0
            else:
                exposure_ms = int(exposure_ns) / 1e6
            stem = Path(filename).stem
            f.write(f"{stem} " f"{ts_s:.9f} " f"{exposure_ms:.6f}\n")


def convert_vrs(vrs_file, output_folder):
    # Make directories
    mav0 = output_folder / "mav0"
    cam0_data = mav0 / "cam0" / "data"
    cam1_data = mav0 / "cam1" / "data"
    imu0 = mav0 / "imu0"
    ensure_dir(cam0_data)
    ensure_dir(cam1_data)
    ensure_dir(imu0)

    # Extract data
    provider = data_provider.create_vrs_data_provider(str(vrs_file))
    left_sid, right_sid, imu_sid = find_streams(provider)
    left_ts, left_exp = extract_camera(provider, left_sid, cam0_data)
    right_ts, right_exp = extract_camera(provider, right_sid, cam1_data)
    write_image_csv(left_ts, mav0 / "cam0")
    write_image_csv(right_ts, mav0 / "cam1")
    write_exposure_csv(left_exp, mav0 / "cam0" / "exposure.csv")
    write_exposure_csv(right_exp, mav0 / "cam1" / "exposure.csv")
    imu_csv = imu0 / "data.csv"
    write_imu_csv(provider, imu_sid, imu_csv)

    # Convert to DSO structure
    dso = output_folder / "dso"
    ensure_dir(dso / "cam0")
    ensure_dir(dso / "cam1")
    cam0_link = dso / "cam0" / "images"
    cam1_link = dso / "cam1" / "images"
    if cam0_link.exists() or cam0_link.is_symlink():
        cam0_link.unlink()
    if cam1_link.exists() or cam1_link.is_symlink():
        cam1_link.unlink()
    os.symlink("../../mav0/cam0/data", cam0_link)
    os.symlink("../../mav0/cam1/data", cam1_link)
    convert_imu_to_dso(imu_csv, dso / "imu_orig.txt")
    make_dso_times(
        mav0 / "cam0" / "data.csv",
        mav0 / "cam0" / "exposure.csv",
        dso / "cam0" / "times.txt",
    )
    make_dso_times(
        mav0 / "cam1" / "data.csv",
        mav0 / "cam1" / "exposure.csv",
        dso / "cam1" / "times.txt",
    )
    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrs_file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    convert_vrs(args.vrs_file, args.output_folder)
