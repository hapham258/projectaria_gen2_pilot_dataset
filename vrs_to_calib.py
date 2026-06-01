import argparse
import json
from pathlib import Path

from scipy.spatial.transform import Rotation as scipyRot
from projectaria_tools.core import data_provider


def transform_to_qvec_tvec(pose):
    R_mat = pose[:3, :3]
    tvec = pose[:3, 3]
    qxyzw = scipyRot.from_matrix(R_mat).as_quat()
    qvec = [
        float(qxyzw[0]),
        float(qxyzw[1]),
        float(qxyzw[2]),
        float(qxyzw[3]),
    ]
    return qvec, list(map(float, tvec))


def camera_to_json(device_calib, cam_name):
    cam = device_calib.get_camera_calib(cam_name)
    width, height = cam.get_image_size()
    pose = cam.get_transform_device_camera().to_matrix()
    qvec, tvec = transform_to_qvec_tvec(pose)
    return {
        "model": cam.get_model_name().name,
        "resolution": {
            "width": int(width),
            "height": int(height),
        },
        "params": list(
            map(
                float,
                cam.get_projection_params(),
            )
        ),
        "T_b_s": {
            "qvec": qvec,
            "tvec": tvec,
        },
    }


def imu_to_json(device_calib, imu_name):
    imu = device_calib.get_imu_calib(imu_name)
    pose = imu.get_transform_device_imu().to_matrix()
    qvec, tvec = transform_to_qvec_tvec(pose)
    return {
        "T_b_s": {
            "qvec": qvec,
            "tvec": tvec,
        },
    }


def export_factory_calibration(vrs_file, output_folder):
    #
    output_folder.mkdir(parents=True, exist_ok=True)
    output_json = output_folder / "factory_calibration.json"
    provider = data_provider.create_vrs_data_provider(str(vrs_file))
    device_calib = provider.get_device_calibration()

    #
    out = {}
    out["cam0"] = camera_to_json(device_calib, "slam-front-left")
    out["cam1"] = camera_to_json(device_calib, "slam-front-right")
    out["imu0"] = imu_to_json(device_calib, "imu-right")
    with open(output_json, "w") as f:
        json.dump(out, f, indent=4)
    print(f"[INFO] Saved: {output_json}")


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
    export_factory_calibration(args.vrs_file, args.output_folder)
