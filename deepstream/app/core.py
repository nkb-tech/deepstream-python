import os
import logging

from app.pipelines import (
    Pipeline,
    AnonymizationPipeline,
    ReIDPipeline,
    SegmentationPipeline,
    YOLOv5DetectionPipeline,
)
from app.config import CONFIGS_DIR, LOGLEVEL

logging.basicConfig(level=LOGLEVEL)


def run_pipeline(video_uri: str):
    pipeline = Pipeline(
        video_uri=video_uri,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/pgie.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        output_format="mp4",
    )
    pipeline.run()


def run_segmentation_pipeline(video_uri: str):
    pipeline = SegmentationPipeline(
        video_uri=video_uri,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/segmentation.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        output_format="mp4",
    )
    pipeline.run()


def run_anonymization_pipeline(video_uri: str):
    pipeline = AnonymizationPipeline(
        video_uri=video_uri,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/yolov4.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        target_classes=[2],
        enable_osd=False,
    )
    pipeline.run()


def run_reid_pipeline(video_uri: str):
    pipeline = ReIDPipeline(
        video_uri=video_uri,
        target_classes=[0],
        save_crops=False,
    )
    pipeline.run()


def run_yolov5_detection_pipeline(video_uri: str):
    pipeline = YOLOv5DetectionPipeline(
        video_uri=video_uri,
        output_video_path="/usr/src/deepstream-python/deepstream/out.mp4",
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/yolov5/config_infer_yolov5_triton.pbtxt"),
        labels_file=os.path.join(CONFIGS_DIR, "classnames/coco.txt"),
        tracker_config_path=None,
        save_crops=True,
        input_shape=(640, 640),
    )
    pipeline.run()
