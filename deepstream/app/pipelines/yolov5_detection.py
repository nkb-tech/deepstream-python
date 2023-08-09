import sys
from typing import List, Optional

import cv2
import pyds

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from app.pipeline import Pipeline
from app.utils.yolo_parser import nvds_infer_parse_custom_yolo, NmsParam, BoxSizeParam
from app.utils.misc import get_label_names_from_file

UNTRACKED_OBJECT_ID = 0xffffffffffffffff


class YOLOv5DetectionPipeline(Pipeline):
    def __init__(self, *args, labels_file: str, **kwargs):
        super().__init__(*args, **kwargs)

        self.nms_param = NmsParam()
        self.box_param = BoxSizeParam(
            screen_height=self.input_height,
            screen_width=self.input_width,
            min_box_height=32,
            min_box_width=32,
        )

        self.target_classes = get_label_names_from_file(labels_file)

    def _yolo_detect(self, batch_meta, l_frame_meta: List, ll_obj_meta: List[List]):
        """YOLO-based detection pipeline."""
        for frame_meta in l_frame_meta:
            l_user = frame_meta.frame_user_meta_list
            while l_user is not None:
                try:
                    # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone.
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if (
                    user_meta.base_meta.meta_type
                    != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    continue

                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                # Boxes in the tensor meta should be in network resolution which is
                # found in tensor_meta.network_info. Use this info to scale boxes to
                # the input frame resolution.
                layers_info = []
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    layers_info.append(layer)
                
                frame_object_list = nvds_infer_parse_custom_yolo(
                    layers_info, self.box_param, self.nms_param,
                )

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

                for frame_object in frame_object_list:
                    self._add_obj_meta_to_frame(frame_object, batch_meta, frame_meta)

    def _add_obj_meta_to_frame(self, frame_object, batch_meta, frame_meta):
        """Inserts an object into the metadata."""
        # this is a good place to insert objects into the metadata.
        # Here's an example of inserting a single object.
        obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        # Set bbox properties. These are in input resolution.
        rect_params = obj_meta.rect_params
        rect_params.left = int(self.input_width * frame_object.left)
        rect_params.top = int(self.input_height * frame_object.top)
        rect_params.width = int(self.input_width * frame_object.width)
        rect_params.height = int(self.input_height * frame_object.height)

        # Semi-transparent yellow backgroud
        rect_params.has_bg_color = 0
        rect_params.bg_color.set(1, 1, 0, 0.4)

        # Red border of width 3
        # CUDA error ?
        # rect_params.border_width = 2
        # # set(red, green, blue, alpha); set to Red
        rect_params.border_color.set(1, 0, 0, 1)

        # Set object info including class, detection confidence, etc.
        obj_meta.confidence = frame_object.detectionConfidence
        obj_meta.class_id = frame_object.classId

        # There is no tracking ID upon detection. The tracker will
        # assign an ID.
        obj_meta.object_id = UNTRACKED_OBJECT_ID

        lbl_id = frame_object.classId
        if lbl_id >= len(self.target_classes):
            lbl_id = 0

        # Set the object classification label.
        obj_meta.obj_label = self.target_classes[lbl_id]

        # Set display text for the object.
        txt_params = obj_meta.text_params
        if txt_params.display_text:
            pyds.free_buffer(txt_params.display_text)

        txt_params.x_offset = int(rect_params.left)
        txt_params.y_offset = max(0, int(rect_params.top) - 10)
        txt_params.display_text = (
            self.target_classes[lbl_id] + " " + "{:04.3f}".format(frame_object.detectionConfidence)
        )
        # Font , font-color and font-size
        txt_params.font_params.font_name = "Serif"
        txt_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        txt_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        # Inser the object into current frame meta
        # This object has no parent
        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)


    def _add_probes(self):
        super()._add_probes()
        pgie_src_pad = self._get_static_pad(self.pgie, "src")
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self._yolo_detect))