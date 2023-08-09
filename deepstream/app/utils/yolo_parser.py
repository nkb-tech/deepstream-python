import sys
import pyds
import time
import ctypes
import numpy as np


class BoxSizeParam:
    """ Class contaning base element for too small object box deletion. """

    def __init__(self, screen_height, screen_width,
                 min_box_height, min_box_width):
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.min_box_height = min_box_height
        self.min_box_width = min_box_width

    def is_percentage_sufficiant(self, percentage_height, percentage_width):
        """ Return True if detection box dimension is large enough,
            False otherwise.
        """
        res = self.screen_width * percentage_width > self.min_box_width
        res &= self.screen_height * percentage_height > self.min_box_height
        return res

    def is_negative(self, left_coordinate, top_coordinate):
        return left_coordinate < 0 or top_coordinate < 0


class NmsParam:
    """ Contains parameter for non maximal suppression algorithm. """

    def __init__(self, max_det=20, iou_thres=0.4, conf_thres=0.1):
        self.max_det = max_det
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres


def clip(elm, mini, maxi):
    """ Clips a value between mini and maxi."""
    return max(min(elm, maxi), mini)


def layer_finder(output_layer_info, name):
    """ Return the layer contained in output_layer_info which corresponds
        to the given name.
    """
    for layer in output_layer_info:
        # dataType == 0 <=> dataType == FLOAT
        if layer.dataType == 0 and layer.layerName == name:
            return layer
    return None

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    # output = [torch.zeros((0, 6), device=prediction.device)] * bs
    output = [np.zeros((0, 6))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = np.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            
            # conf, j = x[:, 5:].max(axis=1, keepdims=True)           #
            a = x[:, 5:]
            conf =  a.max(axis=1, keepdims=True)          
            j = np.argmax(a, axis=1)
            j = np.expand_dims(j, axis=1)
            # print(f"j.shape : {j.shape}")
            x = np.concatenate((box, conf, j.astype('float32')), 1)#[conf[:,0] > conf_thres]
            x = x[conf[:,0] > conf_thres]


        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thres)  # NMS
        i = np.array(i)
        # print(f"i.shape : {i.shape}")
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            sys.stderr.write(
                f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def make_nodi(index, det, box_size_param):
    """ Creates a NvDsInferObjectDetectionInfo object from one layer of yolo.
        Return None if the class Id is invalid, if the detection confidence
        is under the threshold or if the width/height of the bounding box is
        null/negative.
        Return the created NvDsInferObjectDetectionInfo object otherwise.
    """
    res = pyds.NvDsInferObjectDetectionInfo()
    res.detectionConfidence = det[4]
    res.classId = int(det[5])

    res.left = det[0] / box_size_param.screen_width
    res.top = det[1] / box_size_param.screen_height
    res.width = (det[2] - det[0]) / box_size_param.screen_width
    res.height = (det[3] - det[1]) / box_size_param.screen_height

    if not box_size_param.is_percentage_sufficiant(res.height, res.width) or box_size_param.is_negative(det[0], det[1]):
        return None

    return res


def nvds_infer_parse_custom_yolo(output_layer_info, box_size_param, nms_param=NmsParam()):
    """ Get data from output_layer_info and fill object_list
        with several NvDsInferObjectDetectionInfo.

        Keyword arguments:
        - output_layer_info : represents the neural network's output.
            (NvDsInferLayerInfo list)
        - detection_param : contains per class threshold.
            (DetectionParam)
        - box_size_param : element containing information to discard boxes
            that are too small. (BoxSizeParam)
        - nms_param : contains information for performing non maximal
            suppression. (NmsParam)

        Return:
        - Bounding boxes. (NvDsInferObjectDetectionInfo list)
    """

    output_layer = layer_finder(output_layer_info, "output0")

    if not output_layer:
        sys.stderr.write("ERROR: output layer missing\n")
        return []

    pred = ctypes.cast(pyds.get_ptr(output_layer.buffer), ctypes.POINTER(ctypes.c_float))
    pred = np.ctypeslib.as_array(pred, shape=(1, 25200, 85))

    pred = non_max_suppression(
        pred,
        max_det=nms_param.max_det,
        conf_thres=nms_param.conf_thres,
        iou_thres=nms_param.iou_thres,
    )
    pred = np.array(pred)

    num_detection = pred.shape[1]

    pred = pred[0]
    object_list = []
    for i in range(num_detection):
        obj = make_nodi(i, pred[i], box_size_param)
        if obj:
            object_list.append(obj)

    return object_list
