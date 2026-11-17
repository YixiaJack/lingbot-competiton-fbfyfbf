"""Identical static frame-zero target cues for training and evaluation."""
import re


def parse_instruction(text):
    match = re.fullmatch(r'Pick up the (red|green|blue) block and place it on the (yellow|cyan|magenta) plate\.', text)
    if not match:
        return None
    return ('red', 'green', 'blue').index(match[1]), 3 + ('yellow', 'cyan', 'magenta').index(match[2])


def apply_cue(image, instruction, target_box, destination_box, mode):
    if mode == 'text' or target_box is None or destination_box is None:
        return image, instruction
    if mode == 'box':
        import cv2
        out = image.copy()
        for box, color in ((target_box, (255, 255, 255)), (destination_box, (255, 128, 0))):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2-1, y2-1), color, 2)
        return out, instruction
    if mode == 'coords':
        height, width = image.shape[:2]
        def norm(box):
            return ','.join(str(round(v / (width if i % 2 == 0 else height) * 1000)) for i, v in enumerate(box))
        return image, f'{instruction} Target box [0-1000]: ({norm(target_box)}). Destination box: ({norm(destination_box)}).'
    raise ValueError(mode)


def detected_boxes(detector, image, instruction, confidence=.7):
    """CPU YOLO once at episode start; low-confidence/ambiguous detections fall back."""
    selection = parse_instruction(instruction)
    if selection is None:
        return None, None
    # YOLO's numpy image API expects BGR; RoboTwin supplies RGB.
    result = detector.predict(image[..., ::-1], device='cpu', conf=.05, verbose=False)[0]
    chosen = []
    for cls in selection:
        candidates = sorted([(float(b.conf.item()), b.xyxy[0].cpu().tolist())
                             for b in result.boxes if int(b.cls.item()) == cls], reverse=True)
        if not candidates or candidates[0][0] < confidence or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .15):
            return None, None
        chosen.append(candidates[0][1])
    return tuple(chosen)
