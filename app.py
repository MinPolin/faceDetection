import numpy as np
import cv2

import math
from matplotlib import pyplot as plt
import matplotlib.colors as colors

cascade = 0
counter = 0


def createColorHistogram(img, binCount=256, out=plt):
    img = (img.astype(float)) / 255.0
    img_hsv = colors.rgb_to_hsv(img)

    img_hsv = img_hsv[: 0].flatten()

    out.hist(img_hsv * 360, binCount, range=(0.0, binCount), label='Hue')
    out.show()


class FaceDetection(object):
    def __init__(self, video_src):
        self.cam = cv2.VideoCapture(video_src)
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        cv2.namedWindow('mask')
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False

    def show_hist(self, hsv_roi, mask_roi):

        hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self.hist = hist.reshape(-1)

        bin_count = self.hist.shape[0]

        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                          (int(180.0 * i / bin_count), 255, 255), -1)

        cv2.imshow('hist', img)

    def detect(self, img, cascade):


        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(150, 150),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

        if len(rects) == 0:

            return []
        else:

            rects[:, 2:] += rects[:, :2]

            for rec in rects:
                rec[0] = rec[0] + int(math.floor(((rec[2] - rec[0]) * 0.4) / 2))
                rec[2] = rec[2] - int(math.floor(((rec[2] - rec[0]) * 0.4) / 2))

            return rects

    def draw_rects(self, img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def get_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self.detect(gray, cascade)
        self.rects = rects
        img = self.draw_rects(img, rects, (0, 255, 0))

        if len(rects) != 0:
            self.selection = rects[0][1], rects[0][0], rects[0][3], rects[0][2]

        return rects

    def blur_and_skin(self, mask, vis):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        skin = cv2.bitwise_and(vis, vis, mask=mask)
        return mask, skin

    def get_mask(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 51., 89.)), np.array((17., 140., 255.)))

        return hsv, mask

    def get_tr_coords(self, hsv, mask, vis):
        self.selection = None
        prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
        prob &= mask
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

        if self.show_backproj:
            vis[:] = prob[..., np.newaxis]
        pts = np.int0(cv2.boxPoints(track_box))
        return pts

    def run(self):

        counter = 0
        rects = None

        while True:
            counter += 1
            ret, self.frame = self.cam.read()
            vis = self.frame.copy()

            if counter % 150 == 0:
                rects = self.get_face(vis)

            hsv, mask = self.get_mask()
            mask, skin = self.blur_and_skin(mask, vis)

            if rects is not None:
                self.draw_rects(vis, rects, (0, 255, 0))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1 - x0, y1 - y0)
                hsv_roi = hsv[x0:x1, y0:y1]
                mask_roi = mask[x0:x1, y0:y1]

                self.show_hist(hsv_roi, mask_roi)

                vis_roi = vis[x0:x1, y0:y1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0
                self.tracking_state = 1
                self.selection = None

            if self.tracking_state == 1:
                pts = self.get_tr_coords(hsv, mask, vis)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

            cv2.imshow('camshift', np.hstack([vis, skin]))
            cv2.imshow('mask', mask)

            ch = 0xFF & cv2.waitKey(5)
            if ch == ord('q'):
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    cascade_fn = 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_fn)
    video_src = 0

    FaceDetection(video_src).run()
