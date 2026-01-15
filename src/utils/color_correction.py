import pickle 

import cv2
import numpy as np
import colour as cl
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures


class referenceColor:
    def __init__(self):
        self.srgb_d50 = np.array([[115, 82, 68], # 1. Dark skin
                                [195, 149, 128], # 2. Light skin
                                [93, 123, 157], # 3. Blue sky
                                [91, 108, 65],  # 4. Foliage
                                [130, 129, 175], # 5. Blue flower
                                [99, 191, 171], # 6. Bluish green
                                [220, 123, 46], # 7. Orange
                                [72, 92, 168], # 8. Purplish blue 
                                [194, 84, 97], # 9. Moderate red 
                                [91, 59, 104], # 10. Purple
                                [161, 189, 62], # 11. Yellow green 
                                [229, 161, 40], # 12. Orange yellow
                                [42, 63, 147],  # 13. Blue 
                                [72, 149, 72], # 14. Green
                                [175, 50, 57], # 15. Red
                                [238, 200, 22], # 16. Yellow
                                [188, 84, 150], # 17. Magenta
                                [0, 137, 166],  # 18. Cyan
                                [245, 245, 240], # 19. White 9.5
                                [201, 202, 201], # 20. Neutral 8
                                [161, 162, 162], # 21. Neutral 6.5
                                [120, 121, 121], # 22. Neutral 5
                                [83, 85, 85], # 23. Neutral 3.5
                                [50, 50, 51] # 24. Black 2
                                ], dtype="uint8")

        self.srgb_d65 = []
        self.srgb_d55 = []
        self.srgb_d70 = []
        self.srgb_d75 = []

    def getReference(self, chart='Classic', ref_name='D50'):
        if chart == 'Classic' or chart == 'classic':
            Reference = eval('self.srgb_'+ref_name.lower())
            try:
                Reference = cv2.cvtColor(Reference[:, None], cv2.COLOR_RGB2BGR).astype(float) / 255
                Reference = Reference.reshape((-1, 3))
                # print(Reference)
                return Reference
            except AttributeError:
                raise ValueError('Unknown reference color space')
        else:
            raise ValueError('Chart not supported')


def get_patch_sizes(img1, max_value):
    shp = img1.shape[:2]

    max_value = 255

    v = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[:, :, 2]
    v = cv2.normalize(v, None, 0, max_value, cv2.NORM_MINMAX)
    ret, thresh = cv2.threshold(v, 0, max_value, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.01 * shp[0] * shp[1]
    # draw rectangles on image
    Dims = []
    for cnt in contours:
        x, y , w, h = cv2.boundingRect(cnt)

        ratio = w/h
        area = w*h
        if ratio < 0.9 or ratio > 1.20 or area < min_area:  # get rid of rectangles that are too small or too tall/wide
            continue
        # Append dimensions to list
        Dims.append((w,h))

    return np.round(np.mean(Dims, axis=0)).astype(int) 


def extract_color_chart_enhanced(img, max_value):
    """
    Detects a color checker on a contrast-enhanced version of the image,
    but extracts the final color values from the original image.
    """
    # 1. CREATE CONTRAST-ENHANCED IMAGE TO HELP DETECTION
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. DETECT THE CHECKER ON THE ENHANCED IMAGE
    detector = cv2.mcc.CCheckerDetector_create()
    # Use process instead of detect, as it's the recommended function
    if not detector.process(img_clahe, cv2.mcc.MCC24, 1):
        print("Color checker not found.")
        return None, None, None

    checkers = detector.getListColorChecker()
    img_draw = img.copy()  # Create a copy to draw on

    # Assuming only one checker is found
    for checker in checkers:
        # Draw the checker outline on the original image for visualization
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        cdraw.draw(img_draw)

        # 3. GET GEOMETRY FROM THE DETECTED CHECKER
        box_pts = checker.getBox()

        # Define the dimensions of the ideal, flattened color checker (6 patches wide, 4 high)
        # We can make it any size, e.g., 600x400, to maintain the aspect ratio
        w, h = 600, 400
        dst_pts = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype='float32')

        # 4. WARP THE *ORIGINAL* IMAGE
        # Calculate the perspective transform matrix and apply it
        transform_matrix = cv2.getPerspectiveTransform(box_pts.astype('float32'), dst_pts)
        warped_img = cv2.warpPerspective(img, transform_matrix, (w, h))

        # 5. EXTRACT COLORS FROM THE FLATTENED (WARPED) ORIGINAL IMAGE
        colors = []
        patch_w, patch_h = w // 6, h // 4  # Integer division to get patch size

        for i in range(4):  # Rows
            for j in range(6):  # Columns
                # Define the center of the patch
                cx = int(j * patch_w + patch_w / 2)
                cy = int(i * patch_h + patch_h / 2)

                # Define a small ROI in the middle of the patch to avoid edges
                roi_size = 10
                roi = warped_img[cy - roi_size:cy + roi_size, cx - roi_size:cx + roi_size]

                # Calculate the mean color (in BGR) and append
                mean_color = cv2.mean(roi)
                # cv2.mean returns (B, G, R, Alpha), we only need BGR
                colors.append(mean_color[:3])

        # Convert colors to a NumPy array with the correct shape (24, 3) and type
        # The order of colors in mcc is from top-left to bottom-right
        src = np.array(colors, dtype=np.uint8).reshape((24, 3))

        # This part of your original code can now use the original image ROI
        x1 = int(min(box_pts[:, 0]))
        x2 = int(max(box_pts[:, 0]))
        y1 = int(min(box_pts[:, 1]))
        y2 = int(max(box_pts[:, 1]))
        img_roi = img[y1:y2, x1:x2]
        dims = get_patch_sizes(img_roi, max_value)

        # The function now returns the correct colors from the original image
        return src, img_draw, dims

    # Return None if no checkers were processed
    return None, img, None


def extract_color_chart(img, max_value):
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(img, cv2.mcc.MCC24, 1)
    checkers = detector.getListColorChecker()

    for checker in checkers:
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        img_draw = img.copy()
        cdraw.draw(img_draw)

        chartsRGB = checker.getChartsRGB()
        width, height = chartsRGB.shape[:2]
        roi = chartsRGB[0:width, 1]

        box_pts = checker.getBox()
        x1 = int(min(box_pts[:, 0]))
        x2 = int(max(box_pts[:, 0]))
        y1 = int(min(box_pts[:, 1]))
        y2 = int(max(box_pts[:, 1]))

        # crop image to bounding box
        img_roi = img[y1:y2, x1:x2]
        dims = get_patch_sizes(img_roi, max_value)

        rows = int(roi.shape[:1][0])
        src = chartsRGB[:, 1].copy().reshape(int(rows/3), 1, 3)
        src = src.reshape(24, 3)

    return src, img_draw, dims


def DeltaE(rgb1, rgb2):
    lab1 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb1))
    lab2 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb2))
    deltaE = cl.difference.delta_E(lab1, lab2, method='CIE 2000')  # methods: 'CIE 1976', 'CIE 1994', 'CIE 2000',
    # 'CMC', 'CAM02-UCS', 'CAM02-SCD', 'CAM02-LCD', 'CAM16-UCS', 'CAM16-SCD', 'CAM16-LCD', 'cie2000', 'cie1994', 
    # 'cie1976', 'ITP', 'DIN99'

    mean_deltaE = np.mean(deltaE)
    min_deltaE = np.min(deltaE)
    max_deltaE = np.max(deltaE)
    sd_deltaE = np.std(deltaE)

    # print('Mean DeltaE = ', mean_deltaE)
    # print('Min DeltaE = ', min_deltaE)
    # print('Max DeltaE = ', max_deltaE)
    # print('SD DeltaE = ', sd_deltaE)

    return min_deltaE, max_deltaE, mean_deltaE, sd_deltaE, deltaE


def sigmoid_tone_map(values, contrast=1.0):
    """
    Applies a sigmoid curve to map values to the [0, 1] range.
    The input values are expected to be centered around 0.5.
    """
    # The formula is applied to values shifted to be centered around 0
    # The contrast parameter controls the steepness of the curve
    return 1.0 / (1.0 + np.exp(-contrast * (values - 0.5)))


def reinhard_tone_map(values):
    """
    Applies the Reinhard tone mapping operator to compress highlights.
    Shadows are clipped.
    """
    # Handle negative values first by clipping them to 0
    clipped_shadows = np.maximum(0, values)

    # Apply the Reinhard operator
    mapped_values = clipped_shadows / (1.0 + clipped_shadows)

    return mapped_values


def simple_clip(values):
    return np.clip(values, 0.0, 1.0)


def warp_extremes_normalized(values, thresholds=None):
    """
    Warp floating-point values to softly fit within the [0, 1] range.

    This function takes an array of values (typically color data) in a nominal
    [0, 1] range, which may have values slightly outside this range due to
    processing. It non-linearly compresses values below the low threshold and
    above the high threshold, ensuring a smooth transition ("soft clipping")
    instead of harsh clipping.

    The logic is a direct adaptation of the original function that worked on
    a [0, 255] scale.

    Args:
        values (np.ndarray): A numpy array of float values, nominally in the
                             [0, 1] range.
        thresholds (list, optional): A list of two floats [low, high] defining
                                     the range of values that will not be warped.
                                     If None, defaults to [5/255, 250/255], which
                                     matches the original function's behavior.

    Returns:
        np.ndarray: A numpy array of the same shape as `values` with the
                    extreme values warped to fit within [0, 1]. The dtype
                    will be float.
    """
    # Set default thresholds equivalent to the original [5, 250]
    if thresholds is None:
        thresholds = [5.0 / 255.0, 250.0 / 255.0]

    # Ensure input is float for calculations
    values = values.astype(np.float32)
    low_thresh, high_thresh = thresholds[0], thresholds[1]

    # --- Handle Highlights (values > high_thresh) ---
    overshoot_mask = values > high_thresh
    overshoot_values = values[overshoot_mask]

    if overshoot_values.size > 0:
        # Calculate how much each value is over the threshold
        diff = overshoot_values - high_thresh

        # The scaling factor 'k' determines the sharpness of the curve.
        # It's derived from the original function's behavior (0.5 on a 0-255 scale).
        # To get the same curve shape, we scale it by 255.
        # k = 0.5 * 255 = 127.5
        k = 127.5

        # The core warping formula, adapted for the [0, 1] range.
        # (1.0 - high_thresh) is the available "headroom".
        # (1 - (1 / np.exp(k * diff))) is the compression factor, which approaches 1.
        # warped_overshoot = high_thresh + (1.0 - high_thresh) * (1 - (1 / np.exp(k * diff)))
        warped_overshoot = high_thresh + (1.0 - high_thresh) * (1 - np.exp(-k * diff))

        # Place the warped values back into the array
        values[overshoot_mask] = warped_overshoot

    # --- Handle Shadows (values < low_thresh) ---
    undershoot_mask = values < low_thresh
    undershoot_values = values[undershoot_mask]

    if undershoot_values.size > 0:
        # Calculate how much each value is under the threshold
        diff = low_thresh - undershoot_values

        k = 127.5  # Use the same sharpness factor as for highlights

        # The core warping formula for shadows.
        # low_thresh is the available "footroom".
        warped_undershoot = low_thresh - low_thresh * (1 - (1 / np.exp(k * diff)))

        # Place the warped values back into the array
        values[undershoot_mask] = warped_undershoot

    # The function now returns floats. Clipping is a good safety measure
    # to handle any potential floating point inaccuracies at the very edges.
    return np.clip(values, 0.0, 1.0)


class ColorCorrectionNumpy:
    def __init__(self, degrees: int = 3, chart='Classic', illuminant='D50',
                 max_iter=5000, ncomp=None, interactions=True):
        self.degrees = degrees
        ref = referenceColor()
        self.chart = chart
        self.illuminant = illuminant
        self.ncomp = ncomp
        self.max_iter = max_iter
        self.interactions = interactions
        self.reference = ref.getReference(self.chart, self.illuminant)

    def calibrate(self, image, max_value):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        original_colors, img_draw, patch_size = extract_color_chart(image, max_value)
        original_colors_norm = original_colors / max_value

        init_DeltaE = DeltaE(original_colors_norm, self.reference)
        print('Initial mean DeltaE: ', init_DeltaE[2])

        wb = cv2.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(0.95)
        img_white_balance = wb.balanceWhite(image)

        white_balance_colors, _, _ = extract_color_chart(img_white_balance, max_value)
        white_balance_colors_norm = white_balance_colors / max_value

        poly_features = PolynomialFeatures(degree=self.degrees, interaction_only=self.interactions).fit_transform(white_balance_colors_norm)

        if self.ncomp is None or self.ncomp >= poly_features.shape[1]:
            self.ncomp = poly_features.shape[1]-1

        self.model = PLSRegression(n_components=self.ncomp, max_iter=self.max_iter)

        self.model.fit(poly_features, self.reference)

        colors_pred = self.model.predict(poly_features)

        init_DeltaE = DeltaE(colors_pred, self.reference)
        print('Final mean DeltaE: ', init_DeltaE[2])

        print('Model fit score: ', self.model.score(poly_features, self.reference))

        return self.model, img_draw, patch_size

    def process(self, image, max_value):
        # return image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        white_balanced = gray_world_white_balance(image, max_value)

        # return white_balanced
        white_balanced_norm = white_balanced / max_value
        white_balanced_norm_flatten = white_balanced_norm.reshape((-1, 3))

        features_img = PolynomialFeatures(degree=self.degrees,
                                          interaction_only=self.interactions).fit_transform(white_balanced_norm_flatten)

        predicted_image_norm = self.model.predict(features_img)
        predicted_image_norm = warp_extremes_normalized(predicted_image_norm)
        # predicted_image_norm = simple_clip(predicted_image_norm)
        # predicted_image_norm = reinhard_tone_map(predicted_image_norm)
        # predicted_image_norm = simple_clip(white_balanced_norm)
        predicted_image = predicted_image_norm * max_value

        predicted_image = predicted_image.reshape(image.shape)

        predicted_image = predicted_image.astype(image.dtype)

        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        return predicted_image

    def save(self, filepath: str):
        """Serialize this instance (including fitted model) to disk."""
        with open(filepath, 'wb') as f:
            # dump only the attributes you need
            state = {
                'degrees':      self.degrees,
                'chart':        self.chart,
                'illuminant':   self.illuminant,
                'max_iter':     self.max_iter,
                'ncomp':        self.ncomp,
                'interactions': self.interactions,
                'reference':    self.reference,
                'model':        self.model
            }
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'ColorCorrectionNumpy':
        """Reconstruct a saved instance from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # re‑init with the same hyper‑parameters
        obj = cls(
            degrees=state['degrees'],
            chart=state['chart'],
            illuminant=state['illuminant'],
            max_iter=state['max_iter'],
            ncomp=state['ncomp'],
            interactions=state['interactions']
        )

        # overwrite reference & model with the saved ones
        obj.reference = state['reference']
        obj.model = state['model']
        return obj
