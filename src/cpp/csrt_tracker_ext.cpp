#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

namespace py = pybind11;

// Simple single-object CSRT tracker wrapper.
//
// NOTE: This is an initial C++ skeleton. The goal is to mirror the
// behaviour of the current Python CSRTTracker, but for now this class
// only implements basic init/update without anti-stick or re-ID logic.
// We will extend it step by step.
class SingleCsrtTracker {
public:
    SingleCsrtTracker(int max_misses, double motion_threshold, int max_static_frames)
        : max_misses_(max_misses),
          motion_threshold_(motion_threshold),
          max_static_frames_(max_static_frames),
          has_track_(false),
          dormant_(false),
          misses_(0),
          static_frames_(0),
          last_timestamp_(0.0) {}

    // Initialize tracker from an RGB frame and global bbox (x1, y1, x2, y2)
    void init(py::array_t<uint8_t> frame, py::tuple bbox, double timestamp) {
        cv::Mat mat = numpy_to_mat(frame);
        if (mat.empty()) {
            throw std::runtime_error("Empty frame passed to SingleCsrtTracker::init");
        }

        if (bbox.size() != 4) {
            throw std::runtime_error("bbox must be a 4-tuple (x1, y1, x2, y2)");
        }

        double x1 = bbox[0].cast<double>();
        double y1 = bbox[1].cast<double>();
        double x2 = bbox[2].cast<double>();
        double y2 = bbox[3].cast<double>();
        double w  = std::max(1.0, x2 - x1);
        double h  = std::max(1.0, y2 - y1);

        cv::Rect2d rect(x1, y1, w, h);
        cv::Rect rect_i(cvRound(rect.x), cvRound(rect.y), cvRound(rect.width), cvRound(rect.height));

        tracker_ = create_csrt();
        if (!tracker_) {
            throw std::runtime_error("Failed to create CSRT tracker");
        }

        tracker_->init(mat, rect_i);
        bool ok = true;
        if (!ok) {
            tracker_.release();
            has_track_ = false;
            dormant_ = false;
            throw std::runtime_error("CSRT init() failed");
        }

        bbox_ = rect;
        has_track_ = true;
        dormant_ = false;
        misses_ = 0;
        static_frames_ = 0;

        // Initialize motion-heuristic history on the ROI only (around bbox_),
        // щоб не рахувати різницю по всьому кадру.
        cv::Mat gray = to_gray(mat);
        cv::Rect roi = clamp_rect_to_image(rect_i, gray.size());
        if (roi.width > 0 && roi.height > 0) {
            last_roi_gray_ = gray(roi).clone();
            last_roi_ = roi;
        } else {
            last_roi_gray_.release();
            last_roi_ = cv::Rect();
        }

        last_timestamp_ = timestamp;
    }

    // Reset from detection (same as init but keeps existing instance semantics).
    void reset(py::array_t<uint8_t> frame, py::tuple bbox, double timestamp) {
        // For now just call init() – semantics are the same for single-track use.
        init(frame, bbox, timestamp);
    }

    // Update on a new frame. Returns (ok, dormant, x1, y1, x2, y2, motion).
    py::tuple update(py::array_t<uint8_t> frame, double timestamp) {
        cv::Mat mat = numpy_to_mat(frame);
        if (mat.empty()) {
            return py::make_tuple(false, dormant_, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        if (!has_track_ || !tracker_ || dormant_) {
            return py::make_tuple(false, dormant_, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        cv::Rect2d rect2d = bbox_;
        cv::Rect rect(cvRound(rect2d.x), cvRound(rect2d.y), cvRound(rect2d.width), cvRound(rect2d.height));
        bool ok = tracker_->update(mat, rect);
        double motion = 0.0;

        if (!ok) {
            misses_++;
            if (misses_ > max_misses_) {
                has_track_ = false;
                dormant_ = true;
                tracker_.release();
            }
            return py::make_tuple(false, dormant_, 0.0, 0.0, 0.0, 0.0, motion);
        }

        // Basic motion-based heuristic – similar to Python anti-stick logic,
        // але рахуємо тільки в околі bbox (ROI), а не по всьому кадру.
        cv::Mat gray = to_gray(mat);
        cv::Rect roi = clamp_rect_to_image(rect, gray.size());
        if (roi.width > 0 && roi.height > 0) {
            cv::Mat cur_roi = gray(roi);
            if (!last_roi_gray_.empty() &&
                last_roi_gray_.size() == cur_roi.size() &&
                last_roi_gray_.type() == cur_roi.type()) {
                cv::Mat diff;
                cv::absdiff(cur_roi, last_roi_gray_, diff);
                cv::Scalar mean_val = cv::mean(diff);
                motion = mean_val[0];
                if (motion < motion_threshold_) {
                    static_frames_++;
                } else {
                    static_frames_ = 0;
                }
                if (static_frames_ > max_static_frames_) {
                    // Consider this a static background, mark as dormant.
                    has_track_ = false;
                    dormant_ = true;
                    tracker_.release();
                    last_roi_gray_ = cur_roi.clone();
                    last_roi_ = roi;
                    last_timestamp_ = timestamp;
                    return py::make_tuple(false, dormant_, 0.0, 0.0, 0.0, 0.0, motion);
                }
            }
            last_roi_gray_ = cur_roi.clone();
            last_roi_ = roi;
        } else {
            last_roi_gray_.release();
            last_roi_ = cv::Rect();
        }

        bbox_ = cv::Rect2d(rect);
        last_timestamp_ = timestamp;
        misses_ = 0;

        double x1 = bbox_.x;
        double y1 = bbox_.y;
        double x2 = bbox_.x + bbox_.width;
        double y2 = bbox_.y + bbox_.height;

        return py::make_tuple(true, dormant_, x1, y1, x2, y2, motion);
    }

    bool has_track() const { return has_track_ && !dormant_; }
    bool is_dormant() const { return dormant_; }

private:
    // Helper: convert numpy array (H,W,3) or (H,W) uint8 to cv::Mat
    static cv::Mat numpy_to_mat(const py::array_t<uint8_t>& arr) {
        py::buffer_info info = arr.request();
        if (info.ndim == 2) {
            int h = static_cast<int>(info.shape[0]);
            int w = static_cast<int>(info.shape[1]);
            return cv::Mat(h, w, CV_8UC1, info.ptr).clone();
        } else if (info.ndim == 3) {
            int h = static_cast<int>(info.shape[0]);
            int w = static_cast<int>(info.shape[1]);
            int c = static_cast<int>(info.shape[2]);
            if (c == 3) {
                return cv::Mat(h, w, CV_8UC3, info.ptr).clone();
            } else {
                throw std::runtime_error("Unsupported channel count in frame (expected 1 or 3)");
            }
        } else {
            throw std::runtime_error("Unsupported frame ndim (expected 2 or 3)");
        }
    }

    static cv::Mat to_gray(const cv::Mat& mat) {
        if (mat.channels() == 1) {
            return mat.clone();
        }
        cv::Mat gray;
        cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    // Clamp rectangle to valid image region
    static cv::Rect clamp_rect_to_image(const cv::Rect& r, const cv::Size& sz) {
        int x1 = std::max(0, r.x);
        int y1 = std::max(0, r.y);
        int x2 = std::min(sz.width, r.x + r.width);
        int y2 = std::min(sz.height, r.y + r.height);
        if (x2 <= x1 || y2 <= y1) {
            return cv::Rect();
        }
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }

    static cv::Ptr<cv::Tracker> create_csrt() {
        return cv::TrackerCSRT::create();
    }

private:
    cv::Ptr<cv::Tracker> tracker_;
    cv::Rect2d bbox_;
    bool has_track_;
    bool dormant_;
    int max_misses_;
    double motion_threshold_;
    int max_static_frames_;
    int misses_;
    int static_frames_;
    double last_timestamp_;
    // Anti-stick history on ROI only, щоб не рахувати diff по всьому кадру.
    cv::Mat last_roi_gray_;
    cv::Rect last_roi_;
};

PYBIND11_MODULE(csrt_tracker_ext, m) {
    m.doc() = "C++ CSRT tracker extension (single-object)";

    py::class_<SingleCsrtTracker>(m, "SingleCsrtTracker")
        .def(py::init<int, double, int>(),
             py::arg("max_misses") = 10,
             py::arg("motion_threshold") = 3.0,
             py::arg("max_static_frames") = 8)
        .def("init", &SingleCsrtTracker::init,
             py::arg("frame"),
             py::arg("bbox"),
             py::arg("timestamp"))
        .def("reset", &SingleCsrtTracker::reset,
             py::arg("frame"),
             py::arg("bbox"),
             py::arg("timestamp"))
        .def("update", &SingleCsrtTracker::update,
             py::arg("frame"),
             py::arg("timestamp"))
        .def("has_track", &SingleCsrtTracker::has_track)
        .def("is_dormant", &SingleCsrtTracker::is_dormant);
}
