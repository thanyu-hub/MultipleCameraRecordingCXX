#include <QApplication>
#include <QCloseEvent>
#include <QEvent>
#include <QFocusEvent>
#include <QFontMetrics>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QShowEvent>
#include <QTimer>
#include <QWidget>

#include <opencv2/opencv.hpp>

#include "CameraUrls.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kFreshFrameMaxAgeSeconds = 2.0;
constexpr double kNoFrameMaxAgeSeconds = 5.0;
constexpr int kPreviewMaxWidth = 1920;
constexpr int kPreviewMaxHeight = 1080;
constexpr int kPreviewIntervalMs = 66;
constexpr int kOpenTimeoutMs = 3000;
constexpr int kReadTimeoutMs = 3000;
constexpr int kMaxConsecutiveReadFailures = 10;
constexpr int kWorkerJoinTimeoutMs = 6000;
constexpr int kSaveJoinTimeoutMs = 6000;

struct CameraConfig {
    std::string name;
    std::string ip;
    std::string url;
};

struct AppConfig {
    std::string saveRoot;
    std::vector<CameraConfig> cameras;
};

enum class CameraState {
    Connecting,
    Ready,
    NoFrame,
    Disconnected,
    Disabled,
};

struct CameraSnapshot {
    CameraConfig config;
    CameraState state = CameraState::Connecting;
    cv::Mat frame;
    bool hasFrame = false;
    double secondsSinceFrame = -1.0;
    int frameWidth = 0;
    int frameHeight = 0;
    bool hasFreshFrame = false;
    QImage previewImage;
    std::string message;
};

struct CameraSaveResult {
    CameraConfig config;
    bool ok = false;
    bool skipped = false;
    std::string path;
    std::string error;
    int width = 0;
    int height = 0;
};

struct SaveResult {
    int captureIndex = 0;
    int expectedCount = 0;
    int successCount = 0;
    int skippedCount = 0;
    std::vector<CameraSaveResult> cameras;
};

struct SaveTaskState {
    std::atomic<bool> done{false};
    SaveResult result;
};

std::string nowTimestampForFolder() {
    std::time_t now = std::time(nullptr);
    std::tm tmNow{};
    localtime_r(&now, &tmNow);
    std::ostringstream oss;
    oss << std::put_time(&tmNow, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string nowTimestampForLog() {
    std::time_t now = std::time(nullptr);
    std::tm tmNow{};
    localtime_r(&now, &tmNow);
    std::ostringstream oss;
    oss << std::put_time(&tmNow, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string zeroPad(int value, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << value;
    return oss.str();
}

std::string sanitizeForFilename(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '.' || ch == '-' || ch == '_') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "unknown" : out;
}

std::string homeDir() {
    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return std::string(home);
    }
    return ".";
}

std::string expandTilde(const std::string& path) {
    if (path == "~") {
        return homeDir();
    }
    if (path.rfind("~/", 0) == 0) {
        return homeDir() + path.substr(1);
    }
    return path;
}

std::string defaultSaveRoot() {
    return homeDir() + "/record_data/captures";
}

fs::path defaultConfigPath() {
    return fs::path(homeDir()) / ".config" / "multicam_capture" / "config.json";
}

std::string defaultRtspUrlForIp(const std::string& ip) {
    return "rtsp://" + ip + ":554/h264Preview_01_main";
}

AppConfig defaultConfig() {
    return AppConfig{
        defaultSaveRoot(),
        {
            {"Camera 1", "192.168.0.20", defaultRtspUrlForIp("192.168.0.20")},
            {"Camera 2", "192.168.0.21", defaultRtspUrlForIp("192.168.0.21")},
            {"Camera 3", "192.168.0.22", defaultRtspUrlForIp("192.168.0.22")},
            {"Camera 4", "192.168.0.23", defaultRtspUrlForIp("192.168.0.23")},
        },
    };
}

std::vector<CameraConfig> unconfiguredCameras() {
    return {
        {"Camera 1", "not configured", ""},
        {"Camera 2", "not configured", ""},
        {"Camera 3", "not configured", ""},
        {"Camera 4", "not configured", ""},
    };
}

void useUnconfiguredCameras(AppConfig& config) {
    config.cameras = unconfiguredCameras();
}

std::string makeSessionFolderName(const std::string& sessionTimestamp, const std::string& outputLabel) {
    if (outputLabel.empty()) {
        return sessionTimestamp;
    }
    return sessionTimestamp + "_" + outputLabel;
}

std::string makeOutputLabel(const std::string& outputName, int rowNumber) {
    return sanitizeForFilename(outputName) + "-row" + std::to_string(rowNumber);
}

bool createUniqueSessionDirectory(
    const fs::path& saveRoot,
    const std::string& sessionTimestamp,
    const std::string& outputLabel,
    fs::path& saveDir,
    std::string& error) {
    try {
        fs::create_directories(saveRoot);
        const std::string baseName = makeSessionFolderName(sessionTimestamp, outputLabel);
        for (int suffix = 0; suffix < 1000; ++suffix) {
            std::ostringstream name;
            name << baseName;
            if (suffix > 0) {
                name << "_" << std::setw(3) << std::setfill('0') << suffix;
            }
            const fs::path candidate = saveRoot / name.str();
            if (fs::create_directory(candidate)) {
                saveDir = candidate;
                return true;
            }
        }
        error = "Could not create a unique session folder under " + saveRoot.string();
        return false;
    } catch (const std::exception& ex) {
        error = std::string("Failed to create save folder: ") + ex.what();
        return false;
    }
}

bool isFreshFrameAge(double secondsSinceFrame) {
    return secondsSinceFrame >= 0.0 && secondsSinceFrame <= kFreshFrameMaxAgeSeconds;
}

std::string appConfigToJson(const AppConfig& config) {
    QJsonObject root;
    root["save_root"] = QString::fromStdString(config.saveRoot);

    QJsonArray cameras;
    for (const auto& camera : config.cameras) {
        QJsonObject cameraObject;
        cameraObject["name"] = QString::fromStdString(camera.name);
        cameraObject["ip"] = QString::fromStdString(camera.ip);
        cameraObject["url"] = QString::fromStdString(camera.url);
        cameras.append(cameraObject);
    }
    root["cameras"] = cameras;

    const QByteArray json = QJsonDocument(root).toJson(QJsonDocument::Indented);
    return std::string(json.constData(), static_cast<std::size_t>(json.size()));
}

bool writeDefaultConfig(const fs::path& path, const AppConfig& config, std::string& error) {
    try {
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        if (!out) {
            error = "Failed to open config for writing: " + path.string();
            return false;
        }
        out << appConfigToJson(config);
        return true;
    } catch (const std::exception& ex) {
        error = std::string("Failed to create default config: ") + ex.what();
        return false;
    }
}

bool readRequiredJsonString(const QJsonObject& object, const QString& key, std::string& value) {
    const QJsonValue field = object.value(key);
    if (!field.isString()) {
        return false;
    }
    value = field.toString().toStdString();
    return true;
}

bool readOptionalJsonString(const QJsonObject& object, const QString& key, std::string& value) {
    const QJsonValue field = object.value(key);
    if (field.isUndefined() || field.isNull()) {
        value.clear();
        return true;
    }
    if (!field.isString()) {
        return false;
    }
    value = field.toString().toStdString();
    return true;
}

bool loadConfig(const fs::path& path, AppConfig& config, std::vector<std::string>& startupMessages) {
    config.saveRoot = defaultSaveRoot();
    useUnconfiguredCameras(config);
    if (!fs::exists(path)) {
        config = defaultConfig();
        std::string error;
        if (writeDefaultConfig(path, config, error)) {
            startupMessages.push_back("Created default config: " + path.string());
        } else {
            startupMessages.push_back(error);
        }
        return true;
    }

    std::ifstream in(path);
    if (!in) {
        startupMessages.push_back("Failed to read config. Cameras are disabled: " + path.string());
        return true;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    QJsonParseError parseError{};
    const QJsonDocument document = QJsonDocument::fromJson(
        QByteArray(text.data(), static_cast<int>(text.size())),
        &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        startupMessages.push_back("Config is invalid JSON. Cameras are disabled: " + parseError.errorString().toStdString());
        return true;
    }

    const QJsonObject root = document.object();
    const QJsonValue saveRootValue = root.value("save_root");
    if (saveRootValue.isString()) {
        config.saveRoot = expandTilde(saveRootValue.toString().toStdString());
    } else {
        startupMessages.push_back("Config is missing save_root. Using default save root.");
    }

    const QJsonValue camerasValue = root.value("cameras");
    if (!camerasValue.isArray()) {
        startupMessages.push_back("Config is missing cameras. Cameras are disabled.");
        return true;
    }

    const QJsonArray camerasArray = camerasValue.toArray();
    if (camerasArray.size() != 4) {
        startupMessages.push_back("Config must contain exactly four cameras. Cameras are disabled.");
        return true;
    }

    std::vector<CameraConfig> parsedCameras;
    parsedCameras.reserve(4);
    for (int i = 0; i < camerasArray.size(); ++i) {
        if (!camerasArray[i].isObject()) {
            startupMessages.push_back("Config camera " + std::to_string(i) + " is invalid. Cameras are disabled.");
            return true;
        }

        const QJsonObject cameraObject = camerasArray[i].toObject();
        CameraConfig camera;
        if (!readRequiredJsonString(cameraObject, "name", camera.name) ||
            !readRequiredJsonString(cameraObject, "ip", camera.ip) ||
            !readOptionalJsonString(cameraObject, "url", camera.url)) {
            startupMessages.push_back("Config camera " + std::to_string(i) + " is missing name, ip, or url. Cameras are disabled.");
            return true;
        }

        parsedCameras.push_back(camera);
    }

    config.cameras = parsedCameras;
    return true;
}

bool hasRuntimeRtspCredentials() {
    return multicamHasRtspCredentials();
}

void applyRuntimeRtspCredentials(AppConfig& config, std::vector<std::string>& startupMessages) {
    if (!hasRuntimeRtspCredentials()) {
        return;
    }

    int updatedCount = 0;
    for (auto& camera : config.cameras) {
        if (!camera.ip.empty() && camera.url == defaultRtspUrlForIp(camera.ip)) {
            camera.url = multicamRtspUrlForIp(camera.ip);
            ++updatedCount;
        }
    }
    if (updatedCount > 0) {
        startupMessages.push_back(
            "Applied RTSP credentials from " + multicamRtspCredentialSource() +
            " for " + std::to_string(updatedCount) + " camera(s).");
    }
}

std::string redactedUrl(const std::string& url) {
    return multicamRedactUrlCredentials(url);
}

class Logger {
public:
    explicit Logger(const fs::path& logPath) {
        if (!logPath.empty()) {
            try {
                logFile_.open(logPath);
                if (logFile_) {
                    logPath_ = logPath.string();
                }
            } catch (const std::exception& ex) {
                std::cerr << "Failed to open session log: " << ex.what() << std::endl;
            }
        }
    }

    void info(const std::string& message) {
        write("INFO", message);
    }

    void warn(const std::string& message) {
        write("WARN", message);
    }

    void error(const std::string& message) {
        write("ERROR", message);
    }

    std::string path() const {
        return logPath_;
    }

private:
    void write(const std::string& level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string line = nowTimestampForLog() + " [" + level + "] " + message;
        std::cout << line << std::endl;
        if (logFile_) {
            logFile_ << line << '\n';
            logFile_.flush();
        }
    }

    mutable std::mutex mutex_;
    std::ofstream logFile_;
    std::string logPath_;
};

QImage matToQImage(const cv::Mat& mat);

cv::Mat makePreviewFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }
    const double scale = std::min(
        static_cast<double>(kPreviewMaxWidth) / static_cast<double>(frame.cols),
        static_cast<double>(kPreviewMaxHeight) / static_cast<double>(frame.rows));
    if (scale >= 1.0) {
        return frame.clone();
    }

    cv::Mat preview;
    cv::resize(
        frame,
        preview,
        cv::Size(std::max(1, static_cast<int>(frame.cols * scale)), std::max(1, static_cast<int>(frame.rows * scale))),
        0,
        0,
        cv::INTER_AREA);
    return preview;
}

class CameraWorker {
public:
    CameraWorker(CameraConfig config, std::shared_ptr<Logger> logger)
        : config_(std::move(config)), logger_(std::move(logger)) {}

    ~CameraWorker() {
        requestStop();
        if (!detached_) {
            join();
        }
    }

    CameraWorker(const CameraWorker&) = delete;
    CameraWorker& operator=(const CameraWorker&) = delete;

    void start() {
        finished_.store(false);
        thread_ = std::thread(&CameraWorker::run, this);
    }

    void requestStop() {
        stopRequested_.store(true);
        reconnectRequested_.store(true);
    }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool joinIfFinished() {
        if (!thread_.joinable()) {
            return true;
        }
        if (!finished_.load()) {
            return false;
        }
        thread_.join();
        return true;
    }

    void detachIfStillRunning() {
        if (thread_.joinable()) {
            thread_.detach();
            detached_ = true;
        }
    }

    void stop() {
        requestStop();
        if (!detached_) {
            join();
        }
    }

    void requestReconnect() {
        if (config_.url.empty()) {
            setStatus(CameraState::Disabled, "Disabled");
            return;
        }
        reconnectRequested_.store(true);
        setStatus(CameraState::Connecting, "Reconnect requested");
        if (logger_) {
            logger_->info(config_.name + " reconnect requested");
        }
    }

    CameraSnapshot snapshot(bool includeOriginal) const {
        std::lock_guard<std::mutex> lock(mutex_);
        CameraSnapshot out;
        out.config = config_;
        out.state = state_;
        out.hasFrame = !latestFrame_.empty();
        out.previewImage = previewImage_;
        if (out.hasFrame) {
            out.frameWidth = latestFrame_.cols;
            out.frameHeight = latestFrame_.rows;
            const auto age = std::chrono::duration<double>(Clock::now() - lastFrameTime_).count();
            out.secondsSinceFrame = age;
            out.hasFreshFrame = isFreshFrameAge(age);
            if (age > kNoFrameMaxAgeSeconds) {
                out.state = CameraState::NoFrame;
                out.message = "No recent frame";
            } else if (!out.hasFreshFrame) {
                out.message = "Delayed";
            } else {
                out.message = message_;
                if (includeOriginal) {
                    latestFrame_.copyTo(out.frame);
                }
            }
        } else {
            out.message = message_;
        }
        return out;
    }

private:
    void run() {
        struct FinishGuard {
            std::atomic<bool>& finished;
            ~FinishGuard() {
                finished.store(true);
            }
        };

        finished_.store(false);
        FinishGuard finishGuard{finished_};

        while (!stopRequested_.load()) {
            if (config_.url.empty()) {
                setStatus(CameraState::Disabled, "Disabled");
                if (logger_) {
                    logger_->info(config_.name + " disabled because RTSP URL is empty");
                }
                while (!stopRequested_.load() && config_.url.empty()) {
                    sleepWithStop(1000);
                }
                continue;
            }

            setStatus(CameraState::Connecting, "Connecting");
            if (logger_) {
                logger_->info(config_.name + " connecting to " + redactedUrl(config_.url));
            }

            cv::VideoCapture cap;
            try {
                const std::vector<int> params{
                    cv::CAP_PROP_OPEN_TIMEOUT_MSEC, kOpenTimeoutMs,
                    cv::CAP_PROP_READ_TIMEOUT_MSEC, kReadTimeoutMs,
                };
                cap.open(config_.url, cv::CAP_FFMPEG, params);
                if (cap.isOpened()) {
                    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                }
            } catch (const std::exception& ex) {
                setStatus(CameraState::Disconnected, std::string("Open exception: ") + ex.what());
                if (logger_) {
                    logger_->error(config_.name + " open exception: " + ex.what());
                }
                sleepWithStop(1000);
                continue;
            }

            if (!cap.isOpened()) {
                setStatus(CameraState::Disconnected, "Failed to open stream");
                if (logger_) {
                    logger_->error(config_.name + " failed to open stream: " + redactedUrl(config_.url));
                }
                sleepWithStop(1000);
                continue;
            }

            reconnectRequested_.store(false);
            if (logger_) {
                logger_->info(config_.name + " connected");
            }

            cv::Mat frame;
            Clock::time_point lastPreviewTime{};
            int consecutiveFailures = 0;
            while (!stopRequested_.load() && !reconnectRequested_.load()) {
                bool ok = false;
                try {
                    ok = cap.read(frame);
                } catch (const std::exception& ex) {
                    setStatus(CameraState::NoFrame, std::string("Read exception: ") + ex.what());
                    if (logger_) {
                        logger_->error(config_.name + " read exception: " + ex.what());
                    }
                    break;
                }

                if (!ok || frame.empty()) {
                    ++consecutiveFailures;
                    if (consecutiveFailures == 1 || consecutiveFailures % 50 == 0) {
                        setStatus(CameraState::NoFrame, "No frame received");
                        if (logger_) {
                            logger_->warn(config_.name + " did not return a frame");
                        }
                    }
                    if (consecutiveFailures >= kMaxConsecutiveReadFailures) {
                        if (logger_) {
                            logger_->warn(config_.name + " read failures reached reconnect threshold");
                        }
                        break;
                    }
                    sleepWithStop(100);
                    continue;
                }

                consecutiveFailures = 0;
                const auto now = Clock::now();
                QImage preview;
                const bool shouldUpdatePreview =
                    lastPreviewTime.time_since_epoch().count() == 0 ||
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPreviewTime).count() >= kPreviewIntervalMs;
                if (shouldUpdatePreview) {
                    preview = matToQImage(makePreviewFrame(frame));
                    lastPreviewTime = now;
                }

                std::lock_guard<std::mutex> lock(mutex_);
                frame.copyTo(latestFrame_);
                if (!preview.isNull()) {
                    previewImage_ = preview;
                }
                lastFrameTime_ = now;
                state_ = CameraState::Ready;
                message_ = "Ready";
            }

            cap.release();
            if (!stopRequested_.load()) {
                setStatus(CameraState::Disconnected, "Disconnected");
                if (logger_) {
                    logger_->warn(config_.name + " disconnected");
                }
                sleepWithStop(500);
            }
        }
    }

    void setStatus(CameraState state, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = state;
        message_ = message;
    }

    void sleepWithStop(int milliseconds) const {
        const int stepMs = 50;
        int elapsed = 0;
        while (!stopRequested_.load() && elapsed < milliseconds) {
            std::this_thread::sleep_for(std::chrono::milliseconds(stepMs));
            elapsed += stepMs;
        }
    }

    CameraConfig config_;
    std::shared_ptr<Logger> logger_;
    mutable std::mutex mutex_;
    std::thread thread_;
    std::atomic<bool> stopRequested_{false};
    std::atomic<bool> reconnectRequested_{false};
    std::atomic<bool> finished_{false};
    bool detached_ = false;
    cv::Mat latestFrame_;
    QImage previewImage_;
    Clock::time_point lastFrameTime_ = Clock::now();
    CameraState state_ = CameraState::Connecting;
    std::string message_ = "Starting";
};

QImage matToQImage(const cv::Mat& mat) {
    if (mat.empty()) {
        return {};
    }

    cv::Mat converted;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, converted, cv::COLOR_BGR2RGB);
        return QImage(converted.data, converted.cols, converted.rows, static_cast<int>(converted.step), QImage::Format_RGB888).copy();
    }
    if (mat.channels() == 4) {
        cv::cvtColor(mat, converted, cv::COLOR_BGRA2RGBA);
        return QImage(converted.data, converted.cols, converted.rows, static_cast<int>(converted.step), QImage::Format_RGBA8888).copy();
    }
    if (mat.channels() == 1) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
    }
    return {};
}

QColor colorForState(const CameraSnapshot& snapshot, bool recentSaveFailure) {
    if (recentSaveFailure) {
        return QColor(220, 60, 60);
    }
    if (snapshot.state == CameraState::Ready && isFreshFrameAge(snapshot.secondsSinceFrame)) {
        return QColor(70, 190, 100);
    }
    if (snapshot.state == CameraState::Ready || snapshot.state == CameraState::Connecting) {
        return QColor(225, 180, 50);
    }
    if (snapshot.state == CameraState::NoFrame || snapshot.state == CameraState::Disconnected) {
        return QColor(220, 60, 60);
    }
    if (snapshot.state == CameraState::Disabled) {
        return QColor(130, 130, 130);
    }
    return QColor(130, 130, 130);
}

std::string stateLabel(const CameraSnapshot& snapshot, bool recentSaveFailure) {
    if (recentSaveFailure) {
        return "Save failed";
    }
    switch (snapshot.state) {
        case CameraState::Connecting:
            return "Connecting";
        case CameraState::Ready:
            if (snapshot.secondsSinceFrame > kFreshFrameMaxAgeSeconds || snapshot.message == "Delayed") {
                return "Delayed";
            }
            return "Ready";
        case CameraState::NoFrame:
            return "No frame";
        case CameraState::Disconnected:
            return "Disconnected";
        case CameraState::Disabled:
            return "Disabled";
    }
    return "Unknown";
}

SaveResult saveCaptureSet(
    int captureIndex,
    const fs::path& saveDir,
    std::vector<CameraSnapshot> snapshots,
    std::shared_ptr<Logger> logger) {
    SaveResult result;
    result.captureIndex = captureIndex;
    result.cameras.reserve(snapshots.size());

    if (logger) {
        logger->info("Capture " + zeroPad(captureIndex, 6) + " requested");
    }

    std::vector<int> pngParams{cv::IMWRITE_PNG_COMPRESSION, 1};
    for (std::size_t i = 0; i < snapshots.size(); ++i) {
        const auto& snapshot = snapshots[i];
        CameraSaveResult cameraResult;
        cameraResult.config = snapshot.config;

        if (snapshot.state == CameraState::Disabled || snapshot.config.url.empty()) {
            cameraResult.skipped = true;
            cameraResult.error = "Disabled";
            ++result.skippedCount;
            if (logger) {
                logger->info(snapshot.config.name + " skipped because RTSP URL is empty ip=" + snapshot.config.ip);
            }
            result.cameras.push_back(cameraResult);
            continue;
        }

        ++result.expectedCount;
        if (!snapshot.hasFrame || snapshot.frame.empty()) {
            cameraResult.error = "No original frame available";
            if (snapshot.hasFrame && !snapshot.hasFreshFrame) {
                cameraResult.error = "No fresh original frame available; last frame age is " +
                                     std::to_string(snapshot.secondsSinceFrame) + " seconds";
            }
            if (logger) {
                logger->error(snapshot.config.name + " save failed: " + cameraResult.error + " ip=" + snapshot.config.ip);
            }
            result.cameras.push_back(cameraResult);
            continue;
        }

        const std::string fileName =
            "image_" + zeroPad(captureIndex, 6) +
            "_cam" + std::to_string(i) +
            "_" + sanitizeForFilename(snapshot.config.ip) + ".png";
        const fs::path outputPath = saveDir / fileName;
        cameraResult.path = outputPath.string();
        cameraResult.width = snapshot.frame.cols;
        cameraResult.height = snapshot.frame.rows;

        try {
            if (fs::exists(outputPath)) {
                cameraResult.error = "Refusing to overwrite existing file";
                if (logger) {
                    logger->error(snapshot.config.name + " save failed: " + cameraResult.error + " path=" + outputPath.string());
                }
                result.cameras.push_back(cameraResult);
                continue;
            }

            const fs::path tempPath = outputPath.string() + ".tmp.png";
            if (fs::exists(tempPath)) {
                fs::remove(tempPath);
            }
            const bool ok = cv::imwrite(tempPath.string(), snapshot.frame, pngParams);
            if (ok) {
                fs::rename(tempPath, outputPath);
                cameraResult.ok = true;
                ++result.successCount;
                if (logger) {
                    logger->info(
                        snapshot.config.name + " saved " + outputPath.string() +
                        " resolution=" + std::to_string(snapshot.frame.cols) +
                        "x" + std::to_string(snapshot.frame.rows));
                }
            } else {
                cameraResult.error = "cv::imwrite returned false";
                if (fs::exists(tempPath)) {
                    fs::remove(tempPath);
                }
                if (logger) {
                    logger->error(snapshot.config.name + " save failed: " + cameraResult.error + " path=" + outputPath.string());
                }
            }
        } catch (const std::exception& ex) {
            cameraResult.error = ex.what();
            const fs::path tempPath = outputPath.string() + ".tmp.png";
            if (fs::exists(tempPath)) {
                fs::remove(tempPath);
            }
            if (logger) {
                logger->error(snapshot.config.name + " save exception: " + cameraResult.error + " path=" + outputPath.string());
            }
        }

        result.cameras.push_back(cameraResult);
    }

    if (logger) {
        std::string message =
            "Capture " + zeroPad(captureIndex, 6) + " finished: " +
            std::to_string(result.successCount) + "/" + std::to_string(result.expectedCount) + " images saved";
        if (result.skippedCount > 0) {
            message += "; " + std::to_string(result.skippedCount) + " disabled skipped";
        }
        logger->info(message);
    }
    return result;
}

std::string saveFailureLabel(const CameraSaveResult& camera, bool includeError) {
    std::string label = camera.config.name.empty() ? camera.config.ip : camera.config.name;
    if (!camera.config.ip.empty() && camera.config.ip != label) {
        label += " " + camera.config.ip;
    }
    if (includeError && !camera.error.empty()) {
        label += ": " + camera.error;
    }
    return label.empty() ? "unknown camera" : label;
}

std::string disabledSuffix(const SaveResult& result) {
    return result.skippedCount > 0 ? "; " + std::to_string(result.skippedCount) + " disabled" : "";
}

std::string saveResultOverlayMessage(const SaveResult& result) {
    if (result.expectedCount == 0) {
        return "No active cameras to save";
    }

    if (result.successCount == result.expectedCount) {
        return "Saved " + std::to_string(result.successCount) + " images" + disabledSuffix(result);
    }

    std::vector<std::string> failedWithErrors;
    std::vector<std::string> failedLabels;
    for (const auto& camera : result.cameras) {
        if (!camera.ok && !camera.skipped) {
            failedWithErrors.push_back(saveFailureLabel(camera, true));
            failedLabels.push_back(saveFailureLabel(camera, false));
        }
    }

    std::ostringstream detailed;
    detailed << "Saved " << result.successCount << "/" << result.expectedCount << " images";
    if (!failedWithErrors.empty()) {
        detailed << "; failed: ";
        for (std::size_t i = 0; i < failedWithErrors.size(); ++i) {
            if (i != 0) {
                detailed << ", ";
            }
            detailed << failedWithErrors[i];
        }
    }
    detailed << disabledSuffix(result);

    const std::string detailedMessage = detailed.str();
    if (detailedMessage.size() <= 110) {
        return detailedMessage;
    }

    std::ostringstream compact;
    compact << "Saved " << result.successCount << "/" << result.expectedCount << " images";
    if (!failedLabels.empty()) {
        compact << "; failed: ";
        for (std::size_t i = 0; i < failedLabels.size(); ++i) {
            if (i != 0) {
                compact << ", ";
            }
            compact << failedLabels[i];
        }
        compact << " (see session.log)";
    }
    compact << disabledSuffix(result);

    const std::string compactMessage = compact.str();
    if (compactMessage.size() <= 110) {
        return compactMessage;
    }

    return "Saved " + std::to_string(result.successCount) + "/" + std::to_string(result.expectedCount) +
           " images; " + std::to_string(static_cast<int>(failedLabels.size())) + " failed (see session.log)";
}

class MainWindow : public QWidget {
public:
    MainWindow(
        std::vector<std::shared_ptr<CameraWorker>> cameras,
        fs::path saveDir,
        bool saveDirectoryReady,
        std::vector<std::string> startupMessages,
        std::shared_ptr<Logger> logger,
        bool startFullscreen)
        : cameras_(std::move(cameras)),
          saveDir_(std::move(saveDir)),
          saveDirectoryReady_(saveDirectoryReady),
          logger_(std::move(logger)),
          cameraFailureUntil_(cameras_.size(), Clock::time_point{}) {
        setWindowTitle("MultiCam Capture");
        setFocusPolicy(Qt::StrongFocus);
        resize(1600, 900);

        if (!startupMessages.empty()) {
            std::ostringstream oss;
            for (std::size_t i = 0; i < startupMessages.size(); ++i) {
                if (i != 0) {
                    oss << " | ";
                }
                oss << startupMessages[i];
                if (logger_) {
                    logger_->warn(startupMessages[i]);
                }
            }
            setOverlayMessage(oss.str(), QColor(225, 180, 50), 5000);
        } else {
            setOverlayMessage("Starting cameras", QColor(225, 180, 50), 2500);
        }

        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, [this]() { tick(); });
        timer_->start(33);

        if (startFullscreen) {
            showFullScreen();
        }
    }

    ~MainWindow() override {
        finishSaveThreadForShutdown();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.fillRect(rect(), QColor(18, 20, 24));

        const int barHeight = std::clamp(height() / 10, 74, 110);
        const QRect gridRect(0, 0, width(), std::max(0, height() - barHeight));
        const int cellWidth = gridRect.width() / 2;
        const int cellHeight = gridRect.height() / 2;

        for (std::size_t i = 0; i < cameras_.size() && i < 4; ++i) {
            const int row = static_cast<int>(i) / 2;
            const int col = static_cast<int>(i) % 2;
            QRect cell(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
            cell.adjust(8, 8, -8, -8);
            drawCameraCell(painter, cell, i);
        }

        drawBottomBar(painter, QRect(0, height() - barHeight, width(), barHeight));
        drawFlash(painter);
        drawOverlayMessage(painter);
        drawKeyboardFocusHint(painter);
        if (confirmExit_) {
            drawExitConfirmation(painter);
        }
    }

    void keyPressEvent(QKeyEvent* event) override {
        if (confirmExit_) {
            if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
                allowClose_ = true;
                close();
                return;
            }
            if (event->key() == Qt::Key_Escape) {
                confirmExit_ = false;
                setOverlayMessage("Returned to capture", QColor(70, 190, 100), 1200);
                update();
                return;
            }
            event->accept();
            return;
        }

        switch (event->key()) {
            case Qt::Key_Return:
            case Qt::Key_Enter:
            case Qt::Key_Space:
                requestCapture();
                break;
            case Qt::Key_R:
                requestReconnect();
                break;
            case Qt::Key_F:
                if (isFullScreen()) {
                    showNormal();
                } else {
                    showFullScreen();
                }
                break;
            case Qt::Key_Q:
            case Qt::Key_Escape:
                confirmExit_ = true;
                update();
                break;
            default:
                QWidget::keyPressEvent(event);
                break;
        }
    }

    void mousePressEvent(QMouseEvent* event) override {
        activateWindow();
        setFocus(Qt::MouseFocusReason);
        QWidget::mousePressEvent(event);
        update();
    }

    void focusInEvent(QFocusEvent* event) override {
        QWidget::focusInEvent(event);
        update();
    }

    void showEvent(QShowEvent* event) override {
        QWidget::showEvent(event);
        activateWindow();
        setFocus(Qt::ActiveWindowFocusReason);
        update();
    }

    void focusOutEvent(QFocusEvent* event) override {
        QWidget::focusOutEvent(event);
        update();
    }

    void changeEvent(QEvent* event) override {
        QWidget::changeEvent(event);
        if (event->type() == QEvent::ActivationChange) {
            update();
        }
    }

    void closeEvent(QCloseEvent* event) override {
        if (allowClose_) {
            event->accept();
            return;
        }
        confirmExit_ = true;
        event->ignore();
        update();
    }

private:
    void tick() {
        if (saving_ && saveState_ && saveState_->done.load(std::memory_order_acquire)) {
            if (saveThread_.joinable()) {
                saveThread_.join();
            }
            SaveResult result = std::move(saveState_->result);
            saveState_.reset();
            saving_ = false;
            if (result.expectedCount > 0 && result.successCount > 0) {
                if (result.successCount == result.expectedCount) {
                    ++completeSets_;
                } else {
                    ++partialSets_;
                }
            } else {
                ++failedSets_;
            }

            const auto now = Clock::now();
            for (std::size_t i = 0; i < result.cameras.size() && i < cameraFailureUntil_.size(); ++i) {
                if (!result.cameras[i].ok && !result.cameras[i].skipped) {
                    cameraFailureUntil_[i] = now + std::chrono::seconds(4);
                }
            }

            if (result.expectedCount > 0 && result.successCount == result.expectedCount) {
                setOverlayMessage(saveResultOverlayMessage(result), QColor(70, 190, 100), 1600);
                flashUntil_ = now + std::chrono::milliseconds(180);
            } else {
                setOverlayMessage(saveResultOverlayMessage(result), QColor(220, 60, 60), 3500);
            }
        }
        update();
    }

    void requestCapture() {
        if (saving_) {
            setOverlayMessage("Still saving. Please wait.", QColor(225, 180, 50), 1500);
            return;
        }
        if (!saveDirectoryReady_) {
            setOverlayMessage("Save folder is not available", QColor(220, 60, 60), 3500);
            if (logger_) {
                logger_->error("Capture blocked because save folder is not available: " + saveDir_.string());
            }
            return;
        }

        std::vector<CameraSnapshot> snapshots;
        snapshots.reserve(cameras_.size());
        for (const auto& camera : cameras_) {
            snapshots.push_back(camera->snapshot(true));
        }
        const int activeCount = static_cast<int>(std::count_if(
            snapshots.begin(),
            snapshots.end(),
            [](const CameraSnapshot& snapshot) {
                return snapshot.state != CameraState::Disabled && !snapshot.config.url.empty();
            }));
        if (activeCount == 0) {
            setOverlayMessage("No active cameras to save", QColor(220, 60, 60), 2500);
            if (logger_) {
                logger_->warn("Capture blocked because all camera slots are disabled");
            }
            return;
        }

        const int captureIndex = nextCaptureIndex_++;
        saving_ = true;
        setOverlayMessage("Saving...", QColor(225, 180, 50), 1000);
        saveState_ = std::make_shared<SaveTaskState>();
        std::shared_ptr<SaveTaskState> state = saveState_;
        saveThread_ = std::thread(
            [state, captureIndex, saveDir = saveDir_, snapshots = std::move(snapshots), logger = logger_]() mutable {
                try {
                    state->result = saveCaptureSet(captureIndex, saveDir, snapshots, logger);
                } catch (const std::exception& ex) {
                    SaveResult failedResult;
                    failedResult.captureIndex = captureIndex;
                    failedResult.cameras.reserve(snapshots.size());
                    for (const auto& snapshot : snapshots) {
                        CameraSaveResult cameraResult;
                        cameraResult.config = snapshot.config;
                        cameraResult.error = std::string("Save task exception: ") + ex.what();
                        failedResult.cameras.push_back(cameraResult);
                    }
                    if (logger) {
                        logger->error("Capture " + zeroPad(captureIndex, 6) + " save task exception: " + ex.what());
                    }
                    state->result = std::move(failedResult);
                } catch (...) {
                    SaveResult failedResult;
                    failedResult.captureIndex = captureIndex;
                    failedResult.cameras.reserve(snapshots.size());
                    for (const auto& snapshot : snapshots) {
                        CameraSaveResult cameraResult;
                        cameraResult.config = snapshot.config;
                        cameraResult.error = "Unknown save task exception";
                        failedResult.cameras.push_back(cameraResult);
                    }
                    if (logger) {
                        logger->error("Capture " + zeroPad(captureIndex, 6) + " unknown save task exception");
                    }
                    state->result = std::move(failedResult);
                }
                state->done.store(true, std::memory_order_release);
            });
    }

    void requestReconnect() {
        for (auto& camera : cameras_) {
            camera->requestReconnect();
        }
        setOverlayMessage("Reconnecting all cameras", QColor(225, 180, 50), 1800);
    }

    void setOverlayMessage(const std::string& message, QColor color, int milliseconds) {
        overlayMessage_ = message;
        overlayColor_ = color;
        overlayUntil_ = Clock::now() + std::chrono::milliseconds(milliseconds);
    }

    void finishSaveThreadForShutdown() {
        if (!saveThread_.joinable()) {
            return;
        }

        const auto deadline = Clock::now() + std::chrono::milliseconds(kSaveJoinTimeoutMs);
        while (Clock::now() < deadline) {
            if (saveState_ && saveState_->done.load(std::memory_order_acquire)) {
                saveThread_.join();
                saveState_.reset();
                saving_ = false;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (logger_) {
            logger_->warn("Detached save task after shutdown timeout; process will exit.");
        }
        saveThread_.detach();
        saveState_.reset();
        saving_ = false;
    }

    bool keyboardCaptureEnabled() const {
        return isActiveWindow() && hasFocus();
    }

    void drawCameraCell(QPainter& painter, const QRect& rect, std::size_t index) {
        const CameraSnapshot snapshot = cameras_[index]->snapshot(false);
        const bool recentSaveFailure = Clock::now() < cameraFailureUntil_[index];
        const QColor statusColor = colorForState(snapshot, recentSaveFailure);

        painter.fillRect(rect, QColor(34, 36, 42));

        if (!snapshot.previewImage.isNull()) {
            drawPreviewImage(painter, rect.adjusted(3, 3, -3, -3), snapshot.previewImage);
        } else {
            painter.setPen(QColor(210, 210, 210));
            painter.setFont(QFont("Sans Serif", 28, QFont::Bold));
            painter.drawText(rect, Qt::AlignCenter, "No Image");
        }

        QPen border(statusColor, 5);
        painter.setPen(border);
        painter.drawRect(rect.adjusted(2, 2, -2, -2));

        QRect labelRect = rect.adjusted(0, 0, 0, -rect.height() + 88);
        painter.fillRect(labelRect, QColor(0, 0, 0, 175));
        painter.setPen(Qt::white);
        painter.setFont(QFont("Sans Serif", 18, QFont::Bold));
        painter.drawText(
            labelRect.adjusted(14, 8, -14, -48),
            Qt::AlignLeft | Qt::AlignVCenter,
            QString::fromStdString(snapshot.config.name + "  " + snapshot.config.ip));

        std::ostringstream details;
        details << stateLabel(snapshot, recentSaveFailure);
        if (snapshot.frameWidth > 0 && snapshot.frameHeight > 0) {
            details << "  " << snapshot.frameWidth << "x" << snapshot.frameHeight;
        }
        if (snapshot.secondsSinceFrame >= 0.0) {
            details << "  last frame " << std::fixed << std::setprecision(1) << snapshot.secondsSinceFrame << "s ago";
        } else if (!snapshot.message.empty()) {
            details << "  " << snapshot.message;
        }

        painter.setPen(statusColor);
        painter.setFont(QFont("Sans Serif", 15, QFont::Bold));
        painter.drawText(
            labelRect.adjusted(14, 42, -14, -8),
            Qt::AlignLeft | Qt::AlignVCenter,
            QString::fromStdString(details.str()));
    }

    void drawPreviewImage(QPainter& painter, const QRect& rect, const QImage& image) {
        if (image.isNull() || rect.width() <= 0 || rect.height() <= 0) {
            return;
        }
        const double scale = std::min(
            static_cast<double>(rect.width()) / static_cast<double>(image.width()),
            static_cast<double>(rect.height()) / static_cast<double>(image.height()));
        const int targetWidth = std::max(1, static_cast<int>(image.width() * scale));
        const int targetHeight = std::max(1, static_cast<int>(image.height() * scale));
        QRect targetRect(
            rect.x() + (rect.width() - targetWidth) / 2,
            rect.y() + (rect.height() - targetHeight) / 2,
            targetWidth,
            targetHeight);
        painter.drawImage(targetRect, image);
    }

    void drawBottomBar(QPainter& painter, const QRect& rect) {
        painter.fillRect(rect, QColor(8, 10, 14));
        painter.setPen(QColor(80, 90, 105));
        painter.drawLine(rect.topLeft(), rect.topRight());

        painter.setPen(Qt::white);
        painter.setFont(QFont("Sans Serif", 18, QFont::Bold));
        painter.drawText(
            rect.adjusted(22, 8, -22, -rect.height() / 2),
            Qt::AlignLeft | Qt::AlignVCenter,
            "Enter / Space: Capture    R: Reconnect    F: Fullscreen    Q / Esc: Exit");

        std::string status = "Complete sets: " + std::to_string(completeSets_) +
                             "    Partial: " + std::to_string(partialSets_) +
                             "    Failed: " + std::to_string(failedSets_) + "    Save folder: " + saveDir_.string();
        if (saving_) {
            status += "    Saving...";
        }
        if (!saveDirectoryReady_) {
            status += "    Save folder unavailable";
        }
        painter.setPen(saveDirectoryReady_ ? QColor(210, 220, 230) : QColor(220, 60, 60));
        painter.setFont(QFont("Sans Serif", 14));
        painter.drawText(
            rect.adjusted(22, rect.height() / 2 - 4, -22, -8),
            Qt::AlignLeft | Qt::AlignVCenter,
            QString::fromStdString(status));
    }

    void drawOverlayMessage(QPainter& painter) {
        if (overlayMessage_.empty() || Clock::now() > overlayUntil_) {
            return;
        }
        const QString message = QString::fromStdString(overlayMessage_);
        const int boxWidth = std::max(1, std::min(std::max(1, width() - 40), 1100));
        const int fontSize = overlayMessage_.size() > 110 ? 20 : (overlayMessage_.size() > 70 ? 23 : 28);
        QFont font("Sans Serif", fontSize, QFont::Bold);
        const QRect textBounds = QFontMetrics(font).boundingRect(
            QRect(0, 0, std::max(1, boxWidth - 44), 1000),
            Qt::AlignCenter | Qt::TextWordWrap,
            message);
        const int maxBoxHeight = std::max(110, height() - 80);
        const int boxHeight = std::min(maxBoxHeight, std::max(110, textBounds.height() + 42));
        const QRect box((width() - boxWidth) / 2, (height() - boxHeight) / 2, boxWidth, boxHeight);
        painter.fillRect(box, QColor(0, 0, 0, 210));
        painter.setPen(QPen(overlayColor_, 4));
        painter.drawRect(box.adjusted(2, 2, -2, -2));
        painter.setPen(Qt::white);
        painter.setFont(font);
        painter.drawText(box.adjusted(22, 16, -22, -16), Qt::AlignCenter | Qt::TextWordWrap, message);
    }

    void drawKeyboardFocusHint(QPainter& painter) {
        if (keyboardCaptureEnabled() || confirmExit_) {
            return;
        }

        painter.fillRect(rect(), QColor(0, 0, 0, 115));

        const int boxWidth = std::clamp(width() - 80, 360, 860);
        const int boxHeight = 150;
        const QRect box((width() - boxWidth) / 2, (height() - boxHeight) / 2, boxWidth, boxHeight);
        painter.fillRect(box, QColor(10, 12, 16, 230));
        painter.setPen(QPen(QColor(225, 180, 50), 5));
        painter.drawRect(box.adjusted(2, 2, -2, -2));

        painter.setPen(Qt::white);
        const int fontSize = std::clamp(width() / 45, 18, 28);
        painter.setFont(QFont("Sans Serif", fontSize, QFont::Bold));
        painter.drawText(box.adjusted(24, 18, -24, -18), Qt::AlignCenter | Qt::TextWordWrap, "CLICK WINDOW TO ENABLE KEYBOARD CAPTURE");
    }

    void drawFlash(QPainter& painter) {
        if (Clock::now() > flashUntil_) {
            return;
        }
        painter.fillRect(rect(), QColor(255, 255, 255, 95));
    }

    void drawExitConfirmation(QPainter& painter) {
        painter.fillRect(rect(), QColor(0, 0, 0, 180));
        const QRect box(width() / 2 - 390, height() / 2 - 150, 780, 300);
        painter.fillRect(box, QColor(22, 24, 30));
        painter.setPen(QPen(QColor(225, 180, 50), 4));
        painter.drawRect(box.adjusted(2, 2, -2, -2));

        painter.setPen(Qt::white);
        painter.setFont(QFont("Sans Serif", 30, QFont::Bold));
        painter.drawText(box.adjusted(24, 24, -24, -210), Qt::AlignCenter, "Exit MultiCam Capture?");

        painter.setFont(QFont("Sans Serif", 16));
        const std::string detail =
            "Complete sets: " + std::to_string(completeSets_) + "\n" +
            "Partial sets: " + std::to_string(partialSets_) + "\n" +
            "Failed sets: " + std::to_string(failedSets_) + "\n" +
            "Save folder: " + saveDir_.string() + "\n\n" +
            "Enter: Exit    Esc: Return";
        painter.drawText(box.adjusted(34, 98, -34, -24), Qt::AlignCenter | Qt::TextWordWrap, QString::fromStdString(detail));
    }

    std::vector<std::shared_ptr<CameraWorker>> cameras_;
    fs::path saveDir_;
    bool saveDirectoryReady_ = false;
    std::shared_ptr<Logger> logger_;
    QTimer* timer_ = nullptr;
    std::thread saveThread_;
    std::shared_ptr<SaveTaskState> saveState_;
    bool saving_ = false;
    bool confirmExit_ = false;
    bool allowClose_ = false;
    int nextCaptureIndex_ = 0;
    int completeSets_ = 0;
    int partialSets_ = 0;
    int failedSets_ = 0;
    std::string overlayMessage_;
    QColor overlayColor_ = QColor(225, 180, 50);
    Clock::time_point overlayUntil_ = Clock::time_point{};
    Clock::time_point flashUntil_ = Clock::time_point{};
    std::vector<Clock::time_point> cameraFailureUntil_;
};

bool joinCameraWorkersWithTimeout(std::vector<std::shared_ptr<CameraWorker>>& workers, const std::shared_ptr<Logger>& logger) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(kWorkerJoinTimeoutMs);
    while (Clock::now() < deadline) {
        bool allJoined = true;
        for (auto& worker : workers) {
            if (worker && !worker->joinIfFinished()) {
                allJoined = false;
            }
        }
        if (allJoined) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    bool anyDetached = false;
    for (auto& worker : workers) {
        if (worker && !worker->joinIfFinished()) {
            worker->detachIfStillRunning();
            anyDetached = true;
        }
    }

    if (anyDetached && logger) {
        logger->warn("Detached camera worker(s) after shutdown timeout; process will exit.");
    }
    return anyDetached;
}

bool parsePositiveRowNumber(const std::string& value, int& rowNumber) {
    if (value.empty()) {
        return false;
    }
    for (unsigned char ch : value) {
        if (!std::isdigit(ch)) {
            return false;
        }
    }

    try {
        const long long parsed = std::stoll(value);
        if (parsed <= 0 || parsed > 1000000) {
            return false;
        }
        rowNumber = static_cast<int>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " -o NAME --row ROW [--config PATH] [--windowed]\n"
              << "\n"
              << "Required output options:\n"
              << "  -o, --output NAME  Output label appended to the dated session folder\n"
              << "  --row ROW          Positive row number appended as rowROW\n"
              << "\n"
              << "Keyboard controls:\n"
              << "  Enter / Space  Capture full-resolution images\n"
              << "  R              Reconnect all cameras\n"
              << "  F              Toggle fullscreen\n"
              << "  Q / Esc        Exit confirmation\n";
}

}  // namespace

int main(int argc, char** argv) {
    fs::path configPath = defaultConfigPath();
    bool startFullscreen = true;
    std::string outputName;
    int rowNumber = -1;
    bool hasOutput = false;
    bool hasRow = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "--windowed") {
            startFullscreen = false;
            continue;
        }
        if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc || std::string(argv[i + 1]).empty() || std::string(argv[i + 1]).rfind("-", 0) == 0) {
                std::cerr << arg << " requires an output name" << std::endl;
                printUsage(argv[0]);
                return 2;
            }
            outputName = argv[++i];
            hasOutput = !outputName.empty();
            continue;
        }
        if (arg == "--row") {
            if (i + 1 >= argc) {
                std::cerr << "--row requires a positive number" << std::endl;
                printUsage(argv[0]);
                return 2;
            }
            hasRow = parsePositiveRowNumber(argv[++i], rowNumber);
            if (!hasRow) {
                std::cerr << "--row must be a positive number" << std::endl;
                printUsage(argv[0]);
                return 2;
            }
            continue;
        }
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
            continue;
        }
        std::cerr << "Unknown argument: " << arg << std::endl;
        printUsage(argv[0]);
        return 2;
    }
    if (!hasOutput || !hasRow) {
        std::cerr << "-o/--output and --row are required" << std::endl;
        printUsage(argv[0]);
        return 2;
    }

    std::vector<std::string> startupMessages;
    AppConfig config;
    loadConfig(configPath, config, startupMessages);
    config.saveRoot = expandTilde(config.saveRoot);
    applyRuntimeRtspCredentials(config, startupMessages);

    fs::path saveDir;
    bool saveDirectoryReady = false;
    std::string saveDirError;
    const std::string sessionTimestamp = nowTimestampForFolder();
    const std::string outputLabel = makeOutputLabel(outputName, rowNumber);
    if (createUniqueSessionDirectory(fs::path(config.saveRoot), sessionTimestamp, outputLabel, saveDir, saveDirError)) {
        saveDirectoryReady = true;
    } else {
        startupMessages.push_back(saveDirError);
    }

    std::shared_ptr<Logger> logger = std::make_shared<Logger>(saveDirectoryReady ? saveDir / "session.log" : fs::path{});
    logger->info("MultiCam Capture starting");
    logger->info("Config path: " + configPath.string());
    logger->info("Save folder: " + saveDir.string());
    logger->info("Session timestamp: " + sessionTimestamp);
    logger->info("Output label: " + outputLabel);
    logger->info("Row: " + std::to_string(rowNumber));
    logger->info("Session log: " + (logger->path().empty() ? std::string("unavailable") : logger->path()));
    for (const auto& camera : config.cameras) {
        logger->info(camera.name + " ip=" + camera.ip + " url=" + redactedUrl(camera.url));
    }

    QApplication app(argc, argv);
    QApplication::setApplicationName("MultiCam Capture");

    std::vector<std::shared_ptr<CameraWorker>> workers;
    workers.reserve(config.cameras.size());
    for (const auto& cameraConfig : config.cameras) {
        workers.push_back(std::make_shared<CameraWorker>(cameraConfig, logger));
    }
    for (auto& worker : workers) {
        worker->start();
    }

    MainWindow window(workers, saveDir, saveDirectoryReady, startupMessages, logger, startFullscreen);
    if (!startFullscreen) {
        window.show();
    }
    const int exitCode = app.exec();

    logger->info("Stopping camera workers");
    for (auto& worker : workers) {
        worker->requestStop();
    }
    const bool detachedWorkers = joinCameraWorkersWithTimeout(workers, logger);
    if (detachedWorkers) {
        auto* leakedWorkers = new std::vector<std::shared_ptr<CameraWorker>>(std::move(workers));
        (void)leakedWorkers;
    }
    logger->info("MultiCam Capture stopped");
    return exitCode;
}
