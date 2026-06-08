#pragma once

#include <cstdlib>
#include <string>

inline std::string multicamRtspUser() {
    const char* user = std::getenv("MULTICAM_RTSP_USER");
    return (user != nullptr && *user != '\0') ? std::string(user) : std::string("admin");
}

inline std::string multicamRtspPassword() {
    const char* password = std::getenv("MULTICAM_RTSP_PASSWORD");
    return password != nullptr ? std::string(password) : std::string("t123456p");
}

inline bool multicamHasRtspCredentials() {
    return !multicamRtspUser().empty();
}

inline std::string multicamRtspCredentialSource() {
    const char* user = std::getenv("MULTICAM_RTSP_USER");
    return (user != nullptr && *user != '\0') ? std::string("environment") : std::string("built-in lab default");
}

inline std::string multicamRtspUrlForIp(const std::string& ip) {
    const std::string user = multicamRtspUser();
    const std::string auth = user.empty() ? "" : user + ":" + multicamRtspPassword() + "@";

    return "rtsp://" + auth + ip + ":554/h264Preview_01_main";
}

inline std::string multicamRedactUrlCredentials(const std::string& url) {
    const std::size_t schemeEnd = url.find("://");
    if (schemeEnd == std::string::npos) {
        return url;
    }

    const std::size_t authStart = schemeEnd + 3;
    const std::size_t atPos = url.find('@', authStart);
    const std::size_t pathStart = url.find('/', authStart);
    if (atPos == std::string::npos || (pathStart != std::string::npos && pathStart < atPos)) {
        return url;
    }

    return url.substr(0, authStart) + "***@" + url.substr(atPos + 1);
}
