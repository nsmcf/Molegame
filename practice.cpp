#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <string>

// 取时间的工具已准备好：返回一个持续增长的毫秒数，用两个读数相减计算经过的时间。
double currentTimeMs() {
    return static_cast<double>(cv::getTickCount()) * 1000.0 / cv::getTickFrequency();
}

// 鼠标位置读取工具：OpenCV 收到鼠标事件时，更新我们传入的坐标。
// 这部分已准备好，本轮只需要使用 main 中 mouse.x 和 mouse.y 的值。
void recordMouse(int, int x, int y, int, void* userdata) {
    cv::Point* position = static_cast<cv::Point*>(userdata);
    position->x = x;
    position->y = y;
}

// 今日练习：判断鼠标点是否在圆内。只补全一处 TODO。
// 灰色圆周表示判定区域；空格控制黄色圆显示一秒，A/D 调整半径。
int main() {
    const std::string windowName = "MoleGame practice";
    int radius = 80;  // 放在循环外，保存按键修改后的值。
    int cnt = 0;    //进入圆内部的次数
    bool visible = false;  // true 表示显示，false 表示隐藏；同样需要跨循环保存。
    double startedAtMs = 0.0;  // 保存本次显示的开始时间，double 可以保存小数。
    const cv::Point center{320, 260};  // 圆心的 x、y 坐标。
    cv::Point mouse{-1, -1};  // -1 表示暂时还没有收到鼠标位置。
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, recordMouse, &mouse);

    for (;;) {
        // 上次完成的自动隐藏逻辑。
        if (visible == true) {
            const double nowMs = currentTimeMs();
            if (nowMs >= startedAtMs + 1000) visible = false;
        }

        bool inside = false;  // 每一轮根据最新坐标重新判断。
        if (mouse.x >= 0 && mouse.y >= 0) {
            const int dx = mouse.x - center.x;
            const int dy = mouse.y - center.y;
            // TODO：用 dx、dy 和 radius 判断是否在圆内，把比较结果赋给 inside。
            // 提示：圆内的点满足“水平距离的平方 + 垂直距离的平方 <= 半径的平方”。
            if (dx * dx + dy * dy <= radius * radius) inside = true;
        }

        // 每轮创建新画布，避免上一次画的圆留在图上。
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(29, 25, 24));
        // 上次完成的条件绘图：不画圆时，仍然显示新画布并处理按键。
        if (visible == true) {
            cv::circle(frame, center, radius, cv::Scalar(0, 200, 255), -1, cv::LINE_AA);
        }
        cv::circle(frame, center, radius, cv::Scalar(160, 160, 160), 2, cv::LINE_AA);
        cv::putText(frame, "Radius: " + std::to_string(radius), {20, 35},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(235, 235, 235), 1);
        cv::putText(frame, "A/D: size   Space: show for 1 second   Esc: exit", {20, 455},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(235, 235, 235), 1);

        std::string status = "Move mouse into the window";
        if (mouse.x >= 0 && mouse.y >= 0) {
            status = "OUTSIDE";
            if (inside) status = "INSIDE";
            cv::circle(frame, mouse, 5, cv::Scalar(80, 80, 255), -1, cv::LINE_AA);
            cv::putText(frame, "Mouse: " + std::to_string(mouse.x) + ", " + std::to_string(mouse.y),
                        {20, 95}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(235, 235, 235), 1);
        }
        cv::putText(frame, status, {20, 65}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(235, 235, 235), 1);

        cv::imshow(windowName, frame);

        const int key = cv::waitKeyEx(10);
        if (key == 27 || cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) break;

        // 上次完成的半径调整逻辑。
        if (key == 'a' || key == 'A') radius = std::max(20, radius - 10);
        if (key == 'd' || key == 'D') radius = std::min(180, radius + 10);

        // 上次完成的显示和重新计时逻辑。
        if (key == ' ') {
            startedAtMs = currentTimeMs();
            if (visible == false) visible = true;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
