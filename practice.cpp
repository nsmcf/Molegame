#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <cmath>

//将num限制在min和max之间
double limitValue(double num, double min, double max) {
    return std::min(max, std::max(num, min));
}

//判断point点是否在center为圆心radius为半径的圆中
bool isInsideCircle(cv::Point point, cv::Point center, int radius) {
    if(std::pow(point.x - center.x, 2) + std::pow(point.y - center.y, 2) <= std::pow(radius, 2)) return true;
    else return false;
}

//调取当前时间点
double currentTimeSeconds() {
    return cv::getTickCount() / cv::getTickFrequency();
}

int main() {
    // 创建一个窗口，等待按键后退出。
    cv::namedWindow("Practice");

    int width = 540;
    int height = 360;
    int radius = 50;
    
    bool inside = true;
    int cnt = 0;

    bool filled = false;
    cv::Point center(width / 2, height / 2);
    cv::Point blueCenter(width / 2, height / 2);

    const int font = cv::FONT_HERSHEY_SIMPLEX;  // 字体
    const double fontScale = 0.7;  // 字体缩放比例
    const double textThickness = 1;  // 文字线宽

    double startTime = currentTimeSeconds();
    double passTime = 0;  //记录循环时间

    bool paused = false;
    double pausedAt = startTime;
    double pausedTime = 0;

    // 创建图像并显示
    while(true) {
        cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 255, 255));
        cv::circle(image, center, radius, cv::Scalar(0, 0, 255), (filled == false) ? 3 : cv::FILLED);
        cv::circle(image, blueCenter, 4, cv::Scalar(255, 0, 0), cv::FILLED);
    
        if(!paused) {
            passTime = currentTimeSeconds() - startTime;
            pausedAt = currentTimeSeconds();
        }
        else pausedTime = currentTimeSeconds() - pausedAt;
        std::string passTimeText = cv::format("%.2f", passTime);

        bool nowInside = isInsideCircle(blueCenter, center, radius);
        std::string gameStatus = (passTime >= 10) ? "FINISHED" : "RUNNING";
        if(!paused) if(inside == false && nowInside == true && passTime < 10) cnt = cnt + 1;
        inside = nowInside;

        cv::putText(image, 
            "Radius: " + std::to_string(radius) + "  FILLED: " + std::to_string(filled) + "  isInsideCircle: " + std::to_string(inside), 
            cv::Point(20, 30), 
            font, 
            fontScale,  
            cv::Scalar(0, 0, 0), 
            textThickness
        );
        cv::putText(image, 
            "insideCount: " + std::to_string(cnt), 
            cv::Point(20, 50), 
            font, 
            fontScale, 
            cv::Scalar(0, 0, 0), 
            textThickness
        );
        cv::putText(image, 
            "passTime: " + passTimeText + "  " +gameStatus, 
            cv::Point(20, 70), 
            font, 
            fontScale, 
            cv::Scalar(0, 0, 0), 
            textThickness
        );
        cv::imshow("Practice", image);

        int key = cv::waitKey(10);
        if(key == 'a' || key == 'A') radius = (radius > 20) ? (radius - 10) : 20;
        if(key == 'd' || key == 'D') radius = (radius < 180) ? (radius + 10) : 180;

        if(key == 'j' || key == 'J') center.x = limitValue(center.x - 10, 0, width - 1);
        if(key == 'l' || key == 'L') center.x = limitValue(center.x + 10, 0, width - 1);
        if(key == 'i' || key == 'I') center.y = limitValue(center.y - 10, 0, height - 1);
        if(key == 'k' || key == 'K') center.y = limitValue(center.y + 10, 0, height - 1);

        if(key == ' ') filled = !filled;

        if((key == 'p' || key == 'P') && passTime < 10) {
            if(paused == true) {
                paused = !paused;
                startTime = startTime + pausedTime;
                pausedTime = 0;
            }
            else {
                paused = !paused;
            }
        };

        if(key == 'r' || key == 'R') {
            center.x = width / 2;
            center.y = height / 2;
            radius = 50;
            filled = false;
            inside = true;
            cnt = 0;
            startTime = currentTimeSeconds();
            pausedTime = 0;
        }
        if(key == 27) break;
    }
    return 0;
}