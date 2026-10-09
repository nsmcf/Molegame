#include "vision/YoloHandTracker.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#ifdef RHYTHM_USE_TENSORRT_YOLO
#include <NvInfer.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#endif

namespace
{
#ifdef RHYTHM_USE_TENSORRT_YOLO
    class TensorRtLogger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* message) noexcept override
        {
            if (severity <= Severity::kWARNING) {
                std::cerr << "TensorRT: " << message << std::endl;
            }
        }
    };

    template<typename T>
    struct TensorRtDestroy
    {
        void operator()(T* ptr) const
        {
            delete ptr;
        }
    };

    void throwOnCudaError(cudaError_t status, const char* operation)
    {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + cudaGetErrorString(status)
            );
        }
    }

    size_t tensorElementCount(const nvinfer1::Dims& dims)
    {
        size_t count = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            count *= static_cast<size_t>(dims.d[i]);
        }
        return count;
    }

    size_t tensorElementSize(nvinfer1::DataType type)
    {
        switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(__half);
        case nvinfer1::DataType::kINT32:
            return sizeof(std::int32_t);
        case nvinfer1::DataType::kINT8:
            return sizeof(std::int8_t);
        case nvinfer1::DataType::kBOOL:
            return sizeof(bool);
        default:
            throw std::runtime_error("Unsupported TensorRT data type.");
        }
    }

    void throwOnCudaDriverError(CUresult status, const char* operation)
    {
        if (status == CUDA_SUCCESS) {
            return;
        }

        const char* errorName = nullptr;
        const char* errorDescription = nullptr;
        cuGetErrorName(status, &errorName);
        cuGetErrorString(status, &errorDescription);
        throw std::runtime_error(
            std::string(operation) + " failed: "
            + (errorName != nullptr ? errorName : "CUDA driver error")
            + " (" + (errorDescription != nullptr ? errorDescription : "unknown") + ")"
        );
    }

    void throwOnNvrtcError(nvrtcResult status, const char* operation)
    {
        if (status != NVRTC_SUCCESS) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + nvrtcGetErrorString(status)
            );
        }
    }

    std::string readEnvironmentVariable(const char* name)
    {
        char* value = nullptr;
        size_t valueLength = 0;
        if (_dupenv_s(&value, &valueLength, name) != 0 || value == nullptr) {
            return std::string();
        }

        const std::string result(value);
        std::free(value);
        return result;
    }

    struct TensorRtGpuHandResult
    {
        std::int32_t found = 0;
        std::int32_t classId = -1;
        float confidence = 0.0f;
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
    };

    static_assert(sizeof(TensorRtGpuHandResult) == 28);
    static_assert(std::is_standard_layout_v<TensorRtGpuHandResult>);

    struct TensorRtGpuHandBatch
    {
        std::int32_t count = 0;
        std::int32_t reserved = 0;
        std::array<TensorRtGpuHandResult, 2> hands{};
    };

    static_assert(sizeof(TensorRtGpuHandBatch) == 64);
    static_assert(std::is_standard_layout_v<TensorRtGpuHandBatch>);
    static_assert(offsetof(TensorRtGpuHandBatch, hands) == 8);

    struct TensorRtGpuTrackerState
    {
        std::int32_t hasLast = 0;
        std::int32_t missedFrames = 0;
        TensorRtGpuHandResult last;
    };

    static_assert(sizeof(TensorRtGpuTrackerState) == 36);

    constexpr const char* TENSOR_RT_GPU_KERNEL_SOURCE = R"CUDA(
#include <cuda_fp16.h>

struct GpuHandResult
{
    int found;
    int classId;
    float confidence;
    float x;
    float y;
    float width;
    float height;
};

struct GpuHandBatch
{
    int count;
    int reserved;
    GpuHandResult hands[2];
};

static_assert(sizeof(GpuHandResult) == 28, "Unexpected GPU hand result layout");
static_assert(sizeof(GpuHandBatch) == 64, "Unexpected GPU hand batch layout");

struct GpuTrackerState
{
    int hasLast;
    int missedFrames;
    GpuHandResult last;
};

__device__ __forceinline__ int clampInt(int value, int low, int high)
{
    return max(low, min(value, high));
}

__device__ __forceinline__ float readTensorValue(
    const void* data,
    int index,
    int isHalf)
{
    return isHalf
        ? __half2float(reinterpret_cast<const half*>(data)[index])
        : reinterpret_cast<const float*>(data)[index];
}

extern "C" __global__ void fusedPreprocessBgr(
    const unsigned char* source,
    int sourceWidth,
    int sourceHeight,
    int sourceStride,
    void* destination,
    int destinationWidth,
    int destinationHeight,
    int resizedWidth,
    int resizedHeight,
    int padLeft,
    int padTop,
    int destinationIsHalf)
{
    const int destinationIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int planeSize = destinationWidth * destinationHeight;
    if (destinationIndex >= planeSize) {
        return;
    }

    const int destinationX = destinationIndex % destinationWidth;
    const int destinationY = destinationIndex / destinationWidth;
    const int localX = destinationX - padLeft;
    const int localY = destinationY - padTop;

    float red = 114.0f / 255.0f;
    float green = red;
    float blue = red;

    if (localX >= 0 && localX < resizedWidth && localY >= 0 && localY < resizedHeight) {
        const float sourceX = (static_cast<float>(localX) + 0.5f)
            * static_cast<float>(sourceWidth) / static_cast<float>(resizedWidth) - 0.5f;
        const float sourceY = (static_cast<float>(localY) + 0.5f)
            * static_cast<float>(sourceHeight) / static_cast<float>(resizedHeight) - 0.5f;

        const int x0Raw = static_cast<int>(floorf(sourceX));
        const int y0Raw = static_cast<int>(floorf(sourceY));
        const int x0 = clampInt(x0Raw, 0, sourceWidth - 1);
        const int y0 = clampInt(y0Raw, 0, sourceHeight - 1);
        const int x1 = clampInt(x0Raw + 1, 0, sourceWidth - 1);
        const int y1 = clampInt(y0Raw + 1, 0, sourceHeight - 1);
        const float weightX = fminf(fmaxf(sourceX - static_cast<float>(x0Raw), 0.0f), 1.0f);
        const float weightY = fminf(fmaxf(sourceY - static_cast<float>(y0Raw), 0.0f), 1.0f);

        const unsigned char* pixel00 = source + y0 * sourceStride + x0 * 3;
        const unsigned char* pixel01 = source + y0 * sourceStride + x1 * 3;
        const unsigned char* pixel10 = source + y1 * sourceStride + x0 * 3;
        const unsigned char* pixel11 = source + y1 * sourceStride + x1 * 3;

        float channels[3];
        #pragma unroll
        for (int channel = 0; channel < 3; ++channel) {
            const float top = static_cast<float>(pixel00[channel])
                + weightX * (static_cast<float>(pixel01[channel]) - static_cast<float>(pixel00[channel]));
            const float bottom = static_cast<float>(pixel10[channel])
                + weightX * (static_cast<float>(pixel11[channel]) - static_cast<float>(pixel10[channel]));
            channels[channel] = (top + weightY * (bottom - top)) / 255.0f;
        }

        blue = channels[0];
        green = channels[1];
        red = channels[2];
    }

    if (destinationIsHalf) {
        half* output = reinterpret_cast<half*>(destination);
        output[destinationIndex] = __float2half_rn(red);
        output[planeSize + destinationIndex] = __float2half_rn(green);
        output[2 * planeSize + destinationIndex] = __float2half_rn(blue);
    }
    else {
        float* output = reinterpret_cast<float*>(destination);
        output[destinationIndex] = red;
        output[planeSize + destinationIndex] = green;
        output[2 * planeSize + destinationIndex] = blue;
    }
}

extern "C" __global__ void selectAndTrackHand(
    const void* tensorOutput,
    int outputIsHalf,
    int rows,
    int dimensions,
    int handClassId,
    float confidenceThreshold,
    int sourceWidth,
    int sourceHeight,
    float letterboxScale,
    int padLeft,
    int padTop,
    int mirrorX,
    float maxCenterJumpRatio,
    float maxBoxAreaChangeRatio,
    float smoothingAlpha,
    int holdFramesAfterMiss,
    GpuTrackerState* trackerState,
    GpuHandResult* result)
{
    __shared__ float scores[512];
    __shared__ int indices[512];

    const int thread = threadIdx.x;
    float score = -1.0f;
    int rowIndex = -1;
    if (thread < rows) {
        const int base = thread * dimensions;
        const float confidence = readTensorValue(tensorOutput, base + 4, outputIsHalf);
        const int classId = static_cast<int>(readTensorValue(tensorOutput, base + 5, outputIsHalf) + 0.5f);
        if (classId == handClassId && confidence >= confidenceThreshold) {
            score = confidence;
            rowIndex = thread;
        }
    }

    scores[thread] = score;
    indices[thread] = rowIndex;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread < offset && scores[thread + offset] > scores[thread]) {
            scores[thread] = scores[thread + offset];
            indices[thread] = indices[thread + offset];
        }
        __syncthreads();
    }

    if (thread != 0) {
        return;
    }

    GpuHandResult candidate{};
    candidate.classId = handClassId;
    const int bestRow = indices[0];
    if (bestRow >= 0) {
        const int base = bestRow * dimensions;
        const float x1 = readTensorValue(tensorOutput, base, outputIsHalf);
        const float y1 = readTensorValue(tensorOutput, base + 1, outputIsHalf);
        const float x2 = readTensorValue(tensorOutput, base + 2, outputIsHalf);
        const float y2 = readTensorValue(tensorOutput, base + 3, outputIsHalf);

        int left = __float2int_rn((x1 - static_cast<float>(padLeft)) / letterboxScale);
        int top = __float2int_rn((y1 - static_cast<float>(padTop)) / letterboxScale);
        int right = __float2int_rn((x2 - static_cast<float>(padLeft)) / letterboxScale);
        int bottom = __float2int_rn((y2 - static_cast<float>(padTop)) / letterboxScale);
        left = clampInt(left, 0, sourceWidth);
        top = clampInt(top, 0, sourceHeight);
        right = clampInt(right, 0, sourceWidth);
        bottom = clampInt(bottom, 0, sourceHeight);

        if (right > left && bottom > top) {
            candidate.found = 1;
            candidate.confidence = scores[0];
            candidate.x = static_cast<float>(mirrorX ? sourceWidth - right : left);
            candidate.y = static_cast<float>(top);
            candidate.width = static_cast<float>(right - left);
            candidate.height = static_cast<float>(bottom - top);
        }
    }

    if (candidate.found && trackerState->hasLast) {
        const float candidateCenterX = candidate.x + 0.5f * candidate.width;
        const float candidateCenterY = candidate.y + 0.5f * candidate.height;
        const float lastCenterX = trackerState->last.x + 0.5f * trackerState->last.width;
        const float lastCenterY = trackerState->last.y + 0.5f * trackerState->last.height;
        const float deltaX = candidateCenterX - lastCenterX;
        const float deltaY = candidateCenterY - lastCenterY;
        const float frameDiagonal = sqrtf(
            static_cast<float>(sourceWidth * sourceWidth + sourceHeight * sourceHeight)
        );
        const float centerJump = sqrtf(deltaX * deltaX + deltaY * deltaY);
        const float candidateArea = fmaxf(1.0f, candidate.width * candidate.height);
        const float lastArea = fmaxf(1.0f, trackerState->last.width * trackerState->last.height);
        const float areaChange = fmaxf(candidateArea / lastArea, lastArea / candidateArea);
        if (centerJump > maxCenterJumpRatio * frameDiagonal
            || areaChange > maxBoxAreaChangeRatio) {
            candidate.found = 0;
        }
    }

    if (!candidate.found) {
        if (trackerState->hasLast && trackerState->missedFrames < holdFramesAfterMiss) {
            ++trackerState->missedFrames;
            *result = trackerState->last;
            result->found = 1;
            return;
        }

        trackerState->hasLast = 0;
        trackerState->missedFrames = 0;
        *result = candidate;
        return;
    }

    if (trackerState->hasLast) {
        const float alpha = fminf(fmaxf(smoothingAlpha, 0.0f), 1.0f);
        const float oldWeight = 1.0f - alpha;
        candidate.x = roundf(alpha * candidate.x + oldWeight * trackerState->last.x);
        candidate.y = roundf(alpha * candidate.y + oldWeight * trackerState->last.y);
        candidate.width = roundf(alpha * candidate.width + oldWeight * trackerState->last.width);
        candidate.height = roundf(alpha * candidate.height + oldWeight * trackerState->last.height);
    }

    trackerState->hasLast = 1;
    trackerState->missedFrames = 0;
    trackerState->last = candidate;
    *result = candidate;
}

__device__ __forceinline__ GpuHandResult decodeHandRow(
    const void* tensorOutput,
    int outputIsHalf,
    int row,
    int dimensions,
    int handClassId,
    float confidence,
    int sourceWidth,
    int sourceHeight,
    float letterboxScale,
    int padLeft,
    int padTop,
    int mirrorX)
{
    GpuHandResult candidate{};
    candidate.classId = handClassId;
    const int base = row * dimensions;
    const float x1 = readTensorValue(tensorOutput, base, outputIsHalf);
    const float y1 = readTensorValue(tensorOutput, base + 1, outputIsHalf);
    const float x2 = readTensorValue(tensorOutput, base + 2, outputIsHalf);
    const float y2 = readTensorValue(tensorOutput, base + 3, outputIsHalf);
    if (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2)
        || !isfinite(letterboxScale) || letterboxScale <= 0.0f) {
        return candidate;
    }

    int left = __float2int_rn((x1 - static_cast<float>(padLeft)) / letterboxScale);
    int top = __float2int_rn((y1 - static_cast<float>(padTop)) / letterboxScale);
    int right = __float2int_rn((x2 - static_cast<float>(padLeft)) / letterboxScale);
    int bottom = __float2int_rn((y2 - static_cast<float>(padTop)) / letterboxScale);
    left = clampInt(left, 0, sourceWidth);
    top = clampInt(top, 0, sourceHeight);
    right = clampInt(right, 0, sourceWidth);
    bottom = clampInt(bottom, 0, sourceHeight);
    if (right <= left || bottom <= top) {
        return candidate;
    }

    candidate.found = 1;
    candidate.confidence = confidence;
    candidate.x = static_cast<float>(mirrorX ? sourceWidth - right : left);
    candidate.y = static_cast<float>(top);
    candidate.width = static_cast<float>(right - left);
    candidate.height = static_cast<float>(bottom - top);
    return candidate;
}

__device__ __forceinline__ float handBoxIou(
    const GpuHandResult& a,
    const GpuHandResult& b)
{
    const float intersectionLeft = fmaxf(a.x, b.x);
    const float intersectionTop = fmaxf(a.y, b.y);
    const float intersectionRight = fminf(a.x + a.width, b.x + b.width);
    const float intersectionBottom = fminf(a.y + a.height, b.y + b.height);
    const float intersectionWidth = fmaxf(0.0f, intersectionRight - intersectionLeft);
    const float intersectionHeight = fmaxf(0.0f, intersectionBottom - intersectionTop);
    const float intersection = intersectionWidth * intersectionHeight;
    const float unionArea = a.width * a.height + b.width * b.height - intersection;
    return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

extern "C" __global__ void selectTopHands(
    const void* tensorOutput,
    int outputIsHalf,
    int rows,
    int dimensions,
    int handClassId,
    float confidenceThreshold,
    float nmsThreshold,
    int sourceWidth,
    int sourceHeight,
    float letterboxScale,
    int padLeft,
    int padTop,
    int mirrorX,
    GpuHandBatch* result)
{
    __shared__ float scores[512];
    const int thread = threadIdx.x;
    float score = -1.0f;
    if (thread < rows) {
        const int base = thread * dimensions;
        const float confidence = readTensorValue(tensorOutput, base + 4, outputIsHalf);
        const float rawClassId = readTensorValue(tensorOutput, base + 5, outputIsHalf);
        if (isfinite(confidence) && isfinite(rawClassId)) {
            const int classId = static_cast<int>(rawClassId);
            if (classId == handClassId && confidence >= confidenceThreshold) {
                score = confidence;
            }
        }
    }
    scores[thread] = score;
    __syncthreads();

    if (thread != 0) {
        return;
    }

    GpuHandBatch selected{};
    const float maximumIou = fminf(fmaxf(nmsThreshold, 0.0f), 1.0f);
    for (int pick = 0; pick < 2; ++pick) {
        int bestRow = -1;
        float bestScore = -1.0f;
        GpuHandResult bestCandidate{};

        for (int row = 0; row < rows; ++row) {
            const float candidateScore = scores[row];
            if (candidateScore <= bestScore) {
                continue;
            }

            const GpuHandResult candidate = decodeHandRow(
                tensorOutput,
                outputIsHalf,
                row,
                dimensions,
                handClassId,
                candidateScore,
                sourceWidth,
                sourceHeight,
                letterboxScale,
                padLeft,
                padTop,
                mirrorX
            );
            if (!candidate.found) {
                scores[row] = -1.0f;
                continue;
            }

            bool suppressed = false;
            for (int selectedIndex = 0; selectedIndex < selected.count; ++selectedIndex) {
                if (handBoxIou(candidate, selected.hands[selectedIndex]) > maximumIou) {
                    suppressed = true;
                    break;
                }
            }
            if (suppressed) {
                continue;
            }

            bestRow = row;
            bestScore = candidateScore;
            bestCandidate = candidate;
        }

        if (bestRow < 0) {
            break;
        }
        selected.hands[selected.count++] = bestCandidate;
        scores[bestRow] = -1.0f;
    }

    *result = selected;
}
)CUDA";

    const char* tensorDataTypeName(nvinfer1::DataType type)
    {
        switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
        default:
            return "unknown";
        }
    }
#endif

    std::filesystem::path resolveProjectPath(const std::string& rawPath)
    {
        std::filesystem::path path(rawPath);

        if (path.empty()) {
            return path;
        }

        if (path.is_absolute()) {
            return path;
        }

        std::vector<std::filesystem::path> bases;
        bases.push_back(std::filesystem::current_path());
        bases.push_back(std::filesystem::current_path().parent_path());
        bases.push_back(std::filesystem::current_path().parent_path().parent_path());
        bases.push_back(std::filesystem::absolute(std::filesystem::path(__FILE__)).parent_path());

        for (const std::filesystem::path& base : bases) {
            std::filesystem::path candidate = base / path;
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }

        return path;
    }

    std::vector<std::string> loadClassNames(const std::string& classNamesPath)
    {
        std::vector<std::string> names;
        std::ifstream file(resolveProjectPath(classNamesPath));

        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            if (!line.empty()) {
                names.push_back(line);
            }
        }

        if (names.empty()) {
            names.push_back("hand");
            names.push_back("objects");
        }

        return names;
    }

    std::string labelForClassId(const std::vector<std::string>& classNames, int classId)
    {
        if (classId >= 0 && classId < static_cast<int>(classNames.size())) {
            return classNames[classId];
        }

        return "class " + std::to_string(classId);
    }

    cv::Rect clippedRect(int x, int y, int width, int height, const cv::Size& imageSize)
    {
        cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
        cv::Rect box(x, y, width, height);
        return box & imageRect;
    }

    struct LetterboxTransform
    {
        float scale = 1.0f;
        int padLeft = 0;
        int padTop = 0;
        cv::Size imageSize;
    };

    cv::Mat makeLetterboxedImage(
        const cv::Mat& image,
        const cv::Size& targetSize,
        LetterboxTransform& transform)
    {
        transform.imageSize = image.size();
        transform.scale = std::min(
            static_cast<float>(targetSize.width) / static_cast<float>(image.cols),
            static_cast<float>(targetSize.height) / static_cast<float>(image.rows)
        );

        const cv::Size resizedSize(
            static_cast<int>(std::lround(image.cols * transform.scale)),
            static_cast<int>(std::lround(image.rows * transform.scale))
        );
        transform.padLeft = (targetSize.width - resizedSize.width) / 2;
        transform.padTop = (targetSize.height - resizedSize.height) / 2;

        cv::Mat resized;
        cv::resize(image, resized, resizedSize, 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat letterboxed(targetSize, image.type(), cv::Scalar(114, 114, 114));
        resized.copyTo(cv::Mat(
            letterboxed,
            cv::Rect(transform.padLeft, transform.padTop, resizedSize.width, resizedSize.height)
        ));
        return letterboxed;
    }

    cv::Rect letterboxXyxyToImage(
        float x1,
        float y1,
        float x2,
        float y2,
        const LetterboxTransform& transform)
    {
        const int left = static_cast<int>(std::lround((x1 - transform.padLeft) / transform.scale));
        const int top = static_cast<int>(std::lround((y1 - transform.padTop) / transform.scale));
        const int right = static_cast<int>(std::lround((x2 - transform.padLeft) / transform.scale));
        const int bottom = static_cast<int>(std::lround((y2 - transform.padTop) / transform.scale));
        return clippedRect(left, top, right - left, bottom - top, transform.imageSize);
    }

    cv::Rect letterboxCenterBoxToImage(
        float centerX,
        float centerY,
        float width,
        float height,
        const LetterboxTransform& transform)
    {
        return letterboxXyxyToImage(
            centerX - 0.5f * width,
            centerY - 0.5f * height,
            centerX + 0.5f * width,
            centerY + 0.5f * height,
            transform
        );
    }

    void addBoxIfValid(
        std::vector<YoloHandBox>& boxes,
        const std::vector<std::string>& classNames,
        int classId,
        float confidence,
        const cv::Rect& box)
    {
        if (box.width <= 0 || box.height <= 0) {
            return;
        }

        YoloHandBox result;
        result.classId = classId;
        result.label = labelForClassId(classNames, classId);
        result.confidence = confidence;
        result.box = box;
        result.center = cv::Point(box.x + box.width / 2, box.y + box.height / 2);
        boxes.push_back(result);
    }
}

struct YoloHandTracker::Impl
{
    explicit Impl(const YoloHandTrackerConfig& trackerConfig)
        : config(trackerConfig)
    {
    }

    ~Impl()
    {
#ifdef RHYTHM_USE_TENSORRT_YOLO
        if (tensorRtGraphExec != nullptr) {
            cudaGraphExecDestroy(tensorRtGraphExec);
        }

        if (tensorRtGraph != nullptr) {
            cudaGraphDestroy(tensorRtGraph);
        }

        if (tensorRtInputDevice != nullptr) {
            cudaFree(tensorRtInputDevice);
        }

        if (tensorRtOutputDevice != nullptr) {
            cudaFree(tensorRtOutputDevice);
        }

        if (tensorRtInputHost != nullptr) {
            cudaFreeHost(tensorRtInputHost);
        }

        if (tensorRtOutputHost != nullptr) {
            cudaFreeHost(tensorRtOutputHost);
        }

        if (tensorRtRawInputDevice != nullptr) {
            cudaFree(tensorRtRawInputDevice);
        }

        if (tensorRtRawInputHost != nullptr) {
            cudaFreeHost(tensorRtRawInputHost);
        }

        if (tensorRtGpuResultHost != nullptr) {
            cudaFreeHost(tensorRtGpuResultHost);
        }

        if (tensorRtGpuMultiResultHost != nullptr) {
            cudaFreeHost(tensorRtGpuMultiResultHost);
        }

        if (tensorRtGpuTrackerStateDevice != nullptr) {
            cudaFree(tensorRtGpuTrackerStateDevice);
        }

        if (tensorRtGpuModule != nullptr) {
            cuModuleUnload(tensorRtGpuModule);
        }

        if (tensorRtStream != nullptr) {
            cudaStreamDestroy(tensorRtStream);
        }
#endif
    }

    YoloHandTrackerConfig config;
    std::vector<std::string> classNames;
    cv::dnn::Net openCvNet;
    bool initialized = false;
    std::string lastError;
    bool hasLastHand = false;
    int missedHandFrames = 0;
    YoloHandBox lastHand;
    cv::Point lastGameCenter;
    YoloHandPerformance lastPerformance;

#ifdef RHYTHM_USE_TENSORRT_YOLO
    TensorRtLogger tensorRtLogger;
    std::unique_ptr<nvinfer1::IRuntime, TensorRtDestroy<nvinfer1::IRuntime>> tensorRtRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDestroy<nvinfer1::ICudaEngine>> tensorRtEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtDestroy<nvinfer1::IExecutionContext>> tensorRtContext;
    std::string tensorRtInputName;
    std::string tensorRtOutputName;
    nvinfer1::Dims tensorRtInputDims{};
    nvinfer1::Dims tensorRtOutputDims{};
    nvinfer1::DataType tensorRtInputType = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType tensorRtOutputType = nvinfer1::DataType::kFLOAT;
    void* tensorRtInputDevice = nullptr;
    void* tensorRtOutputDevice = nullptr;
    void* tensorRtInputHost = nullptr;
    void* tensorRtOutputHost = nullptr;
    cudaStream_t tensorRtStream = nullptr;
    cudaGraph_t tensorRtGraph = nullptr;
    cudaGraphExec_t tensorRtGraphExec = nullptr;
    bool tensorRtGraphReady = false;
    bool tensorRtSpinWaitActive = false;
    cv::Mat tensorRtResizedBgr;
    std::vector<float> tensorRtOutputFloat;
    std::array<__half, 256> tensorRtNormalizedHalfLut{};
    LetterboxTransform tensorRtTransform;
    cv::Size tensorRtPreparedFrameSize;
    void* tensorRtRawInputDevice = nullptr;
    void* tensorRtRawInputHost = nullptr;
    size_t tensorRtRawInputBytes = 0;
    int tensorRtRawInputStride = 0;
    TensorRtGpuHandResult* tensorRtGpuResultHost = nullptr;
    TensorRtGpuHandResult* tensorRtGpuResultDeviceAlias = nullptr;
    TensorRtGpuHandBatch* tensorRtGpuMultiResultHost = nullptr;
    TensorRtGpuHandBatch* tensorRtGpuMultiResultDeviceAlias = nullptr;
    TensorRtGpuTrackerState* tensorRtGpuTrackerStateDevice = nullptr;
    CUmodule tensorRtGpuModule = nullptr;
    CUfunction tensorRtGpuPreprocessKernel = nullptr;
    CUfunction tensorRtGpuSelectKernel = nullptr;
    CUfunction tensorRtGpuSelectMultiKernel = nullptr;
    bool tensorRtGpuPipelineActive = false;
#endif

    std::vector<YoloHandBox> runOpenCvDnn(const cv::Mat& frame)
    {
        LetterboxTransform transform;
        const cv::Mat networkImage = makeLetterboxedImage(
            frame,
            config.networkInputSize,
            transform
        );
        cv::Mat blob = cv::dnn::blobFromImage(
            networkImage,
            1.0 / 255.0,
            config.networkInputSize,
            cv::Scalar(),
            true,
            false
        );

        openCvNet.setInput(blob);

        std::vector<cv::Mat> outputs;
        openCvNet.forward(outputs, openCvNet.getUnconnectedOutLayersNames());

        std::vector<YoloHandBox> rawBoxes;
        for (const cv::Mat& output : outputs) {
            parseOpenCvOutput(output, transform, rawBoxes);
        }

        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> confidences;
        for (const YoloHandBox& box : rawBoxes) {
            nmsBoxes.push_back(box.box);
            confidences.push_back(box.confidence);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(
            nmsBoxes,
            confidences,
            config.confidenceThreshold,
            config.nmsThreshold,
            indices
        );

        std::vector<YoloHandBox> boxes;
        boxes.reserve(indices.size());
        for (int index : indices) {
            boxes.push_back(rawBoxes[index]);
        }

        return boxes;
    }

    void parseOpenCvOutput(
        const cv::Mat& output,
        const LetterboxTransform& transform,
        std::vector<YoloHandBox>& boxes) const
    {
        cv::Mat dataMat = output;
        if (dataMat.depth() != CV_32F) {
            output.convertTo(dataMat, CV_32F);
        }

        const float* data = reinterpret_cast<const float*>(dataMat.data);
        const int classCount = static_cast<int>(classNames.size());

        if (dataMat.dims == 3) {
            const int dim1 = dataMat.size[1];
            const int dim2 = dataMat.size[2];

            if (dim1 == 300 && dim2 == 6) {
                parseYoloV10Rows(data, dim1, dim2, transform, boxes);
                return;
            }

            if (dim1 == 4 + classCount && dim2 > dim1) {
                parseUltralyticsChannelsFirst(data, dim2, dim1, transform, boxes);
                return;
            }

            parseRows(data, dim1, dim2, transform, boxes);
            return;
        }

        if (dataMat.dims == 2) {
            if (dataMat.rows == 4 + classCount && dataMat.cols > dataMat.rows) {
                parseUltralyticsChannelsFirst(data, dataMat.cols, dataMat.rows, transform, boxes);
                return;
            }

            parseRows(data, dataMat.rows, dataMat.cols, transform, boxes);
        }
    }

    void parseRows(
        const float* data,
        int rows,
        int dimensions,
        const LetterboxTransform& transform,
        std::vector<YoloHandBox>& boxes) const
    {
        const int classCount = static_cast<int>(classNames.size());

        for (int row = 0; row < rows; row++) {
            const float* rowData = data + row * dimensions;

            if (dimensions == 6) {
                parseYoloV10Row(rowData, transform, boxes);
                continue;
            }

            if (dimensions < 5 + classCount) {
                continue;
            }

            const float objectness = rowData[4];
            if (objectness < config.confidenceThreshold) {
                continue;
            }

            int classId = -1;
            float classScore = 0.0f;
            for (int i = 0; i < classCount; i++) {
                const float score = rowData[5 + i];
                if (score > classScore) {
                    classScore = score;
                    classId = i;
                }
            }

            const float confidence = objectness * classScore;
            if (confidence < config.confidenceThreshold) {
                continue;
            }

            const float cx = rowData[0];
            const float cy = rowData[1];
            const float width = rowData[2];
            const float height = rowData[3];

            cv::Rect box = letterboxCenterBoxToImage(cx, cy, width, height, transform);

            addBoxIfValid(boxes, classNames, classId, confidence, box);
        }
    }

    void parseUltralyticsChannelsFirst(
        const float* data,
        int rows,
        int dimensions,
        const LetterboxTransform& transform,
        std::vector<YoloHandBox>& boxes) const
    {
        const int classCount = static_cast<int>(classNames.size());

        for (int row = 0; row < rows; row++) {
            int classId = -1;
            float confidence = 0.0f;

            for (int i = 0; i < classCount; i++) {
                const float score = data[(4 + i) * rows + row];
                if (score > confidence) {
                    confidence = score;
                    classId = i;
                }
            }

            if (confidence < config.confidenceThreshold) {
                continue;
            }

            const float cx = data[row];
            const float cy = data[rows + row];
            const float width = data[2 * rows + row];
            const float height = data[3 * rows + row];

            cv::Rect box = letterboxCenterBoxToImage(cx, cy, width, height, transform);

            addBoxIfValid(boxes, classNames, classId, confidence, box);
        }
    }

    void parseYoloV10Rows(
        const float* data,
        int rows,
        int dimensions,
        const LetterboxTransform& transform,
        std::vector<YoloHandBox>& boxes) const
    {
        for (int row = 0; row < rows; row++) {
            parseYoloV10Row(data + row * dimensions, transform, boxes);
        }
    }

    void parseYoloV10Row(
        const float* rowData,
        const LetterboxTransform& transform,
        std::vector<YoloHandBox>& boxes) const
    {
        const float confidence = rowData[4];
        if (confidence < config.confidenceThreshold) {
            return;
        }

        const int classId = static_cast<int>(rowData[5]);

        cv::Rect box = letterboxXyxyToImage(
            rowData[0],
            rowData[1],
            rowData[2],
            rowData[3],
            transform
        );
        addBoxIfValid(boxes, classNames, classId, confidence, box);
    }

#ifdef RHYTHM_USE_TENSORRT_YOLO
    void releaseTensorRtGpuPipeline() noexcept
    {
        tensorRtGpuPipelineActive = false;

        if (tensorRtRawInputDevice != nullptr) {
            cudaFree(tensorRtRawInputDevice);
            tensorRtRawInputDevice = nullptr;
        }
        if (tensorRtRawInputHost != nullptr) {
            cudaFreeHost(tensorRtRawInputHost);
            tensorRtRawInputHost = nullptr;
        }
        if (tensorRtGpuResultHost != nullptr) {
            cudaFreeHost(tensorRtGpuResultHost);
            tensorRtGpuResultHost = nullptr;
            tensorRtGpuResultDeviceAlias = nullptr;
        }
        if (tensorRtGpuMultiResultHost != nullptr) {
            cudaFreeHost(tensorRtGpuMultiResultHost);
            tensorRtGpuMultiResultHost = nullptr;
            tensorRtGpuMultiResultDeviceAlias = nullptr;
        }
        if (tensorRtGpuTrackerStateDevice != nullptr) {
            cudaFree(tensorRtGpuTrackerStateDevice);
            tensorRtGpuTrackerStateDevice = nullptr;
        }
        if (tensorRtGpuModule != nullptr) {
            cuModuleUnload(tensorRtGpuModule);
            tensorRtGpuModule = nullptr;
            tensorRtGpuPreprocessKernel = nullptr;
            tensorRtGpuSelectKernel = nullptr;
            tensorRtGpuSelectMultiKernel = nullptr;
        }

        tensorRtRawInputBytes = 0;
        tensorRtRawInputStride = 0;
    }

    bool initializeTensorRtGpuPipeline()
    {
        if (!config.useGpuPipelineForTensorRt) {
            return false;
        }
        if (tensorRtInputType != nvinfer1::DataType::kHALF
            && tensorRtInputType != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("GPU preprocessing requires an FP16 or FP32 TensorRT input.");
        }
        if (tensorRtOutputType != nvinfer1::DataType::kHALF
            && tensorRtOutputType != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("GPU postprocessing requires an FP16 or FP32 TensorRT output.");
        }
        if (tensorRtOutputDims.d[2] != 6 || tensorRtOutputDims.d[1] > 512) {
            throw std::runtime_error("GPU postprocessing currently requires a YOLO 300x6-style output.");
        }
        if (config.cameraFrameSize.width <= 0 || config.cameraFrameSize.height <= 0) {
            throw std::runtime_error("GPU preprocessing requires a fixed camera frame size.");
        }

        int deviceIndex = 0;
        throwOnCudaError(cudaGetDevice(&deviceIndex), "cudaGetDevice");
        cudaDeviceProp deviceProperties{};
        throwOnCudaError(
            cudaGetDeviceProperties(&deviceProperties, deviceIndex),
            "cudaGetDeviceProperties"
        );

        nvrtcProgram program = nullptr;
        throwOnNvrtcError(
            nvrtcCreateProgram(
                &program,
                TENSOR_RT_GPU_KERNEL_SOURCE,
                "RhythmTensorRtGpuPipeline.cu",
                0,
                nullptr,
                nullptr
            ),
            "nvrtcCreateProgram"
        );

        try {
            const std::string architecture = "--gpu-architecture=compute_"
                + std::to_string(deviceProperties.major)
                + std::to_string(deviceProperties.minor);

            std::string cudaRootEnvironment = readEnvironmentVariable("CUDA_PATH");
            if (cudaRootEnvironment.empty()) {
                cudaRootEnvironment = readEnvironmentVariable("CUDA_PATH_V11_6");
            }
            const std::filesystem::path cudaRoot = !cudaRootEnvironment.empty()
                ? std::filesystem::path(cudaRootEnvironment)
                : std::filesystem::path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6");
            const std::string includeOption = "--include-path="
                + (cudaRoot / "include").generic_string();

            const std::array<const char*, 4> compileOptions = {
                architecture.c_str(),
                includeOption.c_str(),
                "--std=c++14",
                "--use_fast_math"
            };
            const nvrtcResult compileStatus = nvrtcCompileProgram(
                program,
                static_cast<int>(compileOptions.size()),
                compileOptions.data()
            );

            size_t logSize = 0;
            throwOnNvrtcError(nvrtcGetProgramLogSize(program, &logSize), "nvrtcGetProgramLogSize");
            std::string compileLog(logSize, '\0');
            if (logSize > 1) {
                throwOnNvrtcError(
                    nvrtcGetProgramLog(program, compileLog.data()),
                    "nvrtcGetProgramLog"
                );
            }
            if (compileStatus != NVRTC_SUCCESS) {
                throw std::runtime_error(
                    std::string("NVRTC GPU pipeline compilation failed: ")
                    + nvrtcGetErrorString(compileStatus)
                    + "\n" + compileLog
                );
            }

            size_t ptxSize = 0;
            throwOnNvrtcError(nvrtcGetPTXSize(program, &ptxSize), "nvrtcGetPTXSize");
            std::vector<char> ptx(ptxSize);
            throwOnNvrtcError(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
            throwOnNvrtcError(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");

            throwOnCudaDriverError(cuInit(0), "cuInit");
            CUcontext currentContext = nullptr;
            throwOnCudaDriverError(cuCtxGetCurrent(&currentContext), "cuCtxGetCurrent");
            if (currentContext == nullptr) {
                throw std::runtime_error("No current CUDA context is available for the GPU pipeline.");
            }

            throwOnCudaDriverError(
                cuModuleLoadData(&tensorRtGpuModule, ptx.data()),
                "cuModuleLoadData"
            );
            throwOnCudaDriverError(
                cuModuleGetFunction(
                    &tensorRtGpuPreprocessKernel,
                    tensorRtGpuModule,
                    "fusedPreprocessBgr"
                ),
                "cuModuleGetFunction(fusedPreprocessBgr)"
            );
            throwOnCudaDriverError(
                cuModuleGetFunction(
                    &tensorRtGpuSelectKernel,
                    tensorRtGpuModule,
                    "selectAndTrackHand"
                ),
                "cuModuleGetFunction(selectAndTrackHand)"
            );
            throwOnCudaDriverError(
                cuModuleGetFunction(
                    &tensorRtGpuSelectMultiKernel,
                    tensorRtGpuModule,
                    "selectTopHands"
                ),
                "cuModuleGetFunction(selectTopHands)"
            );

            tensorRtRawInputStride = config.cameraFrameSize.width * 3;
            tensorRtRawInputBytes = static_cast<size_t>(tensorRtRawInputStride)
                * static_cast<size_t>(config.cameraFrameSize.height);
            throwOnCudaError(
                cudaHostAlloc(
                    &tensorRtRawInputHost,
                    tensorRtRawInputBytes,
                    cudaHostAllocDefault
                ),
                "cudaHostAlloc(raw camera input)"
            );
            throwOnCudaError(
                cudaMalloc(&tensorRtRawInputDevice, tensorRtRawInputBytes),
                "cudaMalloc(raw camera input)"
            );
            throwOnCudaError(
                cudaHostAlloc(
                    reinterpret_cast<void**>(&tensorRtGpuResultHost),
                    sizeof(TensorRtGpuHandResult),
                    cudaHostAllocMapped
                ),
                "cudaHostAlloc(mapped result)"
            );
            throwOnCudaError(
                cudaHostGetDevicePointer(
                    reinterpret_cast<void**>(&tensorRtGpuResultDeviceAlias),
                    tensorRtGpuResultHost,
                    0
                ),
                "cudaHostGetDevicePointer(mapped result)"
            );
            throwOnCudaError(
                cudaHostAlloc(
                    reinterpret_cast<void**>(&tensorRtGpuMultiResultHost),
                    sizeof(TensorRtGpuHandBatch),
                    cudaHostAllocMapped
                ),
                "cudaHostAlloc(mapped multi-hand result)"
            );
            throwOnCudaError(
                cudaHostGetDevicePointer(
                    reinterpret_cast<void**>(&tensorRtGpuMultiResultDeviceAlias),
                    tensorRtGpuMultiResultHost,
                    0
                ),
                "cudaHostGetDevicePointer(mapped multi-hand result)"
            );
            throwOnCudaError(
                cudaMalloc(
                    reinterpret_cast<void**>(&tensorRtGpuTrackerStateDevice),
                    sizeof(TensorRtGpuTrackerState)
                ),
                "cudaMalloc(GPU tracker state)"
            );

            std::memset(tensorRtRawInputHost, 0, tensorRtRawInputBytes);
            *tensorRtGpuResultHost = TensorRtGpuHandResult();
            *tensorRtGpuMultiResultHost = TensorRtGpuHandBatch();
            throwOnCudaError(
                cudaMemsetAsync(
                    tensorRtGpuTrackerStateDevice,
                    0,
                    sizeof(TensorRtGpuTrackerState),
                    tensorRtStream
                ),
                "cudaMemsetAsync(GPU tracker state)"
            );
            throwOnCudaError(cudaStreamSynchronize(tensorRtStream), "cudaStreamSynchronize");

            const int inputWidth = tensorRtInputDims.d[3];
            const int inputHeight = tensorRtInputDims.d[2];
            tensorRtTransform.imageSize = config.cameraFrameSize;
            tensorRtTransform.scale = std::min(
                static_cast<float>(inputWidth) / static_cast<float>(config.cameraFrameSize.width),
                static_cast<float>(inputHeight) / static_cast<float>(config.cameraFrameSize.height)
            );
            const int resizedWidth = static_cast<int>(std::lround(
                config.cameraFrameSize.width * tensorRtTransform.scale
            ));
            const int resizedHeight = static_cast<int>(std::lround(
                config.cameraFrameSize.height * tensorRtTransform.scale
            ));
            tensorRtTransform.padLeft = (inputWidth - resizedWidth) / 2;
            tensorRtTransform.padTop = (inputHeight - resizedHeight) / 2;
            tensorRtPreparedFrameSize = config.cameraFrameSize;
            tensorRtGpuPipelineActive = true;
            return true;
        }
        catch (...) {
            if (program != nullptr) {
                nvrtcDestroyProgram(&program);
            }
            releaseTensorRtGpuPipeline();
            throw;
        }
    }

    bool initializeTensorRt(const std::filesystem::path& enginePath)
    {
        const unsigned int requestedDeviceFlags = cudaDeviceMapHost
            | (config.useSpinWaitForTensorRt ? cudaDeviceScheduleSpin : cudaDeviceScheduleAuto);
        const cudaError_t deviceFlagsStatus = cudaSetDeviceFlags(requestedDeviceFlags);
        if (deviceFlagsStatus == cudaSuccess) {
            tensorRtSpinWaitActive = config.useSpinWaitForTensorRt;
        }
        else if (config.useGpuPipelineForTensorRt || config.useSpinWaitForTensorRt) {
            std::cerr << "Requested CUDA device scheduling/mapped-memory flags are unavailable: "
                      << cudaGetErrorString(deviceFlagsStatus)
                      << std::endl;
        }

        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile) {
            lastError = "TensorRT engine not found: " + enginePath.string();
            return false;
        }

        engineFile.seekg(0, std::ios::end);
        const std::streamsize engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);

        std::vector<char> engineData(static_cast<size_t>(engineSize));
        if (!engineFile.read(engineData.data(), engineSize)) {
            lastError = "Failed to read TensorRT engine: " + enginePath.string();
            return false;
        }

        tensorRtRuntime.reset(nvinfer1::createInferRuntime(tensorRtLogger));
        if (!tensorRtRuntime) {
            lastError = "Failed to create TensorRT runtime.";
            return false;
        }

        tensorRtEngine.reset(tensorRtRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!tensorRtEngine) {
            lastError = "Failed to deserialize TensorRT engine.";
            return false;
        }

        tensorRtContext.reset(tensorRtEngine->createExecutionContext());
        if (!tensorRtContext) {
            lastError = "Failed to create TensorRT execution context.";
            return false;
        }

        if (tensorRtEngine->getNbIOTensors() < 2) {
            lastError = "TensorRT engine must have at least one input and one output.";
            return false;
        }

        tensorRtInputName = tensorRtEngine->getIOTensorName(0);
        tensorRtOutputName = tensorRtEngine->getIOTensorName(1);
        tensorRtInputDims = tensorRtEngine->getTensorShape(tensorRtInputName.c_str());
        tensorRtOutputDims = tensorRtEngine->getTensorShape(tensorRtOutputName.c_str());
        tensorRtInputType = tensorRtEngine->getTensorDataType(tensorRtInputName.c_str());
        tensorRtOutputType = tensorRtEngine->getTensorDataType(tensorRtOutputName.c_str());

        if (tensorRtInputType == nvinfer1::DataType::kHALF) {
            for (size_t value = 0; value < tensorRtNormalizedHalfLut.size(); ++value) {
                tensorRtNormalizedHalfLut[value] = __float2half(
                    static_cast<float>(value) / 255.0f
                );
            }
        }

        if (tensorRtInputDims.nbDims != 4 || tensorRtOutputDims.nbDims != 3) {
            lastError = "Unexpected TensorRT tensor dimensions.";
            return false;
        }

        const size_t inputElements = tensorElementCount(tensorRtInputDims);
        const size_t outputElements = tensorElementCount(tensorRtOutputDims);
        tensorRtOutputFloat.resize(outputElements);
        const size_t inputBytes = inputElements * tensorElementSize(tensorRtInputType);
        const size_t outputBytes = outputElements * tensorElementSize(tensorRtOutputType);

        try {
            int leastStreamPriority = 0;
            int greatestStreamPriority = 0;
            throwOnCudaError(
                cudaDeviceGetStreamPriorityRange(&leastStreamPriority, &greatestStreamPriority),
                "cudaDeviceGetStreamPriorityRange"
            );
            throwOnCudaError(
                cudaStreamCreateWithPriority(
                    &tensorRtStream,
                    cudaStreamNonBlocking,
                    greatestStreamPriority
                ),
                "cudaStreamCreateWithPriority"
            );
            throwOnCudaError(
                cudaMalloc(&tensorRtInputDevice, inputBytes),
                "cudaMalloc(input)"
            );
            throwOnCudaError(
                cudaMalloc(&tensorRtOutputDevice, outputBytes),
                "cudaMalloc(output)"
            );
            throwOnCudaError(
                cudaHostAlloc(&tensorRtInputHost, inputBytes, cudaHostAllocDefault),
                "cudaHostAlloc(input)"
            );
            throwOnCudaError(
                cudaHostAlloc(&tensorRtOutputHost, outputBytes, cudaHostAllocDefault),
                "cudaHostAlloc(output)"
            );
        }
        catch (const std::exception& e) {
            lastError = e.what();
            return false;
        }

        if (!tensorRtContext->setTensorAddress(tensorRtInputName.c_str(), tensorRtInputDevice)) {
            lastError = "Failed to bind TensorRT input tensor.";
            return false;
        }

        if (!tensorRtContext->setTensorAddress(tensorRtOutputName.c_str(), tensorRtOutputDevice)) {
            lastError = "Failed to bind TensorRT output tensor.";
            return false;
        }

        config.networkInputSize = cv::Size(tensorRtInputDims.d[3], tensorRtInputDims.d[2]);
        fillTensorRtInputHost(114.0f / 255.0f);

        if (config.useGpuPipelineForTensorRt) {
            try {
                initializeTensorRtGpuPipeline();
            }
            catch (const std::exception& e) {
                std::cerr << "TensorRT GPU pre/post pipeline unavailable; using CPU pipeline: "
                          << e.what()
                          << std::endl;
                releaseTensorRtGpuPipeline();
            }
        }

        if (config.useCudaGraphForTensorRt) {
            tensorRtGraphReady = initializeTensorRtCudaGraph();
        }

        std::cout << "TensorRT engine ready: input "
                  << tensorRtInputDims.d[1] << "x" << tensorRtInputDims.d[2] << "x" << tensorRtInputDims.d[3]
                  << " " << tensorDataTypeName(tensorRtInputType)
                  << ", output " << tensorRtOutputDims.d[1] << "x" << tensorRtOutputDims.d[2]
                  << " " << tensorDataTypeName(tensorRtOutputType)
                  << ", CUDA graph " << (tensorRtGraphReady ? "enabled" : "disabled")
                  << ", aux streams " << tensorRtEngine->getNbAuxStreams()
                  << ", spin wait " << (tensorRtSpinWaitActive ? "enabled" : "disabled")
                  << ", GPU pre/post " << (tensorRtGpuPipelineActive ? "enabled" : "disabled")
                  << ", transfer " << (tensorRtGpuPipelineActive
                      ? (config.multiHandTracking
                          ? "one raw H2D + mapped 64-byte two-hand result"
                          : "one raw H2D + mapped 28-byte result")
                      : "tensor H2D + tensor D2H")
                  << ", engine=" << enginePath.filename().string()
                  << std::endl;

        return true;
    }

    void fillTensorRtInputHost(float value)
    {
        const size_t inputElements = tensorElementCount(tensorRtInputDims);
        if (tensorRtInputType == nvinfer1::DataType::kFLOAT) {
            float* input = static_cast<float*>(tensorRtInputHost);
            std::fill(input, input + inputElements, value);
            return;
        }

        if (tensorRtInputType == nvinfer1::DataType::kHALF) {
            __half* input = static_cast<__half*>(tensorRtInputHost);
            const __half halfValue = __float2half(value);
            std::fill(input, input + inputElements, halfValue);
            return;
        }

        throw std::runtime_error("Only FP32/FP16 TensorRT inputs are supported.");
    }

    void launchTensorRtGpuPreprocess()
    {
        const int sourceWidth = config.cameraFrameSize.width;
        const int sourceHeight = config.cameraFrameSize.height;
        const int destinationWidth = tensorRtInputDims.d[3];
        const int destinationHeight = tensorRtInputDims.d[2];
        const int resizedWidth = static_cast<int>(std::lround(
            static_cast<float>(sourceWidth) * tensorRtTransform.scale
        ));
        const int resizedHeight = static_cast<int>(std::lround(
            static_cast<float>(sourceHeight) * tensorRtTransform.scale
        ));
        const int destinationIsHalf = tensorRtInputType == nvinfer1::DataType::kHALF ? 1 : 0;

        void* arguments[] = {
            &tensorRtRawInputDevice,
            const_cast<int*>(&sourceWidth),
            const_cast<int*>(&sourceHeight),
            &tensorRtRawInputStride,
            &tensorRtInputDevice,
            const_cast<int*>(&destinationWidth),
            const_cast<int*>(&destinationHeight),
            const_cast<int*>(&resizedWidth),
            const_cast<int*>(&resizedHeight),
            &tensorRtTransform.padLeft,
            &tensorRtTransform.padTop,
            const_cast<int*>(&destinationIsHalf)
        };

        constexpr unsigned int blockSize = 256;
        const unsigned int elementCount = static_cast<unsigned int>(
            destinationWidth * destinationHeight
        );
        const unsigned int gridSize = (elementCount + blockSize - 1) / blockSize;
        throwOnCudaDriverError(
            cuLaunchKernel(
                tensorRtGpuPreprocessKernel,
                gridSize,
                1,
                1,
                blockSize,
                1,
                1,
                0,
                reinterpret_cast<CUstream>(tensorRtStream),
                arguments,
                nullptr
            ),
            "cuLaunchKernel(fusedPreprocessBgr)"
        );
    }

    void launchTensorRtGpuSelect()
    {
        const int outputIsHalf = tensorRtOutputType == nvinfer1::DataType::kHALF ? 1 : 0;
        const int rows = tensorRtOutputDims.d[1];
        const int dimensions = tensorRtOutputDims.d[2];
        const int sourceWidth = config.cameraFrameSize.width;
        const int sourceHeight = config.cameraFrameSize.height;
        const int mirrorX = config.mirrorX ? 1 : 0;

        if (config.multiHandTracking) {
            void* multiArguments[] = {
                &tensorRtOutputDevice,
                const_cast<int*>(&outputIsHalf),
                const_cast<int*>(&rows),
                const_cast<int*>(&dimensions),
                &config.handClassId,
                &config.confidenceThreshold,
                &config.nmsThreshold,
                const_cast<int*>(&sourceWidth),
                const_cast<int*>(&sourceHeight),
                &tensorRtTransform.scale,
                &tensorRtTransform.padLeft,
                &tensorRtTransform.padTop,
                const_cast<int*>(&mirrorX),
                &tensorRtGpuMultiResultDeviceAlias
            };

            throwOnCudaDriverError(
                cuLaunchKernel(
                    tensorRtGpuSelectMultiKernel,
                    1,
                    1,
                    1,
                    512,
                    1,
                    1,
                    0,
                    reinterpret_cast<CUstream>(tensorRtStream),
                    multiArguments,
                    nullptr
                ),
                "cuLaunchKernel(selectTopHands)"
            );
            return;
        }

        void* arguments[] = {
            &tensorRtOutputDevice,
            const_cast<int*>(&outputIsHalf),
            const_cast<int*>(&rows),
            const_cast<int*>(&dimensions),
            &config.handClassId,
            &config.confidenceThreshold,
            const_cast<int*>(&sourceWidth),
            const_cast<int*>(&sourceHeight),
            &tensorRtTransform.scale,
            &tensorRtTransform.padLeft,
            &tensorRtTransform.padTop,
            const_cast<int*>(&mirrorX),
            &config.maxCenterJumpRatio,
            &config.maxBoxAreaChangeRatio,
            &config.smoothingAlpha,
            &config.holdFramesAfterMiss,
            &tensorRtGpuTrackerStateDevice,
            &tensorRtGpuResultDeviceAlias
        };

        throwOnCudaDriverError(
            cuLaunchKernel(
                tensorRtGpuSelectKernel,
                1,
                1,
                1,
                512,
                1,
                1,
                0,
                reinterpret_cast<CUstream>(tensorRtStream),
                arguments,
                nullptr
            ),
            "cuLaunchKernel(selectAndTrackHand)"
        );
    }

    void enqueueTensorRtCommands()
    {
        if (tensorRtGpuPipelineActive) {
            throwOnCudaError(
                cudaMemcpyAsync(
                    tensorRtRawInputDevice,
                    tensorRtRawInputHost,
                    tensorRtRawInputBytes,
                    cudaMemcpyHostToDevice,
                    tensorRtStream
                ),
                "cudaMemcpyAsync(raw camera input)"
            );

            launchTensorRtGpuPreprocess();
            if (!tensorRtContext->enqueueV3(tensorRtStream)) {
                throw std::runtime_error("TensorRT enqueueV3 failed.");
            }
            launchTensorRtGpuSelect();
            return;
        }

        const size_t inputBytes = tensorElementCount(tensorRtInputDims)
            * tensorElementSize(tensorRtInputType);
        const size_t outputBytes = tensorElementCount(tensorRtOutputDims)
            * tensorElementSize(tensorRtOutputType);

        throwOnCudaError(
            cudaMemcpyAsync(
                tensorRtInputDevice,
                tensorRtInputHost,
                inputBytes,
                cudaMemcpyHostToDevice,
                tensorRtStream
            ),
            "cudaMemcpyAsync(input)"
        );

        if (!tensorRtContext->enqueueV3(tensorRtStream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed.");
        }

        throwOnCudaError(
            cudaMemcpyAsync(
                tensorRtOutputHost,
                tensorRtOutputDevice,
                outputBytes,
                cudaMemcpyDeviceToHost,
                tensorRtStream
            ),
            "cudaMemcpyAsync(output)"
        );
    }

    bool initializeTensorRtCudaGraph()
    {
        bool captureStarted = false;

        try {
            // TensorRT may initialize resources lazily, so execute once before capture.
            enqueueTensorRtCommands();
            throwOnCudaError(cudaStreamSynchronize(tensorRtStream), "cudaStreamSynchronize(warmup)");

            throwOnCudaError(
                cudaStreamBeginCapture(tensorRtStream, cudaStreamCaptureModeGlobal),
                "cudaStreamBeginCapture"
            );
            captureStarted = true;
            enqueueTensorRtCommands();
            throwOnCudaError(
                cudaStreamEndCapture(tensorRtStream, &tensorRtGraph),
                "cudaStreamEndCapture"
            );
            captureStarted = false;

            throwOnCudaError(
                cudaGraphInstantiate(
                    &tensorRtGraphExec,
                    tensorRtGraph,
                    nullptr,
                    nullptr,
                    0
                ),
                "cudaGraphInstantiate"
            );
            return true;
        }
        catch (const std::exception& e) {
            if (captureStarted) {
                cudaGraph_t abandonedGraph = nullptr;
                cudaStreamEndCapture(tensorRtStream, &abandonedGraph);
                if (abandonedGraph != nullptr) {
                    cudaGraphDestroy(abandonedGraph);
                }
            }

            if (tensorRtGraphExec != nullptr) {
                cudaGraphExecDestroy(tensorRtGraphExec);
                tensorRtGraphExec = nullptr;
            }

            if (tensorRtGraph != nullptr) {
                cudaGraphDestroy(tensorRtGraph);
                tensorRtGraph = nullptr;
            }

            std::cerr << "TensorRT CUDA graph unavailable; using enqueueV3: "
                      << e.what()
                      << std::endl;
            return false;
        }
    }

    void prepareTensorRtInput(const cv::Mat& frame)
    {
        const int channels = tensorRtInputDims.d[1];
        const int inputHeight = tensorRtInputDims.d[2];
        const int inputWidth = tensorRtInputDims.d[3];
        if (channels != 3) {
            throw std::runtime_error("Only 3-channel TensorRT inputs are supported.");
        }

        if (frame.type() != CV_8UC3) {
            throw std::runtime_error("TensorRT input frame must be a BGR CV_8UC3 image.");
        }

        LetterboxTransform nextTransform;
        nextTransform.imageSize = frame.size();
        nextTransform.scale = std::min(
            static_cast<float>(inputWidth) / static_cast<float>(frame.cols),
            static_cast<float>(inputHeight) / static_cast<float>(frame.rows)
        );
        const cv::Size resizedSize(
            static_cast<int>(std::lround(frame.cols * nextTransform.scale)),
            static_cast<int>(std::lround(frame.rows * nextTransform.scale))
        );
        nextTransform.padLeft = (inputWidth - resizedSize.width) / 2;
        nextTransform.padTop = (inputHeight - resizedSize.height) / 2;

        if (tensorRtPreparedFrameSize != frame.size()) {
            fillTensorRtInputHost(114.0f / 255.0f);
            tensorRtPreparedFrameSize = frame.size();
        }
        tensorRtTransform = nextTransform;

        const cv::Mat* resizedImage = &frame;
        if (frame.size() != resizedSize) {
            cv::resize(frame, tensorRtResizedBgr, resizedSize, 0.0, 0.0, cv::INTER_LINEAR);
            resizedImage = &tensorRtResizedBgr;
        }

        const size_t planeSize = static_cast<size_t>(inputWidth)
            * static_cast<size_t>(inputHeight);
        constexpr float NORMALIZE_SCALE = 1.0f / 255.0f;

        const auto writeFloatRows = [&](const cv::Range& range) {
            float* input = static_cast<float*>(tensorRtInputHost);
            float* red = input;
            float* green = input + planeSize;
            float* blue = input + 2 * planeSize;

            for (int y = range.start; y < range.end; ++y) {
                const cv::Vec3b* sourceRow = resizedImage->ptr<cv::Vec3b>(y);
                const size_t destinationRow = static_cast<size_t>(y + nextTransform.padTop)
                    * static_cast<size_t>(inputWidth)
                    + static_cast<size_t>(nextTransform.padLeft);

                for (int x = 0; x < resizedSize.width; ++x) {
                    const size_t index = destinationRow + static_cast<size_t>(x);
                    const cv::Vec3b& pixel = sourceRow[x];
                    red[index] = static_cast<float>(pixel[2]) * NORMALIZE_SCALE;
                    green[index] = static_cast<float>(pixel[1]) * NORMALIZE_SCALE;
                    blue[index] = static_cast<float>(pixel[0]) * NORMALIZE_SCALE;
                }
            }
        };

        if (tensorRtInputType == nvinfer1::DataType::kHALF) {
            cv::parallel_for_(cv::Range(0, resizedSize.height), [&](const cv::Range& range) {
                __half* input = static_cast<__half*>(tensorRtInputHost);
                __half* red = input;
                __half* green = input + planeSize;
                __half* blue = input + 2 * planeSize;

                for (int y = range.start; y < range.end; ++y) {
                    const cv::Vec3b* sourceRow = resizedImage->ptr<cv::Vec3b>(y);
                    const size_t destinationRow = static_cast<size_t>(y + nextTransform.padTop)
                        * static_cast<size_t>(inputWidth)
                        + static_cast<size_t>(nextTransform.padLeft);

                    for (int x = 0; x < resizedSize.width; ++x) {
                        const size_t index = destinationRow + static_cast<size_t>(x);
                        const cv::Vec3b& pixel = sourceRow[x];
                        red[index] = tensorRtNormalizedHalfLut[pixel[2]];
                        green[index] = tensorRtNormalizedHalfLut[pixel[1]];
                        blue[index] = tensorRtNormalizedHalfLut[pixel[0]];
                    }
                }
            });
        }
        else if (tensorRtInputType == nvinfer1::DataType::kFLOAT) {
            cv::parallel_for_(cv::Range(0, resizedSize.height), writeFloatRows);
        }
        else {
            throw std::runtime_error("Only FP32/FP16 TensorRT inputs are supported.");
        }
    }

    void stageTensorRtRawInput(const cv::Mat& frame)
    {
        if (frame.type() != CV_8UC3) {
            throw std::runtime_error("TensorRT GPU input frame must be a BGR CV_8UC3 image.");
        }
        if (frame.size() != config.cameraFrameSize) {
            throw std::runtime_error(
                "TensorRT GPU pipeline requires a fixed "
                + std::to_string(config.cameraFrameSize.width)
                + "x" + std::to_string(config.cameraFrameSize.height)
                + " camera frame."
            );
        }

        const size_t rowBytes = static_cast<size_t>(tensorRtRawInputStride);
        if (frame.isContinuous() && frame.step == rowBytes) {
            std::memcpy(tensorRtRawInputHost, frame.data, tensorRtRawInputBytes);
            return;
        }

        auto* destination = static_cast<unsigned char*>(tensorRtRawInputHost);
        for (int row = 0; row < frame.rows; ++row) {
            std::memcpy(
                destination + static_cast<size_t>(row) * rowBytes,
                frame.ptr(row),
                rowBytes
            );
        }
    }

    std::vector<YoloHandBox> runTensorRtGpu(const cv::Mat& frame)
    {
        const auto preprocessStart = std::chrono::steady_clock::now();
        stageTensorRtRawInput(frame);
        const auto preprocessEnd = std::chrono::steady_clock::now();
        lastPerformance.preprocessMs = std::chrono::duration<double, std::milli>(
            preprocessEnd - preprocessStart
        ).count();

        const auto inferenceStart = std::chrono::steady_clock::now();
        if (tensorRtGraphReady) {
            throwOnCudaError(
                cudaGraphLaunch(tensorRtGraphExec, tensorRtStream),
                "cudaGraphLaunch"
            );
        }
        else {
            enqueueTensorRtCommands();
        }
        throwOnCudaError(cudaStreamSynchronize(tensorRtStream), "cudaStreamSynchronize");
        const auto inferenceEnd = std::chrono::steady_clock::now();
        lastPerformance.inferenceMs = std::chrono::duration<double, std::milli>(
            inferenceEnd - inferenceStart
        ).count();

        std::vector<YoloHandBox> boxes;
        if (config.multiHandTracking) {
            const TensorRtGpuHandBatch result = *tensorRtGpuMultiResultHost;
            const int count = std::clamp(result.count, 0, 2);
            boxes.reserve(static_cast<size_t>(count));
            for (int index = 0; index < count; ++index) {
                const TensorRtGpuHandResult& hand = result.hands[index];
                if (hand.found == 0) {
                    continue;
                }
                const cv::Rect box(
                    static_cast<int>(std::lround(hand.x)),
                    static_cast<int>(std::lround(hand.y)),
                    static_cast<int>(std::lround(hand.width)),
                    static_cast<int>(std::lround(hand.height))
                );
                addBoxIfValid(
                    boxes,
                    classNames,
                    hand.classId,
                    hand.confidence,
                    box
                );
            }
            return boxes;
        }

        const TensorRtGpuHandResult result = *tensorRtGpuResultHost;
        if (result.found == 0) {
            return boxes;
        }

        const cv::Rect box(
            static_cast<int>(std::lround(result.x)),
            static_cast<int>(std::lround(result.y)),
            static_cast<int>(std::lround(result.width)),
            static_cast<int>(std::lround(result.height))
        );
        addBoxIfValid(
            boxes,
            classNames,
            result.classId,
            result.confidence,
            box
        );
        return boxes;
    }

    const float* runTensorRtForward(const cv::Mat& frame)
    {
        const auto preprocessStart = std::chrono::steady_clock::now();
        prepareTensorRtInput(frame);
        const auto preprocessEnd = std::chrono::steady_clock::now();
        lastPerformance.preprocessMs = std::chrono::duration<double, std::milli>(
            preprocessEnd - preprocessStart
        ).count();

        const size_t outputElements = tensorElementCount(tensorRtOutputDims);

        const auto inferenceStart = std::chrono::steady_clock::now();
        if (tensorRtGraphReady) {
            throwOnCudaError(
                cudaGraphLaunch(tensorRtGraphExec, tensorRtStream),
                "cudaGraphLaunch"
            );
        }
        else {
            enqueueTensorRtCommands();
        }
        throwOnCudaError(cudaStreamSynchronize(tensorRtStream), "cudaStreamSynchronize");

        const float* output = nullptr;
        if (tensorRtOutputType == nvinfer1::DataType::kHALF) {
            const __half* outputHalf = static_cast<const __half*>(tensorRtOutputHost);
            for (size_t i = 0; i < outputElements; ++i) {
                tensorRtOutputFloat[i] = __half2float(outputHalf[i]);
            }
            output = tensorRtOutputFloat.data();
        }
        else if (tensorRtOutputType == nvinfer1::DataType::kFLOAT) {
            output = static_cast<const float*>(tensorRtOutputHost);
        }
        else {
            throw std::runtime_error("Only FP32/FP16 TensorRT outputs are supported.");
        }

        const auto inferenceEnd = std::chrono::steady_clock::now();
        lastPerformance.inferenceMs = std::chrono::duration<double, std::milli>(
            inferenceEnd - inferenceStart
        ).count();

        return output;
    }
#endif

    std::vector<YoloHandBox> runTensorRt(const cv::Mat& frame)
    {
        std::vector<YoloHandBox> rawBoxes;

#ifdef RHYTHM_USE_TENSORRT_YOLO
        if (tensorRtGpuPipelineActive) {
            return runTensorRtGpu(frame);
        }

        const float* data = runTensorRtForward(frame);
        const int rows = tensorRtOutputDims.d[1];
        const int dimensions = tensorRtOutputDims.d[2];
        const int configuredClassCount = static_cast<int>(classNames.size());

        if (rows == 4 + configuredClassCount && dimensions > rows) {
            for (int row = 0; row < dimensions; ++row) {
                int classId = -1;
                float confidence = 0.0f;
                for (int i = 0; i < configuredClassCount; ++i) {
                    const float score = data[(4 + i) * dimensions + row];
                    if (score > confidence) {
                        confidence = score;
                        classId = i;
                    }
                }

                if (confidence < config.confidenceThreshold) {
                    continue;
                }

                const cv::Rect box = letterboxCenterBoxToImage(
                    data[row],
                    data[dimensions + row],
                    data[2 * dimensions + row],
                    data[3 * dimensions + row],
                    tensorRtTransform
                );
                addBoxIfValid(rawBoxes, classNames, classId, confidence, box);
            }
        }
        else if (dimensions == 6) {
            for (int row = 0; row < rows; ++row) {
                const float* rowData = data + row * dimensions;
                const float confidence = rowData[4];
                if (confidence < config.confidenceThreshold) {
                    continue;
                }

                const int classId = static_cast<int>(rowData[5]);
                const cv::Rect box = letterboxXyxyToImage(
                    rowData[0],
                    rowData[1],
                    rowData[2],
                    rowData[3],
                    tensorRtTransform
                );
                addBoxIfValid(rawBoxes, classNames, classId, confidence, box);
            }
        }
        else if (dimensions >= 4 + configuredClassCount) {
            const int classCount = dimensions - 4;
            for (int row = 0; row < rows; ++row) {
                const float* rowData = data + row * dimensions;

                int classId = -1;
                float confidence = 0.0f;
                for (int i = 0; i < classCount; ++i) {
                    const float score = rowData[4 + i];
                    if (score > confidence) {
                        confidence = score;
                        classId = i;
                    }
                }

                if (confidence < config.confidenceThreshold) {
                    continue;
                }

                const cv::Rect box = letterboxCenterBoxToImage(
                    rowData[0],
                    rowData[1],
                    rowData[2],
                    rowData[3],
                    tensorRtTransform
                );
                addBoxIfValid(rawBoxes, classNames, classId, confidence, box);
            }
        }
#else
        (void)frame;
#endif

        return applyNms(rawBoxes);
    }

    std::vector<YoloHandBox> applyNms(const std::vector<YoloHandBox>& rawBoxes) const
    {
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> confidences;
        for (const YoloHandBox& box : rawBoxes) {
            nmsBoxes.push_back(box.box);
            confidences.push_back(box.confidence);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(
            nmsBoxes,
            confidences,
            config.confidenceThreshold,
            config.nmsThreshold,
            indices
        );

        std::vector<YoloHandBox> boxes;
        boxes.reserve(indices.size());
        for (int index : indices) {
            boxes.push_back(rawBoxes[index]);
        }

        return boxes;
    }

};

YoloHandTracker::YoloHandTracker(const YoloHandTrackerConfig& config)
    : impl(std::make_unique<Impl>(config))
{
}

YoloHandTracker::~YoloHandTracker() = default;

bool YoloHandTracker::initialize()
{
    impl->classNames = loadClassNames(impl->config.classNamesPath);
    impl->lastError.clear();

    try {
        if (impl->config.backend == YoloBackend::OpenCvDnn) {
            const std::filesystem::path modelPath = resolveProjectPath(impl->config.onnxModelPath);

            if (!std::filesystem::exists(modelPath)) {
                impl->lastError = "ONNX model not found: " + modelPath.string();
                impl->initialized = false;
                return false;
            }

            impl->openCvNet = cv::dnn::readNetFromONNX(modelPath.string());

            if (impl->config.useCudaForOpenCvDnn) {
                impl->openCvNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                impl->openCvNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
            else {
                impl->openCvNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                impl->openCvNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }

            cv::Mat warmupFrame(impl->config.networkInputSize, CV_8UC3, cv::Scalar());
            cv::Mat warmupBlob = cv::dnn::blobFromImage(
                warmupFrame,
                1.0 / 255.0,
                impl->config.networkInputSize,
                cv::Scalar(),
                true,
                false
            );

            impl->openCvNet.setInput(warmupBlob);
            std::vector<cv::Mat> warmupOutputs;
            impl->openCvNet.forward(warmupOutputs, impl->openCvNet.getUnconnectedOutLayersNames());

            impl->initialized = true;
            return true;
        }

        if (impl->config.backend == YoloBackend::TensorRt) {
#ifdef RHYTHM_USE_TENSORRT_YOLO
            std::filesystem::path enginePath;

            if (impl->config.preferFastTensorRtEngine) {
                const std::filesystem::path fastEnginePath = resolveProjectPath(
                    impl->config.tensorRtFastEnginePath
                );
                if (std::filesystem::exists(fastEnginePath)) {
                    enginePath = fastEnginePath;
                }
            }

            if (impl->config.preferFp16IoTensorRtEngine) {
                const std::filesystem::path fp16IoEnginePath = resolveProjectPath(
                    impl->config.tensorRtFp16IoEnginePath
                );
                if (enginePath.empty() && std::filesystem::exists(fp16IoEnginePath)) {
                    enginePath = fp16IoEnginePath;
                }
            }

            // Preserve compatibility by falling back to the original FP32 I/O engine.
            if (enginePath.empty()) {
                enginePath = resolveProjectPath(impl->config.tensorRtEnginePath);
            }

            if (!std::filesystem::exists(enginePath)) {
                impl->lastError = "TensorRT engine not found: " + enginePath.string();
                impl->initialized = false;
                return false;
            }

            impl->initialized = impl->initializeTensorRt(enginePath);
            return impl->initialized;
#else
            impl->lastError =
                "TensorRT backend is not compiled. Define RHYTHM_USE_TENSORRT_YOLO "
                "and add the TensorRT-YOLO source files, CUDA, and TensorRT libraries.";
            impl->initialized = false;
            return false;
#endif
        }
    }
    catch (const cv::Exception& e) {
        impl->lastError = e.what();
        impl->initialized = false;
        return false;
    }
    catch (const std::exception& e) {
        impl->lastError = e.what();
        impl->initialized = false;
        return false;
    }

    impl->lastError = "Unknown YOLO backend.";
    impl->initialized = false;
    return false;
}

bool YoloHandTracker::isInitialized() const
{
    return impl->initialized;
}

bool YoloHandTracker::detect(const cv::Mat& frame, YoloHandDetection& detection)
{
    detection = YoloHandDetection();
    impl->lastPerformance = YoloHandPerformance();

    if (!impl->initialized) {
        impl->lastError = "YoloHandTracker is not initialized.";
        return false;
    }

    if (frame.empty()) {
        impl->lastError = "Input frame is empty.";
        return false;
    }

    impl->lastError.clear();
    const auto detectStart = std::chrono::steady_clock::now();
    const auto finishTiming = [&]() {
        impl->lastPerformance.totalMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - detectStart
        ).count();
        impl->lastPerformance.postprocessMs = std::max(
            0.0,
            impl->lastPerformance.totalMs
                - impl->lastPerformance.preprocessMs
                - impl->lastPerformance.inferenceMs
        );
    };

    std::vector<YoloHandBox> boxes;
    bool tensorRtGpuPostprocessedResult = false;

#ifdef RHYTHM_USE_TENSORRT_YOLO
    tensorRtGpuPostprocessedResult = impl->config.backend == YoloBackend::TensorRt
        && impl->tensorRtGpuPipelineActive;
#endif

    try {
        if (impl->config.backend == YoloBackend::OpenCvDnn) {
            boxes = impl->runOpenCvDnn(frame);
        }
        else {
            boxes = impl->runTensorRt(frame);
        }
    }
    catch (const cv::Exception& e) {
        impl->lastError = e.what();
        finishTiming();
        return false;
    }
    catch (const std::exception& e) {
        impl->lastError = e.what();
        finishTiming();
        return false;
    }

    if (impl->config.mirrorX && !tensorRtGpuPostprocessedResult) {
        for (YoloHandBox& box : boxes) {
            box.box.x = frame.cols - box.box.x - box.box.width;
            box.center.x = box.box.x + box.box.width / 2;
        }
    }

    detection.boxes = boxes;

    if (impl->config.multiHandTracking) {
        auto& hands = detection.boxes;
        hands.erase(std::remove_if(hands.begin(), hands.end(), [&](const YoloHandBox& box) {
            return box.classId != impl->config.handClassId;
        }), hands.end());
        std::sort(hands.begin(), hands.end(), [](const YoloHandBox& a, const YoloHandBox& b) {
            return a.confidence > b.confidence;
        });
        if (hands.size() > 2) hands.resize(2);
        detection.found = !hands.empty();
        if (detection.found) {
            detection.hand = hands.front();
            detection.gameCenter = mapCameraPointToGame(detection.hand.center, frame.size());
        }
        finishTiming();
        return detection.found;
    }

    if (tensorRtGpuPostprocessedResult) {
        if (detection.boxes.empty()) {
            impl->hasLastHand = false;
            impl->missedHandFrames = 0;
            finishTiming();
            return false;
        }

        detection.found = true;
        detection.hand = detection.boxes.front();
        detection.gameCenter = mapCameraPointToGame(detection.hand.center, frame.size());
        impl->hasLastHand = true;
        impl->missedHandFrames = 0;
        impl->lastHand = detection.hand;
        impl->lastGameCenter = detection.gameCenter;
        finishTiming();
        return true;
    }

    if (impl->hasLastHand) {
        const double frameDiagonal = std::hypot(
            static_cast<double>(frame.cols),
            static_cast<double>(frame.rows)
        );
        const double maxCenterJump = impl->config.maxCenterJumpRatio * frameDiagonal;
        const double lastArea = std::max(1, impl->lastHand.box.area());

        detection.boxes.erase(
            std::remove_if(
                detection.boxes.begin(),
                detection.boxes.end(),
                [&](const YoloHandBox& box) {
                    if (box.classId != impl->config.handClassId) {
                        return false;
                    }

                    const double centerJump = std::hypot(
                        static_cast<double>(box.center.x - impl->lastHand.center.x),
                        static_cast<double>(box.center.y - impl->lastHand.center.y)
                    );
                    const double currentArea = std::max(1, box.box.area());
                    const double areaChange = std::max(
                        currentArea / lastArea,
                        lastArea / currentArea
                    );
                    return centerJump > maxCenterJump
                        || areaChange > impl->config.maxBoxAreaChangeRatio;
                }
            ),
            detection.boxes.end()
        );
    }

    auto bestHand = std::max_element(
        detection.boxes.begin(),
        detection.boxes.end(),
        [this](const YoloHandBox& a, const YoloHandBox& b) {
            const bool aIsHand = (a.classId == impl->config.handClassId);
            const bool bIsHand = (b.classId == impl->config.handClassId);

            if (aIsHand != bIsHand) {
                return !aIsHand;
            }

            return a.confidence < b.confidence;
        }
    );

    if (bestHand == detection.boxes.end() || bestHand->classId != impl->config.handClassId) {
        if (impl->hasLastHand && impl->missedHandFrames < impl->config.holdFramesAfterMiss) {
            impl->missedHandFrames++;
            detection.found = true;
            detection.hand = impl->lastHand;
            detection.gameCenter = impl->lastGameCenter;
            detection.boxes.push_back(impl->lastHand);
            finishTiming();
            return true;
        }

        impl->hasLastHand = false;
        impl->missedHandFrames = 0;
        finishTiming();
        return false;
    }

    YoloHandBox hand = *bestHand;
    if (impl->hasLastHand) {
        const float alpha = std::clamp(impl->config.smoothingAlpha, 0.0f, 1.0f);
        const float oldWeight = 1.0f - alpha;

        hand.box.x = static_cast<int>(std::lround(alpha * hand.box.x + oldWeight * impl->lastHand.box.x));
        hand.box.y = static_cast<int>(std::lround(alpha * hand.box.y + oldWeight * impl->lastHand.box.y));
        hand.box.width = static_cast<int>(std::lround(alpha * hand.box.width + oldWeight * impl->lastHand.box.width));
        hand.box.height = static_cast<int>(std::lround(alpha * hand.box.height + oldWeight * impl->lastHand.box.height));
        hand.center = cv::Point(hand.box.x + hand.box.width / 2, hand.box.y + hand.box.height / 2);
    }

    detection.found = true;
    detection.hand = hand;
    detection.gameCenter = mapCameraPointToGame(hand.center, frame.size());

    impl->hasLastHand = true;
    impl->missedHandFrames = 0;
    impl->lastHand = detection.hand;
    impl->lastGameCenter = detection.gameCenter;
    finishTiming();
    return true;
}

cv::Point YoloHandTracker::mapCameraPointToGame(const cv::Point& cameraPoint, const cv::Size& cameraFrameSize) const
{
    if (!impl->config.cameraToGameHomography.empty()) {
        std::vector<cv::Point2f> input = {
            cv::Point2f(static_cast<float>(cameraPoint.x), static_cast<float>(cameraPoint.y))
        };
        std::vector<cv::Point2f> output;

        cv::perspectiveTransform(input, output, impl->config.cameraToGameHomography);

        if (!output.empty()) {
            return cv::Point(
                static_cast<int>(std::lround(output[0].x)),
                static_cast<int>(std::lround(output[0].y))
            );
        }
    }

    const cv::Size sourceSize = (cameraFrameSize.width > 0 && cameraFrameSize.height > 0)
        ? cameraFrameSize
        : impl->config.cameraFrameSize;

    const double xScale = static_cast<double>(impl->config.gameOutputSize.width) / static_cast<double>(sourceSize.width);
    const double yScale = static_cast<double>(impl->config.gameOutputSize.height) / static_cast<double>(sourceSize.height);

    return cv::Point(
        static_cast<int>(std::lround(cameraPoint.x * xScale)),
        static_cast<int>(std::lround(cameraPoint.y * yScale))
    );
}

cv::Point YoloHandTracker::mapCameraPointToGame(const cv::Point& cameraPoint) const
{
    return mapCameraPointToGame(cameraPoint, impl->config.cameraFrameSize);
}

void YoloHandTracker::drawDebug(cv::Mat& frame, const YoloHandDetection& detection) const
{
    for (const YoloHandBox& box : detection.boxes) {
        const bool isHand = (box.classId == impl->config.handClassId);
        const cv::Scalar color = isHand ? cv::Scalar(60, 220, 60) : cv::Scalar(0, 180, 255);

        cv::rectangle(frame, box.box, color, 2);

        std::ostringstream label;
        label << box.label << " " << std::fixed << std::setprecision(2) << box.confidence;

        cv::putText(
            frame,
            label.str(),
            cv::Point(box.box.x, std::max(18, box.box.y - 6)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        );

        cv::circle(frame, box.center, 4, color, -1);
    }

    if (detection.found) {
        cv::circle(frame, detection.hand.center, 8, cv::Scalar(255, 255, 255), 2);
    }
}

const YoloHandTrackerConfig& YoloHandTracker::config() const
{
    return impl->config;
}

const std::string& YoloHandTracker::lastError() const
{
    return impl->lastError;
}

const YoloHandPerformance& YoloHandTracker::lastPerformance() const
{
    return impl->lastPerformance;
}
