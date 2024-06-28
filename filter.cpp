/*
Basil Reji & Kevin Sebastian Sani
Spring 2024
Pattern Recognition & Computer Vision
Assignment 2 - Content-based Image Retrieval
Filter.cpp file containing all functions relating to Image Retrieval
*/

#include<opencv2/opencv.hpp>
#include <iostream>
#include<math.h>
#include <ctime>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

// Applies a 3x3 Sobel filter in the X direction to detect horizontal edges in an image.

int sobelX3x3(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat temp2 = cv::Mat::zeros(src.size(), CV_16SC3);

    // Horizontal filter
    for (int i = 1; i < src.rows - 1; ++i)
    {
        cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s* dptr = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; ++j)
        {
            for (int c = 0; c <= 2; c++)
            {
                dptr[j][c] = (-1 * rptr[j - 1][c] + rptr[j + 1][c]) / 2;
            }
        }
    }

    // Vertical filter
    for (int i = 1; i < src.rows - 1; ++i)
    {
        cv::Vec3s* rptr = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s* dptr = temp2.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; ++j)
        {
            for (int c = 0; c <= 2; c++)
            {
                dptr[j][c] = (-1 * rptr[j - 1][c] + rptr[j + 1][c]) / 2;
            }
        }
    }

    temp2.copyTo(dst);
    return 0;
}


// Applies a 3x3 Sobel filter in the Y direction to detect vertical edges in an image.

int sobelY3x3(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int i = 1; i < src.rows - 1; ++i)
    {
        cv::Vec3b* rptrm1 = src.ptr <cv::Vec3b>(i - 1);
        cv::Vec3b* rptr = src.ptr <cv::Vec3b>(i);
        cv::Vec3b* rptrp1 = src.ptr <cv::Vec3b>(i + 1);

        cv::Vec3s* dptr = temp.ptr <cv::Vec3s>(i);
        for (int j = 1; j < src.cols - 1; ++j)
        {
            for (int c = 0; c <= 2; c++)
            {
                dptr[j][c] = (1 * rptrm1[j - 1][c] + 2 * rptrm1[j][c] + rptrm1[j + 1][c]
                    - 1 * rptrp1[j - 1][c] - 2 * rptrp1[j][c] - rptrp1[j + 1][c]) / 4;
            }
        }
    }

    temp.copyTo(dst);
    return 0;

}

// Calculates a 3D color histogram for an image using specified bins for each color channel.
// It normalizes the histogram so that its sum is 1.0, to compare histograms.

cv::Mat calculateColorHistogram(const cv::Mat& image, int bins) {
    int histSize[] = { bins, bins, bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range, range, range };
    int channels[] = { 0, 1, 2 };
    cv::Mat hist;
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    return hist;
}

// Computes the cosine distance between two feature vectors 'v1' and 'v2'.
// This is used for measuring similarity between images based on their feature vectors.

float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        denom_a += v1[i] * v1[i];
        denom_b += v2[i] * v2[i];
    }
    denom_a = sqrt(denom_a);
    denom_b = sqrt(denom_b);
    float cos = dot / (denom_a * denom_b);
    return 1.0 - cos;
}

// Extracts a central region of a given image. 

cv::Mat extractCentralRegion(const cv::Mat& image) {
    const int size = 9;
    cv::Rect roi((image.cols - size) / 2, (image.rows - size) / 2, size, size);
    return image(roi);
}

// Calculates the Sum of Squared Differences (SSD) between two images 

double calculateSSD(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff = diff.mul(diff);
    cv::Scalar sum = cv::sum(diff);
    return sum[0] + sum[1] + sum[2];
}

// Computes a 2D histogram for an image,to be used for texture analysis.

cv::Mat calculate2DHistogram(const cv::Mat& image, int bins) {
    int histSize[] = { bins, bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range, range };
    int channels[] = { 0, 1 };
    cv::Mat hist;

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return hist;
}

// Calculates the intersection between two histograms 

double histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2) {
    double intersection = 0;
    for (int i = 0; i < hist1.rows; ++i) {
        for (int j = 0; j < hist1.cols; ++j) {
            intersection += (std::min)(hist1.at<float>(i, j), hist2.at<float>(i, j));
        }
    }
    return intersection;
}

// Calculates the gradient magnitude of an image based on Sobel X and Y gradients.

void calculateGradientMagnitude(const cv::Mat& sobelX, const cv::Mat& sobelY, cv::Mat& gradientMagnitude) {
    gradientMagnitude = cv::Mat::zeros(sobelX.size(), CV_32F);
    for (int i = 0; i < sobelX.rows; ++i) {
        for (int j = 0; j < sobelX.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                float gx = sobelX.at<cv::Vec3s>(i, j)[c];
                float gy = sobelY.at<cv::Vec3s>(i, j)[c];
                gradientMagnitude.at<float>(i, j) += std::sqrt(gx * gx + gy * gy) / 3.0f;
            }
        }
    }
}

// Computes a histogram based on the gradient magnitude of an image, for texture analysis.
// The histogram is normalized.

cv::Mat calculateTextureHistogram(const cv::Mat& gradientMagnitude, int bins) {
    int histSize[] = { bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    cv::Mat hist;
    cv::calcHist(&gradientMagnitude, 1, nullptr, cv::Mat(), hist, 1, histSize, ranges, true, false);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    return hist;
}

// Combines color and texture histogram intersection scores to compute a similarity metric.
// This function averages the intersections of color and texture histograms between two sets of histograms.

double combinedDistanceMetric(const cv::Mat& histColor1, const cv::Mat& histTexture1,
    const cv::Mat& histColor2, const cv::Mat& histTexture2) {
    double colorIntersection = histogramIntersection(histColor1, histColor2);
    double textureIntersection = histogramIntersection(histTexture1, histTexture2);
    return (colorIntersection + textureIntersection) / 2.0;
}
