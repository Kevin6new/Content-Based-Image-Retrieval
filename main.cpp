/*
Basil Reji & Kevin Sebastian Sani
Spring 2024
Pattern Recognition & Computer Vision
Assignment 2 - Content-based Image Retrieval
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <cstring>
#include "csv_util.h"

// Function declarations for various image processing and feature extraction tasks.

cv::Mat calculateColorHistogram(const cv::Mat& image, int bins);
cv::Mat extractCentralRegion(const cv::Mat& image);
double calculateSSD(const cv::Mat& img1, const cv::Mat& img2);
cv::Mat calculate2DHistogram(const cv::Mat& image, int bins);
double histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2);
float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2);
int sobelX3x3(cv::Mat& src, cv::Mat& dst);
int sobelY3x3(cv::Mat& src, cv::Mat& dst);
double combinedDistanceMetric(const cv::Mat& histColor1, const cv::Mat& histTexture1, const cv::Mat& histColor2, const cv::Mat& histTexture2);
void calculateGradientMagnitude(const cv::Mat& sobelX, const cv::Mat& sobelY, cv::Mat& gradientMagnitude);
cv::Mat calculateTextureHistogram(const cv::Mat& gradientMagnitude, int bins);

//Base Line Matching
// Compares the target image against a dataset to find the most similar images based on grayscale intensity. 
// It extracts the central region of the target and dataset images, computes the sum of squared differences (SSD) for similarity, 
// and returns the top N matches.

std::vector<std::pair<double, std::string>> baselineMatching(const std::string& targetImagePath, const std::string& directoryPath, int topN) {
    cv::Mat targetImageColor = cv::imread(targetImagePath, cv::IMREAD_COLOR);
    cv::Mat targetImage;
    
    cv::cvtColor(targetImageColor, targetImage, cv::COLOR_BGR2GRAY);
    cv::Mat targetRegion = extractCentralRegion(targetImage);
    std::vector<std::pair<double, std::string>> distances;
    
    DIR* dir = opendir(directoryPath.c_str());
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".jpg")) {
            std::string filePath = directoryPath + "/" + entry->d_name;
            cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
            if (image.empty()) continue;

            cv::Mat imageRegion = extractCentralRegion(image);
            double dist = calculateSSD(targetRegion, imageRegion);
            distances.emplace_back(dist, filePath);
        }
    }
    closedir(dir);

    std::sort(distances.begin(), distances.end());
    if (distances.size() > static_cast<size_t>(topN)) distances.resize(topN);
    return distances;
}

//Histogram Matching
// Utilizes color histograms to compare the target image with images in a dataset, identifying images with similar color distributions. 
// This function calculates a 2D histogram for each image, measures similarity using histogram intersection, 
// and returns the top N matches.

std::vector<std::pair<double, std::string>> histogramMatching(const std::string& targetImagePath, const std::string& directoryPath, int topN) {
    cv::Mat targetImage = cv::imread(targetImagePath);
    cv::Mat targetHist = calculate2DHistogram(targetImage, 16);
    
    DIR* dir = opendir(directoryPath.c_str());
    std::vector<std::pair<double, std::string>> imageDistances;
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != nullptr) {
        std::string filePath = directoryPath + "/" + entry->d_name;
        if (filePath == targetImagePath || strstr(entry->d_name, ".jpg") == nullptr) continue;
        cv::Mat image = cv::imread(filePath);
        if (image.empty()) continue;

        cv::Mat hist = calculate2DHistogram(image, 16);
        double intersection = histogramIntersection(targetHist, hist);
        imageDistances.push_back(std::make_pair(intersection, filePath));
    }
    closedir(dir);

    std::sort(imageDistances.begin(), imageDistances.end(), std::greater<>());
    if (imageDistances.size() > static_cast<size_t>(topN)) imageDistances.resize(topN);
    return imageDistances;
}

//Multi Histogram Matching
// Enhances the histogram matching technique by using both global and central region histograms for comparison. 

std::vector<std::pair<double, std::string>> multiHistogramMatching(const std::string& targetImagePath, const std::string& directoryPath, int topN) {
    cv::Mat targetImage = cv::imread(targetImagePath);
    int bins = 8; 
    
    cv::Mat targetHistWhole = calculateColorHistogram(targetImage, bins);
    cv::Rect centerRegion(targetImage.cols / 4, targetImage.rows / 4, targetImage.cols / 2, targetImage.rows / 2);
    cv::Mat targetHistCenter = calculateColorHistogram(targetImage, bins);

    std::vector<std::pair<double, std::string>> imageScores;
    DIR* dir = opendir(directoryPath.c_str());
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != nullptr) {
        std::string filePath = directoryPath + "/" + entry->d_name;
        if (filePath == targetImagePath || strstr(entry->d_name, ".jpg") == nullptr) continue;

        cv::Mat image = cv::imread(filePath);
        if (image.empty()) continue;

        cv::Mat histWhole = calculateColorHistogram(image, bins);
        cv::Mat histCenter = calculateColorHistogram(image, bins);

        double scoreWhole = histogramIntersection(targetHistWhole, histWhole);
        double scoreCenter = histogramIntersection(targetHistCenter, histCenter);
        double combinedScore = (scoreWhole + scoreCenter) / 2.0;

        imageScores.emplace_back(combinedScore, filePath);
    }
    closedir(dir);

    std::sort(imageScores.begin(), imageScores.end(), std::greater<>());
    if (imageScores.size() > static_cast<size_t>(topN)) imageScores.resize(topN);
    return imageScores;
}

//Texture and Color Matching
// Combines texture and color information to find similar images. 
// It employs Sobel filters to capture texture details and color histograms for color information, 
// using a combined distance metric to rank the top N matches.

std::vector<std::pair<double, std::string>> textureAndColorMatching(const std::string& targetImagePath, const std::string& directoryPath, int topN, int bins) {
    cv::Mat targetImage = cv::imread(targetImagePath);
    cv::Mat targetSobelX, targetSobelY, targetGradientMagnitude;
    sobelX3x3(targetImage, targetSobelX);
    sobelY3x3(targetImage, targetSobelY);
    calculateGradientMagnitude(targetSobelX, targetSobelY, targetGradientMagnitude);
    cv::Mat targetColorHist = calculateColorHistogram(targetImage, bins);
    cv::Mat targetTextureHist = calculateTextureHistogram(targetGradientMagnitude, bins);

    std::vector<std::pair<double, std::string>> imageScores;

    DIR* dir = opendir(directoryPath.c_str());
    dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filePath = directoryPath + entry->d_name;
        if (filePath == targetImagePath || !strstr(entry->d_name, ".jpg")) continue;

        cv::Mat image = cv::imread(filePath);
        cv::Mat imageSobelX, imageSobelY, imageGradientMagnitude;
        sobelX3x3(image, imageSobelX);
        sobelY3x3(image, imageSobelY);
        calculateGradientMagnitude(imageSobelX, imageSobelY, imageGradientMagnitude);
        cv::Mat imageColorHist = calculateColorHistogram(image, bins);
        cv::Mat imageTextureHist = calculateTextureHistogram(imageGradientMagnitude, bins);

        double score = combinedDistanceMetric(targetColorHist, targetTextureHist, imageColorHist, imageTextureHist);
        imageScores.emplace_back(score, filePath);
    }
    closedir(dir);

    std::sort(imageScores.begin(), imageScores.end());
    if (imageScores.size() > topN) imageScores.resize(topN);

    return imageScores;
}

//Deep Network Embeddings
// Leverages deep neural network (DNN) embeddings to compare images. 
// It extracts feature vectors using a pre-trained model, calculates cosine distances between the target and dataset images, 
// and identifies the top N closest matches based on these embeddings.

std::vector<std::pair<float, std::string>> dnnEmbeddingMatching(const std::string& csvFilePath, const std::string& targetImageName, int topN) {
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;

    read_image_data_csv(const_cast<char*>(csvFilePath.c_str()), filenames, data, 0);

    std::vector<float> targetEmbedding;
    for (int i = 0; i < filenames.size(); ++i) {
        if (std::string(filenames[i]) == targetImageName) {
            targetEmbedding = data[i];
            break;
        }
    }
    std::vector<std::pair<float, std::string>> distances;
    for (int i = 0; i < data.size(); ++i) {
        if (std::string(filenames[i]) != targetImageName) {
            float dist = cosineDistance(targetEmbedding, data[i]);
            distances.emplace_back(dist, std::string(filenames[i]));
        }
    }
    std::sort(distances.begin(), distances.end());

    if (distances.size() > topN) {
        distances.resize(topN);
    }

    for (char* fname : filenames) {
        delete[] fname;
    }

    return distances;
}

//Custom Matching
// Performs a custom image matching by combining deep neural network (DNN) embeddings and color histogram comparisons.
// This function first retrieves DNN embeddings for the target image and compares them with embeddings of images in a dataset to calculate cosine distances.
// Additionally, it computes color histograms for both the target and dataset images to measure similarity based on color distribution.
// The final similarity score for each image is a combination (average) of the DNN embedding distance and the color histogram intersection.
// The function sorts all images based on this combined score and returns the top N matches along with a selection of least similar matches,
// effectively utilizing both texture (via DNN embeddings) and color information for content-based image retrieval.


std::vector<std::pair<float, std::string>> customMatching(const std::string& targetImagePath, const std::string& directoryPath, const std::string& csvFilePath, int topN) {
    auto embeddings = dnnEmbeddingMatching(csvFilePath, targetImagePath,topN); 
    std::map<std::string, std::vector<float>> embeddingMap;
    for (auto& pair : embeddings) {
        embeddingMap[pair.second] = {}; 
    }

    cv::Mat targetImageColor = cv::imread(targetImagePath);
    cv::Mat targetHist = calculateColorHistogram(targetImageColor, 16);
    std::vector<std::pair<float, std::string>> combinedScores;

    DIR* dir = opendir(directoryPath.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string fileName = entry->d_name;
            std::string filePath = directoryPath + "/" + fileName;
            if (filePath.find(".jpg") != std::string::npos && fileName != targetImagePath) {
                if (embeddingMap.find(fileName) != embeddingMap.end()) {
                    float distDnn = cosineDistance(embeddingMap[targetImagePath], embeddingMap[fileName]);
                    cv::Mat img = cv::imread(filePath);
                    cv::Mat hist = calculateColorHistogram(img, 16);
                    double distHist = histogramIntersection(targetHist, hist);
                    float combinedScore = (distDnn + static_cast<float>(distHist)) / 2.0f;
                    combinedScores.emplace_back(combinedScore, filePath);
                }
            }
        }
        closedir(dir);
    }

    std::sort(combinedScores.begin(), combinedScores.end());

    if (combinedScores.size() > topN * 2) {
        combinedScores.erase(combinedScores.begin() + topN, combinedScores.end() - topN);
    }

    return combinedScores;
}

int main() {

    // Paths for the target image, directory of images, and CSV file for DNN embeddings

    std::string targetImagePath = "C://Users//kevin//source//repos//Project 2//olympus//olympus//pic.0387.jpg";
    std::string imageDirectory = "C://Users//kevin//source//repos//Project 2//olympus//olympus";
    std::string csvFilePath = "C://Users//kevin//source//repos//Project 2//ResNet18_olym.csv";

    std::cout << "Select matching technique:\n";
    std::cout << "1. Baseline Matching\n";
    std::cout << "2. Histogram Matching\n";
    std::cout << "3. Multi-Histogram Matching\n";
    std::cout << "4. Texture and Color Matching\n";
    std::cout << "5. DNN Embedding Matching\n";
    std::cout << "6. Custom Matching\n";
    std::cout << "Enter choice: ";
    int choice;
    std::cin >> choice;

    // Process the choice and perform the selected image matching technique

    switch (choice) {
    // Case 1: Baseline Matching

    case 1: {
        auto matches = baselineMatching(targetImagePath, imageDirectory, 3);
        for (size_t i = 0; i < matches.size(); ++i) {
            const auto& match = matches[i];
            cv::Mat img = cv::imread(match.second);
            std::string windowName = "Match #" + std::to_string(i + 1);
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
            cv::imshow(windowName, img);
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
        break;
    }
    
    // Case 2: Histogram Matching

    case 2: {
        auto matches = histogramMatching(targetImagePath, imageDirectory, 3);
        for (size_t i = 0; i < matches.size(); ++i) {
            cv::Mat img = cv::imread(matches[i].second);
            if (!img.empty()) {
                std::string windowName = "Match #" + std::to_string(i + 1);
                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                cv::imshow(windowName, img);
            }
            cv::waitKey(0);
            cv::destroyAllWindows();
            break;
        }
    }

    // Case 3: Multi-Histogram Matching

    case 3: {
        auto matches = multiHistogramMatching(targetImagePath, imageDirectory, 3);
        for (size_t i = 0; i < matches.size(); ++i) {
            cv::Mat img = cv::imread(matches[i].second);
            if (!img.empty()) {
                std::string windowName = "Match #" + std::to_string(i + 1);
                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                cv::imshow(windowName, img);
            }
            cv::waitKey(0);
            cv::destroyAllWindows();
            break;
        }
    }

    // Case 4: Texture and Color Matching

    case 4: {
        auto matches = textureAndColorMatching(targetImagePath, imageDirectory, 3, 16);

        cv::Mat targetImg = cv::imread(targetImagePath);
        if (!targetImg.empty()) {
            cv::namedWindow("Target Image", cv::WINDOW_AUTOSIZE);
            cv::imshow("Target Image", targetImg);
        }

        for (size_t i = 0; i < matches.size(); ++i) {
            cv::Mat img = cv::imread(matches[i].second);
            if (!img.empty()) {
                std::string windowName = "Match #" + std::to_string(i + 1);
                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                cv::imshow(windowName, img);
            }
        }

        cv::waitKey(0);
        cv::destroyAllWindows();
        break;
    }

    // Case 5: DNN Embedding Matching

    case 5: {
        auto matches = dnnEmbeddingMatching(csvFilePath, targetImagePath, 5);
        cv::Mat targetImg = cv::imread(targetImagePath);
        if (!targetImg.empty()) {
            cv::namedWindow("Target Image", cv::WINDOW_AUTOSIZE);
            cv::imshow("Target Image", targetImg);
        }

        for (size_t i = 0; i < matches.size(); ++i) {
            std::string matchImagePath = matches[i].second;
            cv::Mat img = cv::imread(matchImagePath);
            if (!img.empty()) {
                std::string windowName = "Match #" + std::to_string(i + 1);
                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                cv::imshow(windowName, img);
            }
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
        break;
    }

    // Case 6: Custom Matching

    case 6: {
        auto combinedScores = customMatching(targetImagePath, imageDirectory, csvFilePath, 5);
        
        cv::Mat targetImg = cv::imread(targetImagePath);
        if (!targetImg.empty()) {
            cv::namedWindow("Target Image", cv::WINDOW_AUTOSIZE);
            cv::imshow("Target Image", targetImg);
        }

        std::cout << "Most Similar Matches:" << std::endl;
        for (int i = 0; i < 5 && i < combinedScores.size() / 2; i++) { 
            cv::Mat img = cv::imread(combinedScores[i].second);
            if (!img.empty()) {
                std::string windowName = "Most Similar#" + std::to_string(i + 1);
                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                cv::imshow(windowName, img);
            }
        }

        std::cout << "Least Similar Matches:" << std::endl;
        for (int i = combinedScores.size() / 2; i < combinedScores.size(); i++) {
            cv::Mat img = cv::imread(combinedScores[i].second);
            if (!img.empty()) {
                std::string windowName = "Least Similar#" + std::to_string(i - (combinedScores.size() / 2) + 1);
                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                cv::imshow(windowName, img);
            }
        }

        cv::waitKey(0);
        cv::destroyAllWindows();
        break;
    }

    default:
        std::cout << "Invalid choice." << std::endl;
    }

          return 0;

    }
