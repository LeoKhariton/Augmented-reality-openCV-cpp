#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	setlocale(0, "ru");

	VideoCapture cap(1);
	Mat imgCam;

	Mat imgTarget = imread("D:target.jpg");
	resize(imgTarget, imgTarget, Size(), .5, .5);
	int hT = imgTarget.rows;
	int wT = imgTarget.cols;
	
	//VideoCapture myVid("D:\\video.mp4");
	Mat imgVid;
	//myVid.read(imgVid);
	imgVid = imread("D:\\image.png");
	resize(imgVid, imgVid, Size(wT, hT));
	
	Ptr<ORB> detector = ORB::create(1000);

	vector<KeyPoint> keypoints1;
	Mat descriptors1;
	detector->detectAndCompute(imgTarget, noArray(), keypoints1, descriptors1);
	//drawKeypoints(imgTarget, keypoints1, imgTarget, Scalar(0, 0, 255));

	int key = 0;
	while (key != 'q')
	{
		cap.read(imgCam);
		Mat imgAug;
		copyTo(imgCam, imgAug, noArray());

		vector<KeyPoint> keypoints2;
		Mat descriptors2;
		detector->detectAndCompute(imgCam, noArray(), keypoints2, descriptors2);
		//drawKeypoints(imgCam, keypoints2, imgCam);

		Ptr<BFMatcher> matcher = BFMatcher::create();

		vector<vector<DMatch>> knn_matches;
		matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

		vector<DMatch> good_matches;

		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < .75f * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}

		Mat img_matches;
		drawMatches(imgTarget, keypoints1, imgCam, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		if (good_matches.size() > 20)
		{
			vector<Point2f> obj, scene;
			for (size_t i = 0; i < good_matches.size(); i++)
			{
				obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
				scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
			}
			Mat H = findHomography(obj, scene, RANSAC, 5);

			vector<Point2f> obj_cornes(4);
			obj_cornes[0] = Point2f(0, 0);
			obj_cornes[1] = Point2f((float)wT, 0);
			obj_cornes[2] = Point2f((float)wT, (float)hT);
			obj_cornes[3] = Point2f(0, (float)hT);
			
			vector<Point2f> scene_cornes(4);

			perspectiveTransform(obj_cornes, scene_cornes, H);

			vector<vector<Point>> scene_cornes1 = { { scene_cornes[0] + Point2f((float)wT, 0), scene_cornes[1] + Point2f((float)wT, 0),
													  scene_cornes[1] + Point2f((float)wT, 0), scene_cornes[2] + Point2f((float)wT, 0),
													  scene_cornes[2] + Point2f((float)wT, 0), scene_cornes[3] + Point2f((float)wT, 0),
													  scene_cornes[3] + Point2f((float)wT, 0), scene_cornes[0] + Point2f((float)wT, 0) } };
			polylines(img_matches, scene_cornes1, true, Scalar(0, 255, 0), 4);

			Mat imgWarp;
			warpPerspective(imgVid, imgWarp, H, Size(imgCam.cols, imgCam.rows));
			//imshow("Warp Image", imgWarp);

			Mat maskNew = Mat::zeros(Size(imgCam.cols, imgCam.rows), CV_8U);

			vector<vector<Point>> scene_cornes2 = { { scene_cornes[0], scene_cornes[1],
													  scene_cornes[1], scene_cornes[2],
													  scene_cornes[2], scene_cornes[3],
													  scene_cornes[3], scene_cornes[0] } };
			fillPoly(maskNew, scene_cornes2, Scalar(255, 255, 255));

			//imshow("Mask", maskNew);
			Mat maskInv;
			bitwise_not(maskNew, maskInv);
			Mat res;
			bitwise_and(imgAug, imgAug, res, maskInv);
			//imshow("Final", res);
			bitwise_or(imgWarp, res, imgAug);
			//imshow("MaskInv", maskInv);
			imshow("ImgAug", imgAug);
		}
		//imshow("Èçîáðàæåíèå ñ êàìåðû", img_matches);
		key = waitKey(1);
	}
	waitKey();
	return 0;
}
