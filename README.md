# 2DHeadPose: a simple and effective annotation method for the head pose in RGB images and its dataset

Head pose estimation is one of the essential tasks in computer vision, which predicts
the Euler angles of the head in an image. In recent years, CNN-based methods for
head pose estimation have achieved excellent performance. Their training relies on
RGB images providing facial landmarks or depth images from RGBD cameras.
However, labeling facial landmarks is complex for large angular head poses in RGB
images, and RGBD cameras are unsuitable for outdoor scenes. We propose a simple,
effective, and convenient method for annotating the Euler angles of head poses in
RGB images. The method novely uses a 3D virtual human head to simulate the head
pose in the RGB image. The Euler angle can be calculated from the change in
coordinates of the 3D virtual head. We then create a dataset using our annotation
method: 2DHeadPose dataset, which contains a rich set of attributes, dimensions, and
angles. Finally, we propose a label smoothing method to suppress the noise
introduced by the head pose annotation process and establish a baseline approach.
Experiments demonstrate that our annotation method, datasets, and label smoothing
are very effective. Our baseline approach surpasses most current state-of-the-art
methods. The annotation tool, dataset, and source code are publicly available at
https://github.com/youngnuaa/2DHeadPose .

LIST:
 annotation tool (uploaded)
 dataset （coming soon）
 source code （coming soon）
 
 
 
The dataset baidu download path：https://pan.baidu.com/s/1CZPqyOMMCKNmV9uSLg8QNA?pwd=581p  code:581p

The dataset google download path：https://drive.google.com/file/d/14BGFd8zrz-e1xaRxMQvgnUU3FV41pMMm/view?usp=share_link
