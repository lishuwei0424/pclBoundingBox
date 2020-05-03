#include <vtkAutoInit.h>         
VTK_MODULE_INIT(vtkRenderingOpenGL);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
typedef pcl::PointXYZ PointType;

int main(int argc, char **argv)
{
	// 导入点云
	pcl::PointCloud<PointType>::Ptr original_cloud(new pcl::PointCloud<PointType>());
	
	std::string fileName("pcl.pcd");
	pcl::io::loadPCDFile(fileName, *original_cloud);

	// PCA：计算主方向
	Eigen::Vector4f centroid;							// 质心
	pcl::compute3DCentroid(*original_cloud, centroid);	// 齐次坐标，（c0,c1,c2,1）
	
	Eigen::Matrix3f covariance;	
	computeCovarianceMatrixNormalized(*original_cloud, centroid, covariance);		// 计算归一化协方差矩阵
	
	// 计算主方向：特征向量和特征值
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
	//Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();
	eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));	// 校正主方向间垂直（特征向量方向： (e0, e1, e0 × e1) --- note: e0 × e1 = +/- e2）
	
	// 转到参考坐标系，将点云主方向与参考坐标系的坐标轴进行对齐
	Eigen::Matrix4f transformation(Eigen::Matrix4f::Identity());
	transformation.block<3, 3>(0, 0) = eigen_vectors.transpose();										// R^(-1) = R^T
	transformation.block<3, 1>(0, 3) = -1.f * (transformation.block<3, 3>(0, 0) * centroid.head<3>());	// t^(-1) = -R^T * t

	pcl::PointCloud<PointType> transformed_cloud;	// 变换后的点云
	pcl::transformPointCloud(*original_cloud, transformed_cloud, transformation);

	PointType min_pt, max_pt;						// 沿参考坐标系坐标轴的边界值
	pcl::getMinMax3D(transformed_cloud, min_pt, max_pt);
	const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());	// 形心

	// 参考坐标系到主方向坐标系的变换关系
	const Eigen::Quaternionf qfinal(eigen_vectors);
	const Eigen::Vector3f tfinal = eigen_vectors * mean_diag + centroid.head<3>();

	// 显示结果
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(original_cloud);

	viewer.addCoordinateSystem();

	// 显示点云主方向
	Eigen::Vector3f whd;		// 3个方向尺寸：宽高深
	whd = max_pt.getVector3fMap() - min_pt.getVector3fMap();// getVector3fMap:返回Eigen::Map<Eigen::Vector3f> 
	float scale = (whd(0) + whd(1) + whd(2)) / 3;			// 点云平均尺度，用于设置主方向箭头大小

	// std::cout << "width/heigth/depth：" << whd << endl;

	PointType cp;			// 箭头由质心分别指向pirncipal_dir_X、pirncipal_dir_Y、pirncipal_dir_Z
	cp.x = centroid(0);
	cp.y = centroid(1);
	cp.z = centroid(2);

	PointType principal_dir_X;
	principal_dir_X.x = scale * eigen_vectors(0, 0) + cp.x;
	principal_dir_X.y = scale * eigen_vectors(1, 0) + cp.y;
	principal_dir_X.z = scale * eigen_vectors(2, 0) + cp.z;

	PointType principal_dir_Y;
	principal_dir_Y.x = scale * eigen_vectors(0, 1) + cp.x;
	principal_dir_Y.y = scale * eigen_vectors(1, 1) + cp.y;
	principal_dir_Y.z = scale * eigen_vectors(2, 1) + cp.z;

	PointType principal_dir_Z;
	principal_dir_Z.x = scale * eigen_vectors(0, 2) + cp.x;
	principal_dir_Z.y = scale * eigen_vectors(1, 2) + cp.y;
	principal_dir_Z.z = scale * eigen_vectors(2, 2) + cp.z;

	viewer.addArrow(principal_dir_X, cp, 1.0, 0.0, 0.0, false, "arrow_x");		// 箭头附在起点上
	viewer.addArrow(principal_dir_Y, cp, 0.0, 1.0, 0.0, false, "arrow_y");
	viewer.addArrow(principal_dir_Z, cp, 0.0, 0.0, 1.0, false, "arrow_z");

	// 显示包围盒，并设置包围盒属性，以显示透明度
	viewer.addCube(tfinal, qfinal, whd(0), whd(1), whd(2), "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "bbox");
	
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
	}
	return 0;
}

