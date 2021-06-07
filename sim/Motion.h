#ifndef __MOTION_H__
#define __MOTION_H__
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <string>
class BVH;

class Motion
{
public:
	Motion():mNumFrames(0){};
	Motion(BVH* bvh);

	void registerBVHHierarchy(BVH* bvh);
	void append(const Eigen::Vector3d& position,
		const Eigen::MatrixXd& rotation,
		bool compute_velocity = true);
	void append(const std::vector<Eigen::Vector3d>& positions,
		const std::vector<Eigen::MatrixXd>& rotations,
		bool compute_velocity = true);
	void clear();

	int getNumJoints(){return mNumJoints;}
	double getTimestep(){return mTimestep;}
	
	const std::vector<std::string>& getNames(){return mNames;}
	const std::vector<Eigen::Vector3d>& getOffsets(){return mOffsets;}
	const std::vector<int>& getParents(){return mParents;}

	int getNumFrames(){return mNumFrames;}
	
	BVH* getBVH(){return mBVH;}	
	const Eigen::Vector3d& getPosition(int idx){return mPositions[idx];}
	const Eigen::MatrixXd& getRotation(int idx){return mRotations[idx];}
	const Eigen::Vector3d& getLinearVelocity(int idx){return mLinearVelocities[idx];}
	const Eigen::MatrixXd& getAngularVelocity(int idx){return mAngularVelocities[idx];}

	void computeVelocity();
private:
	
	void computeVelocity(int start);
	void computeVelocity(int start, int end);//[start,end)

	BVH* mBVH;
	int mNumJoints;
	double mTimestep;
	std::vector<std::string> mNames;
	std::vector<Eigen::Vector3d> mOffsets;
	std::vector<int> mParents;

	int mNumFrames;
	std::vector<Eigen::Vector3d> mPositions; // 1x3
	std::vector<Eigen::MatrixXd> mRotations; // 3*3n
	std::vector<Eigen::Vector3d> mLinearVelocities; // 1*3
	std::vector<Eigen::MatrixXd> mAngularVelocities; // 3*n

	
};
class MotionUtils
{
public:
	static Eigen::MatrixXd computePoseDifferences(Motion* m);

	// static void registerJointWeights(Motion* m, const std::map<std::string, double>& joint_weights);
	static double computePoseDifference(const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rj);
	static Eigen::MatrixXd computePoseDisplacement(const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rj);
	static Eigen::MatrixXd addDisplacement(const Eigen::MatrixXd& R, const Eigen::MatrixXd& d);
	static double easeInEaseOut(double x, double yp0 = 0.0, double yp1 = 0.0);

	static Eigen::Isometry3d getReferenceTransform(const Eigen::Vector3d& pos, const Eigen::MatrixXd& rot);
	static Motion* blendUpperLowerMotion(BVH* bvh_lb, BVH* bvh_ub, int start_lb, int start_ub);
	static Motion* parseMotionLabel(const std::string& line, int fps = 30);

private:
	// static Eigen::VectorXd gJointWeights;
};
#endif