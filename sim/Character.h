#ifndef __CHARACTER_H__
#define __CHARACTER_H__
#include "dart/dart.hpp"

class Motion;
class Character
{
public:
	Character(dart::dynamics::SkeletonPtr& skel,
			const std::vector<dart::dynamics::BodyNode*>& end_effectors,
			const std::vector<std::string>& bvh_map,
			const Eigen::VectorXd& w_joint,
			const Eigen::VectorXd& kp,
			const Eigen::VectorXd& maxf);

	Eigen::Isometry3d getRootTransform();
	void setRootTransform(const Eigen::Isometry3d& T);

	Eigen::Isometry3d getReferenceTransform();
	void setReferenceTransform(const Eigen::Isometry3d& T_ref);

	void setPose(const Eigen::Vector3d& position,
				const Eigen::MatrixXd& rotation);
	void setPose(const Eigen::Vector3d& position,
				const Eigen::MatrixXd& rotation,
				const Eigen::Vector3d& linear_velocity,
				const Eigen::MatrixXd& angular_velocity);

	Eigen::VectorXd computeTargetPosition(const Eigen::VectorXd& action);
	Eigen::VectorXd computeAvgVelocity(const Eigen::VectorXd& p0, const Eigen::VectorXd& p1, double dt);
	void actuate(const Eigen::VectorXd& target_position);

	std::vector<Eigen::Vector3d> getState();

	Eigen::VectorXd getStateAMP();

	Eigen::VectorXd saveState();
	void restoreState(const Eigen::VectorXd& state);

	void buildBVHIndices(const std::vector<std::string>& bvh_names);

	dart::dynamics::SkeletonPtr getSkeleton(){return mSkeleton;}
	Motion* getMotion(){return mMotion;}
	const Eigen::VectorXd& getTargetPositions(){return mTargetPositions;}
	const std::vector<dart::dynamics::BodyNode*>& getEndEffectors(){return mEndEffectors;}
	int getBVHIndex(int idx){return mBVHIndices[idx];}
	const std::vector<int>& getBVHIndices(){return mBVHIndices;}
	const Eigen::VectorXd& getJointWeights(){return mJointWeights;}
	std::map<std::string, Eigen::MatrixXd> getStateBody();
	std::map<std::string, Eigen::MatrixXd> getStateJoint();
private:

	Eigen::VectorXd toSimPose(const Eigen::Vector3d& position,
							const Eigen::MatrixXd& rotation);
	Eigen::VectorXd toSimVel(const Eigen::Vector3d& position,
							const Eigen::MatrixXd& rotation,
							const Eigen::Vector3d& linear_velocity,
							const Eigen::MatrixXd& angular_velcity);

	dart::dynamics::SkeletonPtr mSkeleton;

	std::vector<dart::dynamics::BodyNode*> mEndEffectors;

	Motion* mMotion;
	std::vector<std::string> mBVHNames;
	std::vector<std::string> mBVHMap;
	std::vector<int> mBVHIndices;

	Eigen::VectorXd mJointWeights;

	Eigen::VectorXd mKp, mKv, mMinForces, mMaxForces, mTargetPositions;
};

#endif