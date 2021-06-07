#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__
#include "dart/dart.hpp"
#include <tuple>
#include "Event.h"
#include "Distribution.hpp"
#include "MassSpringDamperSystem.h"
#include "ForceSensor.h"

class BVH;
class Motion;
class Character;
class CartesianMSDSystem;
class Environment
{
public:
	Environment();

	int getDimState();
	int getDimAction();
	int getDimStateAMP();

	void resetGoal();
	void reset(int frame=-1);

	void updateGoal();
	void step(const Eigen::VectorXd& action);
	
	double getRewardGoal();

	const Eigen::VectorXd& getState();
	const Eigen::VectorXd& getStateGoal();
	const Eigen::VectorXd& getStateAMP();

	Eigen::MatrixXd getStateAMPExpert();

	bool inspectEndOfEpisode();

	void updateObstacle();
	void generateObstacle();
	void generateObstacleForce();

	void applyRootSupportForce();

	const dart::simulation::WorldPtr& getWorld(){return mWorld;}
	dart::dynamics::SkeletonPtr getObstacle(){return mObstacle;}

	Character* getSimCharacter(){return mSimCharacter;}
	Character* getKinCharacter(){return mKinCharacter;}
	dart::dynamics::SkeletonPtr getGround(){return mGround;}
	double getTargetHeading(){return mTargetHeading;}
	double getTargetSpeed(){return mTargetSpeed;}
	bool isEnableGoal(){return mEnableGoal;}
	bool isKinematics(){return mKinematics;}
	void setKinematics(bool kin){mKinematics = kin;}
	const dart::dynamics::SkeletonPtr& getDoor(){return mDoor;}
	double getTargetDoorAngle(){return mTargetDoorAngle;}
	Eigen::VectorXd computePoseDiffState(const Eigen::Vector3d& position,
				const Eigen::MatrixXd& rotation,
				const Eigen::Vector3d& linear_velocity,
				const Eigen::MatrixXd& angular_velocity);
	double computePoseDiffReward(const Eigen::Vector3d& position,
				const Eigen::MatrixXd& rotation,
				const Eigen::Vector3d& linear_velocity,
				const Eigen::MatrixXd& angular_velocity);
	const Eigen::Vector3d& getCurrentTargetHandPos(){return mCurrentTargetHandPos;}
private:
	double computeGroundHeight();
	void recordState();
	void recordGoal();

	Eigen::MatrixXd getActionSpace();
	Eigen::VectorXd getActionWeight();
	Eigen::VectorXd convertToRealActionSpace(const Eigen::VectorXd& a_norm);

	Eigen::MatrixXd mActionSpace;
	Eigen::VectorXd mActionWeight;

	dart::simulation::WorldPtr mWorld;
	int mControlHz, mSimulationHz;
	int mElapsedFrame, mFrame;
	int mMaxElapsedFrame;
	bool mKinematics;
	Distribution1D<int>* mInitialStateDistribution;
	Character *mSimCharacter,*mKinCharacter;
	Motion* mCurrentMotion;
	std::vector<Motion*> mMotions;

	double mGroungHeight;
	dart::dynamics::SkeletonPtr mGround;
	dart::dynamics::SkeletonPtr mDoor;
	dart::constraint::BallJointConstraintPtr mDoorConstraint;

	Eigen::VectorXd mPrevPositions, mPrevVelocities, mPrevCOM;
	Eigen::VectorXd mPrevPositions2;
	std::vector<Eigen::VectorXd> mPrevMSDStates;
	Eigen::VectorXd mState, mStateGoal, mStateAMP;
	Eigen::MatrixXd mStateAMPExpert;
	Eigen::Vector6d mRestRootPosition;

	bool mContactEOE;
	bool mEnableGoal;
	bool mEnableObstacle;

	double mRewardGoal;
	bool mEnableGoalEOE;
	std::vector<double> mRewardGoals;

	double mTargetHeading, mTargetSpeed;
	double mTargetSpeedMin, mTargetSpeedMax;
	double mTargetFrame;
	double mSharpTurnProb, mSpeedChangeProb, mMaxHeadingTurnRate;

	double mTargetDoorAngle, mPrevDoorAngle;
	int mObstacleCount, mObstacleFinishCount;
	dart::dynamics::SkeletonPtr mObstacle;

	CartesianMSDSystem* mCartesianMSDSystem;
	
	int mForceCount;
	dart::constraint::BallJointConstraintPtr mBallConstraint;

	dart::dynamics::SkeletonPtr mWeldObstacle;

	Eigen::Vector3d mInitHandPos, mCurrentTargetHandPos, mCurrentTargetHandVel, mTargetHandPos, mTargetHandPos2;
	void generateTargetHandPos();
	void updateTargetHandPos();
	int mTargetHandCount, mTargetHandFinishCount;

	Eigen::MatrixXd mInitialRotation;
	Eigen::Vector3d mInitialPosition;
};

#endif