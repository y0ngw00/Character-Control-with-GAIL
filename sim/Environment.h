#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__
#include "dart/dart.hpp"
#include <tuple>

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
	int getDimStateLabel();
	int getNumTotalLabel();

	double getTargetHeading();
	double getTargetHeight();
	double getTargetSpeed();
	const Eigen::Vector3d getTargetDirection();

	void setTargetHeading(double heading);
	void setTargetSpeed(double speed);
	void setTargetHeight(double height);


	void reset(int frame=-1);

	void step(const Eigen::VectorXd& action);
	
	void resetGoal();
	void updateGoal();
	double getRewardGoal();
	int getStateLabel();
	void setStateLabel(int label);

	const Eigen::VectorXd& getState();
	const Eigen::VectorXd& getStateGoal();
	const Eigen::VectorXd& getStateAMP();

	Eigen::MatrixXd getStateAMPExpert();

	bool inspectEndOfEpisode();

	const dart::simulation::WorldPtr& getWorld(){return mWorld;}

	Character* getSimCharacter(){return mSimCharacter;}
	Character* getKinCharacter(){return mKinCharacter;}
	dart::dynamics::SkeletonPtr getGround(){return mGround;}

	bool isEnableGoal(){return mEnableGoal;}
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

	Character *mSimCharacter,*mKinCharacter;
	std::vector<Motion*> mMotions;

	dart::dynamics::SkeletonPtr mGround;

	Eigen::VectorXd mPrevPositions, mPrevPositions2,mPrevPositions3, mPrevCOM;
	Eigen::VectorXd mState, mStateGoal, mStateAMP;

	bool mContactEOE;
	bool mEnableGoal;

	double mRewardGoal;

	int mNumMotions, mStateLabel, mNumFeatureAdds,mDimLabel;

	double mTargetHeading, mTargetSpeed;
	Eigen::Vector3d mTargetDirection;
	double mTargetSpeedMin, mTargetSpeedMax;
	double mTargetFrame;
	double mTargetHeight, mIdleHeight;
	double mSharpTurnProb, mSpeedChangeProb,mHeightChangeProb, mMaxHeadingTurnRate;

	double mTargetHeightMin, mTargetHeightMax;

};
#endif