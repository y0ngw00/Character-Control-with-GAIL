#include <functional>
#include <fstream>
#include <sstream>
#include "Environment.h"
#include "DARTUtils.h"
#include "MassSpringDamperSystem.h"
#include "MathUtils.h"
#include "Event.h"
#include "Character.h"
#include "BVH.h"
#include "Motion.h"
#include "TwoJointIK.h"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/fcl/fcl.hpp"

using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;

Environment::
Environment()
	:mWorld(std::make_shared<World>()),
	mControlHz(30),
	mSimulationHz(300),
	mElapsedFrame(0),
	mMaxElapsedFrame(300),
	mSimCharacter(nullptr),
	mKinCharacter(nullptr),
	mTargetSpeedMin(0.5),
	mTargetSpeedMax(3.5),
	mSharpTurnProb(0.01),
	mSpeedChangeProb(0.05),
	mMaxHeadingTurnRate(0.15),
	mRewardGoal(0.0),
	mEnableGoal(false),
	mEnableObstacle(true),
	mEnableGoalEOE(false),
	mKinematics(false)
{
	dart::math::Random::generateSeed(true);

	mSimCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel_c.xml");
	mKinCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel_c.xml");

	
	std::vector<int> start_frames = {100, 2910};
	for(int i=0;i<10;i++)
	{
		start_frames.emplace_back(i*300+300);
	}

	std::vector<std::string> motion_lists = {
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Knockback_BtoF_01.bvh",
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Knockback_FtoB_01.bvh",
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Knockback_LtoR_04.bvh",
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Knockback_RtoL_01.bvh",
	"Anim_KraalA_A_Ax2H_A_Standard_Ext_KnockDown_BtoF_01.bvh",
	"Anim_KraalA_A_Ax2H_A_Standard_Ext_KnockDown_FtoB_02.bvh",
	"Anim_KraalA_A_Ax2H_A_Standard_Ext_KnockDown_LtoR_02.bvh",
	"Anim_KraalA_A_Ax2H_A_Standard_Ext_KnockDown_RtoL_04.bvh"
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Slide_BtoF_05.bvh",
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Slide_FtoB_01.bvh",
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Slide_LtoR_04.bvh",
	// "Anim_KraalA_A_Ax2H_A_Standard_Ext_Slide_RtoL_01.bvh"
	};

	// for(int i =0;i<motion_lists.size();i++)
	// {
	// 	BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/mona_retarget/"+motion_lists[i]);
	// 	Motion* motion = new Motion(bvh);
	// 	for(int j=0;j<bvh->getNumFrames();j++)
	// 		motion->append(bvh->getPosition(j), bvh->getRotation(j),false);
	// 	// for(int j=66;j<66+40;j++)
	// 	// 	motion->append(bvh->getPosition(j), bvh->getRotation(j),false);
		
	// 	motion->computeVelocity();
	// 	mMotions.emplace_back(motion);

	// 	if(i==0)
	// 	{
	// 		mSimCharacter->buildBVHIndices(bvh->getNodeNames());
	// 		mKinCharacter->buildBVHIndices(bvh->getNodeNames());
	// 		int n = motion->getNumFrames()-1;

	// 		mInitialStateDistribution = new Distribution1D<int>(n, [=](int i)->int{return i;}, 0.7);
	// 	}
	// }
	// auto motion = MotionUtils::parseMotionLabel("walk_long.bvh 1:44:04 1:47:23");
	
	
	
	{
		BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/walk_long.bvh");
		Motion* motion = new Motion(bvh);
		// int hand = bvh->getNodeIndex("simRightHand");
		// TwoJointIK ik(bvh, hand);
		Eigen::Vector3d pos = bvh->getPosition(84);
		Eigen::MatrixXd rot = bvh->getRotation(84);
		// Eigen::Isometry3d T_target = bvh->forwardKinematics(pos, rot, hand)[0];
		// T_target.translation()[0] += 0.25;
		// T_target.translation()[1] += 0.3;
		// T_target.translation()[2] += 0.3;
		// ik.solve(T_target, pos, rot);
		for(int j=0;j<300;j++)
			motion->append(pos, rot,false);

		motion->computeVelocity();
		mMotions.emplace_back(motion);
		// mSimCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
		// mKinCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
	}

	auto motion = MotionUtils::parseMotionLabel("walk_long.bvh 1:44:04 2:24:00");
	// auto motion = MotionUtils::parseMotionLabel("walk_long.bvh 0:04:04 2:24:00");

	mSimCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
	mKinCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
	mMotions.emplace_back(motion);

	Eigen::Vector3d mass_coeffs(0.01, 1e7, 0.01);
	Eigen::Vector3d spring_coeffs(5.0, 0.0, 5.0);
	Eigen::Vector3d damper_coeffs(0.6, 0.0, 0.6);

	// mCartesianMSDSystem = new CartesianMSDSystem(mass_coeffs, spring_coeffs, damper_coeffs, 1.0/mSimulationHz);

	std::string msd_path = std::string(ROOT_DIR)+"/data/msd.txt";
	std::ifstream ifs(msd_path);
	std::string msd_line;
	while(!ifs.eof())
	{
		msd_line.clear();
		std::getline(ifs, msd_line);
		if(msd_line.size() == 0)
			continue;
		mSimCharacter->parseMSD(msd_line, 1.0/mControlHz);	
		mKinCharacter->parseMSD(msd_line, 1.0/mControlHz);	
	}
	ifs.close();
	// mSimCharacter->parseMSD("ROOT 0.01 20.0 0.6");
	// mSimCharacter->parseMSD("JOINT simSpine 0.01 20.0 0.6");
	// mSimCharacter->parseMSD("JOINT simLeftArm 0.01 20.0 0.6");
	// mSimCharacter->parseMSD("JOINT simLeftForeArm 0.01 20.0 0.6");
	// mSimCharacter->parseMSD("JOINT simRightArm 0.01 20.0 0.6");
	// mSimCharacter->parseMSD("JOINT simRightForeArm 0.01 20.0 0.6");
	// mSimCharacter->parseMSD("JOINT simHead 0.01 20.0 0.6");

	// mStartFrame = mInitialStateDistribution->sample();
	int stride = 60;
	mInitialStateDistribution = new Distribution1D<int>(motion->getNumFrames()/stride, [=](int i){return i*stride;}, 0.7);
	// 	Eigen::Map<Eigen::ArrayXd> rs(mRewards["r"].data(), mRewards["r"].size());
	

	mGroungHeight = this->computeGroundHeight();
	mGround = DARTUtils::createGround(mGroungHeight);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	// mSimCharacter->getSkeleton()->enableSelfCollisionCheck();
	mWorld->addSkeleton(mSimCharacter->getSkeleton());
	mWorld->addSkeleton(mGround);
	mWorld->setTimeStep(1.0/(double)mSimulationHz);
	mWorld->setGravity(Eigen::Vector3d(0,-9.81,0.0));
	mElapsedFrame = 0;
	mFrame = mInitialStateDistribution->sample();

	mSimCharacter->getSkeleton()->setSelfCollisionCheck(true);
	mSimCharacter->getSkeleton()->setAdjacentBodyCheck(false);
	this->reset();

	mActionSpace = this->getActionSpace();
	mActionWeight = this->getActionWeight();
}

int
Environment::
getDimState()
{
	return this->getState().rows();
}
int
Environment::
getDimAction()
{
	int n = mSimCharacter->getSkeleton()->getNumDofs();
	return n-6;
}
int
Environment::
getDimStateAMP()
{
	return this->getStateAMP().rows();
}
void
Environment::
reset(int frame)
{

	mContactEOE = false;
	mRewardGoals.clear();
	for(int i=0;i<mControlHz;i++)
		mRewardGoals.emplace_back(1.0);
	// int mi = dart::math::Random::uniform<int>(0, mMotions.size()-1);
	// auto motion = mMotions[mi];
	auto motion = mMotions[0];
	mCurrentMotion = mMotions[0];
	int nf = motion->getNumFrames();
	// mInitialStateDistribution->update(mElapsedFrame);
	// mFrame = mInitialStateDistribution->sample();
	mFrame = 0;
	// mFrame = 66;
	// std::cout<<mInitialStateDistribution->getValue().transpose()<<std::endl;
	mElapsedFrame = 0;

	mSimCharacter->resetMSD();
	mSimCharacter->resetForceSensor();
	// if(frame<0)
		// mStartFrame = mInitialStateDistribution->sample();

	// mFrame = dart::math::Random::uniform<int>(0, nf-2);
		// mFrame = dart::math::Random::uniform<int>(0, 3000);
	


	Eigen::Vector3d position = mCurrentMotion->getPosition(mFrame);
	Eigen::MatrixXd rotation = mCurrentMotion->getRotation(mFrame);
	Eigen::Vector3d linear_velocity = mCurrentMotion->getLinearVelocity(mFrame);
	Eigen::MatrixXd angular_velocity = mCurrentMotion->getAngularVelocity(mFrame);

	mSimCharacter->setPose(position, rotation, linear_velocity, angular_velocity);

	mRestRootPosition = mSimCharacter->getSkeleton()->getPositions().head<6>();

	mSimCharacter->getSkeleton()->clearConstraintImpulses();
	mSimCharacter->getSkeleton()->clearInternalForces();
	mSimCharacter->getSkeleton()->clearExternalForces();

	mKinCharacter->setPose(position, rotation, linear_velocity, angular_velocity);


	// mCartesianMSDSystem->reset();

	
	
	mPrevPositions2 = mSimCharacter->getSkeleton()->getPositions(); 
	mPrevPositions = mSimCharacter->getSkeleton()->getPositions();
	// mPrevVelocities = mSimCharacter->getSkeleton()->getVelocities();
	mPrevCOM = mSimCharacter->getSkeleton()->getCOM();
	mPrevDoorAngle = 0.0;
	mPrevMSDStates = mSimCharacter->saveStateMSD();

	// if(mEnableGoal)
	if(0)
	{

		if(mDoor ==nullptr)
		{
			double door_size = 1.2;
			Eigen::Vector3d door_hinge_pos(0.27-door_size,mGroungHeight+0.01,0.48);
			mDoor = DARTUtils::createDoor(door_hinge_pos, door_size);
			mDoorConstraint =
		        std::make_shared<dart::constraint::BallJointConstraint>(mSimCharacter->getSkeleton()->getBodyNode("LeftHand"), 
						mDoor->getBodyNode(1),mSimCharacter->getSkeleton()->getBodyNode("LeftHand")->getCOM());

		    mWorld->getConstraintSolver()->addConstraint(mDoorConstraint);	
			mWorld->addSkeleton(mDoor);
		}
		else
		{
			Eigen::VectorXd p = mDoor->getPositions();
			Eigen::VectorXd v = mDoor->getVelocities();
			p.setZero();
			v.setZero();
			mDoor->setPositions(p);
			mDoor->setVelocities(v);
		}

		
	}

	if(mEnableGoal){
		this->resetGoal();
		this->recordGoal();	
	}
	if(mEnableObstacle)
		this->generateObstacle();
	mForceCount = 0;
	mObstacleCount = 0;
	mObstacleFinishCount = 0;
	if(mObstacle !=nullptr)
	{
		Eigen::VectorXd pos = mObstacle->getPositions();
		Eigen::VectorXd vel = mObstacle->getVelocities();
		
		pos.setZero();
		vel.setZero();
		
		mObstacle->setPositions(pos);
		mObstacle->setVelocities(vel);	
	}
	

	// if(mWorld->hasSkeleton(mObstacle))
	// 	mWorld->removeSkeleton(mObstacle);
	this->recordState();

	mTargetHandPos = mSimCharacter->getSkeleton()->getBodyNode("LeftHand")->getCOM();
	// mTargetHandPos2 = mTargetHandPos + 5.0*Eigen::Vector3d::Random();
	mTargetHandPos2 = mTargetHandPos;
	mTargetHandCount = 0;
	mTargetHandFinishCount = 0;//dart::math::Random::uniform<int>(0,1);

	// mTargetHandFinishCount = dart::math::Random::uniform<int>(30,60);
	mInitHandPos = mTargetHandPos;
}
void
Environment::
step(const Eigen::VectorXd& _action)
{
	Eigen::VectorXd action = this->convertToRealActionSpace(_action);

	auto sim_skel = mSimCharacter->getSkeleton();
	int num_sub_steps = mSimulationHz/mControlHz;

	auto target_pos = mSimCharacter->computeTargetPosition(action);

	this->generateTargetHandPos();
	this->updateTargetHandPos();
	if(mEnableObstacle)
	{
		// if(dart::math::Random::uniform<double>(0.0,1.0)<0.02)
		// if(dart::math::Random::uniform<double>(0.0,1.0)<0.05)
		this->generateObstacle();
		this->updateObstacle();
	}
	
	// if(mForceCount>60 && dart::math::Random::uniform<double>(0.0,1.0)<0.2)
	// {
	// 	mForceCount = 0;
	// 	std::vector<Eigen::Vector3d> random_forces = {
	// 		Eigen::Vector3d::UnitX()*3.0,
	// 		-Eigen::Vector3d::UnitX()*3.0,
	// 		Eigen::Vector3d::UnitZ()*3.0,
	// 		-Eigen::Vector3d::UnitZ()*3.0};
		
	// 	mCartesianMSDSystem->setPosition(random_forces[dart::math::Random::uniform<int>(0, 3)]);
	// }

	mForceCount++;
	if(mKinematics)
	{
		auto motion = mCurrentMotion;
		mFrame %= motion->getNumFrames();
		Eigen::Vector3d position = motion->getPosition(mFrame);
		Eigen::MatrixXd rotation = motion->getRotation(mFrame);
		Eigen::Vector3d linear_velocity = motion->getLinearVelocity(mFrame);
		Eigen::MatrixXd angular_velocity = motion->getAngularVelocity(mFrame);
		mSimCharacter->setPose(position, rotation, linear_velocity, angular_velocity);
	}
	else
	{

		mSimCharacter->applyForceMSD();
		mSimCharacter->stepMSD();
		for(int i=0;i<num_sub_steps;i++)
		{
			mSimCharacter->actuate(target_pos);
			if(mEnableObstacle)
				this->updateObstacle();
			mWorld->step();
			auto cr = mWorld->getConstraintSolver()->getLastCollisionResult();
			auto hand = mSimCharacter->getSkeleton()->getBodyNode("RightHand");
			double kp = 100.0;
			double kv = 2.0*sqrt(kp);
			// Eigen::Vector3d hand_force = kp*(mCurrentTargetHandPos - hand->getCOM()) - kv*hand->getCOMLinearVelocity();
			Eigen::Vector3d hand_force = kp*(mCurrentTargetHandPos - hand->getCOM());

			// mSimCharacter->getClosestForceSensor(hand->getCOM())->addExternalForce(hand_force);
			mSimCharacter->getClosestForceSensor(hand->getCOM())->addExternalForce(hand->getConstraintImpulse().tail<3>()*mSimulationHz);

			// mSimCharacter->applyForceMSD(hand, Eigen::Vector3d::Zero(), hand_force);
			for(int j=0;j<cr.getNumContacts();j++)
			{
				auto contact = cr.getContact(j);
				auto shapeFrame1 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject1->getShapeFrame());
				auto shapeFrame2 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject2->getShapeFrame());

				auto bn1 = shapeFrame1->asShapeNode()->getBodyNodePtr();
				auto bn2 = shapeFrame2->asShapeNode()->getBodyNodePtr();

				auto skel1 = bn1->getSkeleton();
				auto skel2 = bn2->getSkeleton();

				// if(skel1 == mObstacle && skel2->getName() == "humanoid")
				// 	mSimCharacter->applyForceMSD(bn2, bn2->getTransform().inverse()*(contact.point),-10.0*contact.force);
				// else if(skel1->getName() == "humanoid" && skel2 == mObstacle)
				// 	mSimCharacter->applyForceMSD(bn1, bn1->getTransform().inverse()*(contact.point), 10.0*contact.force);

				if(bn1->getName().find("Foot") != std::string::npos)
					continue;
				else if(bn2->getName().find("Foot") != std::string::npos)
					continue;

				if(skel1->getName() == "humanoid" && skel2->getName() == "ground"){
					mContactEOE = true;
					break;
				}

				if(skel1->getName() == "ground" && skel2->getName() == "humanoid"){
					mContactEOE = true;
					break;
				}

			}
			auto fss = mSimCharacter->getForceSensors();
			for(auto fs: fss)
				fs->step();
		}
		
	}
	
	auto motion = mCurrentMotion;
	Eigen::Vector3d position = motion->getPosition(0);
	Eigen::MatrixXd rotation = motion->getRotation(0);
	Eigen::Vector3d linear_velocity = motion->getLinearVelocity(0);
	Eigen::MatrixXd angular_velocity = motion->getAngularVelocity(0);
	rotation = mSimCharacter->addRotMSD(rotation);
	mKinCharacter->setPose(position, rotation, linear_velocity, angular_velocity);
	// this->applyRootSupportForce();
	if(mEnableGoal)
	{
		this->recordGoal();
		this->updateGoal();
	}
	
	this->recordState();

	
	// mPrevPositions = mSimCharacter->getSkeleton()->getPositions();

	mPrevPositions2 = mPrevPositions;
	mPrevPositions = mSimCharacter->getSkeleton()->getPositions();
	// mPrevVelocities = mSimCharacter->getSkeleton()->getVelocities();
	mPrevCOM = mSimCharacter->getSkeleton()->getCOM();
	mPrevMSDStates = mSimCharacter->saveStateMSD();
	// if(mEnableGoal)
	// 	mPrevDoorAngle = mDoor->getPositions()[0];
	mElapsedFrame++;
	mFrame++;
}
void
Environment::
resetGoal()
{
	// Eigen::Vector3d pos = mCartesianMSDSystem->getPosition();
	// Eigen::Vector3d com_vel = (mSimCharacter->getSkeleton()->getCOM() - mPrevCOM)*mControlHz;
	// com_vel[1] = 0.0;


	// mTargetFrame = 36;
	// mTargetDoorAngle = dart::math::Random::uniform<double>(-0.5*M_PI, 0.5*M_PI);
	// Eigen::Vector3d z_sim = mSimCharacter->getReferenceTransform().linear().col(2);
	// Eigen::Vector3d sim_com_vel = mSimCharacter->getSkeleton()->getCOMLinearVelocity();
	// double sign = z_sim.dot(sim_com_vel);
	// if(sign>0)
	// 	mTargetDoorAngle = -0.5*M_PI;
	// else
	// 	mTargetDoorAngle = 0.5*M_PI;
	// std::vector<double> pool = {-0.5*M_PI,0.5*M_PI,0.0};
	// auto it = std::find(mMotions.begin(), mMotions.end(), mCurrentMotion);
	// int idx = std::distance(mMotions.begin(), it);
	// if(idx == 1)
	// 	mTargetDoorAngle = 0.5*M_PI;
	// else
	// 	mTargetDoorAngle = -0.5*M_PI;

	// Eigen::Isometry3d T_ref = mSimCharacter->getReferenceTransform();
	// Eigen::Matrix3d R_ref = T_ref.linear();
	// Eigen::AngleAxisd aa_ref(R_ref);
	// double heading = aa_ref.angle()*aa_ref.axis()[1];
	// mTargetHeading = heading;

	// // mTargetSpeed = dart::math::Random::uniform<double>(mTargetSpeedMin, mTargetSpeedMax);
	// Eigen::Vector3d com_vel = mSimCharacter->getSkeleton()->getCOMLinearVelocity();
	// com_vel[1] =0.0;
	// mTargetSpeed = std::max(1.0, com_vel.norm());

	// mTargetHeading = dart::math::Random::uniform<double>(-M_PI, M_PI);
	// mTargetSpeed = dart::math::Random::uniform(mTargetSpeedMin, mTargetSpeedMax);
}
void
Environment::
updateGoal()
{
	// if(mRewardGoal>0.6)
	// 	mTargetFrame = mTargetFrame>0?0:36;
	// mTargetFrame = 36;
	// double error_door_angle = mTargetDoorAngle - mDoor->getPositions()[0];
	// if(std::abs(error_door_angle)<0.2)
	// {
	// 	std::vector<double> pool = {-0.5*M_PI,0.5*M_PI,0.0};
	// 	mTargetDoorAngle = dart::math::Random::uniform<double>(-0.5*M_PI, 0.5*M_PI);
	// }
	// mTargetDoorAngle = 1.7*std::sin(mElapsedFrame/50.0);

	// bool sharp_turn = dart::math::Random::uniform<double>(0.0, 1.0)<mSharpTurnProb?true:false;
	// double delta_heading = 0;
	// if(sharp_turn)
	// 	delta_heading = dart::math::Random::uniform<double>(-M_PI, M_PI);
	// else
	// 	delta_heading = dart::math::Random::normal<double>(0.0, mMaxHeadingTurnRate);
	// mTargetHeading += delta_heading;

	// bool change_speed = dart::math::Random::uniform<double>(0.0, 1.0)<mSpeedChangeProb?true:false;
	// if(change_speed)
	// 	mTargetSpeed = dart::math::Random::uniform(mTargetSpeedMin, mTargetSpeedMax);
}
void
Environment::
recordGoal()
{

	Eigen::Isometry3d T_ref = mSimCharacter->getReferenceTransform();
	Eigen::Matrix3d R_ref = T_ref.linear();

	Eigen::Vector3d target_com_vel = mSimCharacter->getCartesianMSD()->getPosition();
	Eigen::Vector3d com_vel = (mSimCharacter->getSkeleton()->getCOM() - mPrevCOM)*mControlHz;
	com_vel[1] = 0.0;

	// mStateGoal = R_ref.transpose()*(com_vel-target_com_vel);
	// mStateGoal = (com_vel-target_com_vel);
	mStateGoal.resize(0);

	// mStateGoal.setZero();
	// std::cout<<mStateGoal.transpose()<<std::endl;

	mRewardGoal = 1.0;

	if(target_com_vel.norm()>2e-1)
	{
		double vel = (target_com_vel - MathUtils::projectOnVector(com_vel, target_com_vel)).norm();
		// std::cout<<vel<<std::endl;
		// std::cout<<target_com_vel.norm()<<" "<<MathUtils::projectOnVector(com_vel, target_com_vel).norm()<<std::endl;
		// if(vel > 0.0){
			// mRewardGoal = 2*std::exp(-0.25*vel*vel) - 1.0;
		mRewardGoal = std::exp(-1.5*vel*vel);
		// }
	}
	else
	{
		com_vel[1] = 0.0;
		double vel = com_vel.norm();
		mRewardGoal = std::exp(-1.5*vel*vel);
	}
	// std::cout<<mRewardGoal<<std::endl;
	mRewardGoals.emplace_back(mRewardGoal);
	// auto motion = mCurrentMotion;
	// Eigen::Vector3d position = motion->getPosition(mTargetFrame);
	// Eigen::MatrixXd rotation = motion->getRotation(mTargetFrame);
	// Eigen::Vector3d linear_velocity = motion->getLinearVelocity(mTargetFrame);
	// Eigen::MatrixXd angular_velocity = motion->getAngularVelocity(mTargetFrame);
	// mStateGoal = this->computePoseDiffState(position, rotation, linear_velocity, angular_velocity);
	// mRewardGoal = this->computePoseDiffReward(position, rotation, linear_velocity, angular_velocity);
	// mRewardGoals.emplace_back(mRewardGoal);

	// double error_door_angle = mTargetDoorAngle - mDoor->getPositions()[0];

	// Eigen::Isometry3d T_door = mDoor->getBodyNode(1)->getTransform();

	// Eigen::Vector3d p_left_hand = mSimCharacter->getSkeleton()->getBodyNode("LeftHand")->getCOM();
	// p_left_hand = T_door.inverse()*p_left_hand;
	// Eigen::VectorXd g(3);
	// g<<p_left_hand[0],p_left_hand[2],error_door_angle;
	// mStateGoal = g;

	// if(mContactEOE){
	// 	mRewardGoal = 0.0;
	// 	return;
	// }

	// double door_dir = error_door_angle>0?1:-1;
	// double door_vel = (mDoor->getPositions()[0] - mPrevDoorAngle)*mControlHz;
	// double proj_vel = door_dir*door_vel;
	// double err = std::max(0.5 - proj_vel, 0.0);
	// double r_dir = std::exp(-4.0*err*err); 

	// double r_angle = std::exp(-error_door_angle*error_door_angle);
	// mRewardGoal = 0.3*r_dir + 0.7*r_angle;
	// if(std::abs(error_door_angle)<0.2)
	// 	r_dir = 1.0;
	// mRewardGoals.emplace_back(mRewardGoal);
	// Eigen::Vector3d com_vel = (mSimCharacter->getSkeleton()->getCOM() - mPrevCOM)*mControlHz;
	// com_vel[1] = 0.0;

	// Eigen::Vector3d target_direction = R_target.col(2);
	// // Eigen::Vector3d target_direction(std::cos(mTargetHeading), 0.0, -std::sin(mTargetHeading));
	// double proj_vel = target_direction.dot(com_vel);
	// mRewardGoal = 0.0;
	// if(proj_vel > 0.0)
	// {
	// 	double err = std::max(mTargetSpeed - proj_vel, 0.0);
	// 	mRewardGoal = std::exp(-0.25*err*err);
	// }
}

double
Environment::
getRewardGoal()
{
	return mRewardGoal;
}


const Eigen::VectorXd&
Environment::
getStateGoal()
{
	return mStateGoal;
}

const Eigen::VectorXd&
Environment::
getState()
{
	return mState;
}
const Eigen::VectorXd&
Environment::
getStateAMP()
{
	return mStateAMP;
}
void
Environment::
recordState()
{
	Eigen::VectorXd state_sim = MathUtils::ravel(mSimCharacter->getState());
	Eigen::VectorXd state_msd = MathUtils::ravel(mSimCharacter->getStateMSD(mSimCharacter->getReferenceTransform()));
	Eigen::VectorXd state = MathUtils::ravel(state_sim,state_msd);
	if(mEnableGoal)
	{
		Eigen::VectorXd goal = this->getStateGoal();
		mState = Eigen::VectorXd(state.rows() + goal.rows());
		mState<<state, goal;	
	}
	else
		mState = state;	
	
	auto save_state = mKinCharacter->saveState();
	

	Eigen::VectorXd prev_velocities = mSimCharacter->computeAvgVelocity(mPrevPositions2, mPrevPositions, 1.0/mControlHz);
	mKinCharacter->getSkeleton()->setPositions(mPrevPositions);
	mKinCharacter->getSkeleton()->setVelocities(prev_velocities);
	mKinCharacter->restoreStateMSD(mPrevMSDStates);
	
	Eigen::VectorXd s = mKinCharacter->getStateAMP();

	Eigen::VectorXd velocities = mSimCharacter->computeAvgVelocity(mPrevPositions, mSimCharacter->getSkeleton()->getPositions(), 1.0/mControlHz);
	mKinCharacter->getSkeleton()->setPositions(mSimCharacter->getSkeleton()->getPositions());
	mKinCharacter->getSkeleton()->setVelocities(velocities);
	mKinCharacter->restoreStateMSD(mSimCharacter->saveStateMSD());
	Eigen::VectorXd s1 = mKinCharacter->getStateAMP();
	mKinCharacter->restoreState(save_state);
	mStateAMP.resize(s.rows() + s1.rows());
	mStateAMP<<s, s1;
	
	// int n = mStateAMPExpert.rows();
	// int m = mStateAMPExpert.cols();

	// std::cout<<s1.transpose()<<std::endl<<std::endl;
	// std::cout<<mStateAMPExpert.row(0)<<std::endl;
	// std::cout<<(mStateAMPExpert.block(0,0,1,m))<<std::endl;
	// if(n==0)
	// 	return;
	// int min_idx = -1;
	// double min_val = 1e6;
	// for(int i =0;i<n;i++)
	// {
	// 	double val = (s1- (mStateAMPExpert.block(i,0,1,m/2).transpose())).norm();
	// 	if(val<min_val)
	// 	{
	// 		min_idx = i;
	// 		min_val = val;
	// 	}
	// 	if(i==0)
	// 		i += mMotions[0]->getNumFrames()-2;
	// }
	// int motion=0, frame = min_idx;
	// int cursor = 0;
	// for(auto m: mMotions)
	// {
	// 	int nf = m->getNumFrames();
	// 	if( (frame - (nf - 1))<0)
	// 		break;
	// 	frame -= nf - 1;
	// 	motion++;
	// }

	// std::cout<<motion<<"th : "<<frame<<std::endl;
	// auto motion = mCurrentMotion;
	// mFrame %= motion->getNumFrames();
	// Eigen::Vector3d position = motion->getPosition(mFrame);
	// Eigen::MatrixXd rotation = motion->getRotation(mFrame);
	// Eigen::Vector3d linear_velocity = motion->getLinearVelocity(mFrame);
	// Eigen::MatrixXd angular_velocity = motion->getAngularVelocity(mFrame);
	// mKinCharacter->setPose(position, rotation, linear_velocity, angular_velocity);
}


Eigen::MatrixXd
Environment::
getStateAMPExpert()
{
	int total_num_frames = 0;
	int m = this->getDimStateAMP();
	int m2 = m/2;
	int o = 0;
	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		total_num_frames += nf-1;
	}
	Eigen::MatrixXd state_expert(total_num_frames-1,m);

	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		mKinCharacter->setPose(motion->getPosition(0),
							motion->getRotation(0),
							motion->getLinearVelocity(0),
							motion->getAngularVelocity(0));
		Eigen::VectorXd s = mKinCharacter->getStateAMP();

		Eigen::VectorXd s1;
		for(int i=0;i<nf-1;i++)
		{
			mKinCharacter->setPose(motion->getPosition(i+1),
							motion->getRotation(i+1),
							motion->getLinearVelocity(i+1),
							motion->getAngularVelocity(i+1));
			s1 = mKinCharacter->getStateAMP();
			state_expert.row(o+i).head(m2) = s.transpose();
			state_expert.row(o+i).tail(m2) = s1.transpose();
			s = s1;
		}
		o += nf - 1;
	}
	mStateAMPExpert = state_expert;
	return state_expert;
}
bool
Environment::
inspectEndOfEpisode()
{
	if(mContactEOE)
		return true;
	else if(mElapsedFrame>mMaxElapsedFrame)
		return true;

	if(mEnableGoalEOE)
	{
		double r_mean = 0.0;
		const std::vector<double>& rs = mRewardGoals;
		int n = rs.size();
		for(int i=n-mControlHz;i<n;i++){
			r_mean += rs[i];
		}
		r_mean /= (double)mControlHz;
		if(r_mean<0.1)
			return true;
	}

	
	Eigen::Matrix3d R_root = mSimCharacter->getRootTransform().linear();
	Eigen::Quaterniond Q_root(R_root);
	double a = Q_root.w();
	double b = Q_root.vec()[1];
	double theta = 2*std::atan2(b, a);
	Eigen::Matrix3d R_y = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()).toRotationMatrix();
	if(dart::math::logMap(R_y.transpose()*R_root).norm()>0.4)
		return true;
	return false;
}
void
Environment::
updateObstacle()
{
	// if(mObstacleCount>mObstacleFinishCount && mObstacle !=nullptr){
	// 	if(mWorld->hasSkeleton(mObstacle))
	// 		mWorld->removeSkeleton(mObstacle);

	// }
	// mObstacleCount++;
	// Eigen::VectorXd generalized_pos = Eigen::VectorXd::Zero(6);
	// Eigen::VectorXd generalized_vel = Eigen::VectorXd::Zero(6);
	// generalized_pos.tail<3>() = mCurrentTargetHandPos;
	// generalized_vel.tail<3>() = mCurrentTargetHandVel;
	// mObstacle->setPositions(generalized_pos);
	// mObstacle->setVelocities(generalized_vel);	
}
void
Environment::
generateObstacle()
{
	if(mObstacle == nullptr){
		double width = 1.0;
		Eigen::Vector3d c0 = Eigen::Vector3d(-width+0.25, 0.0, 0.32);

		Eigen::Vector3d hand_com = mSimCharacter->getSkeleton()->getBodyNode("RightHand")->getCOM();
		c0 += hand_com;
		c0[1] = 0.0;
		Eigen::Vector3d anchor = hand_com;
		anchor[0] += 0.3;
		anchor[1] += 0.3;
		anchor[2] += 0.3;


		

		mObstacle = DARTUtils::createDoor(c0, 1.0);

		mWorld->addSkeleton(mObstacle);

		// Eigen::VectorXd generalized_pos = Eigen::VectorXd::Zero(6);
		// Eigen::VectorXd generalized_vel = Eigen::VectorXd::Zero(6);
		// generalized_pos.tail<3>() = mCurrentTargetHandPos;
		// generalized_vel.tail<3>() = mCurrentTargetHandVel;
		// mObstacle->setPositions(generalized_pos);
		// mObstacle->setVelocities(generalized_vel);
		mObstacleCount = 0;
		mObstacleFinishCount = dart::math::Random::uniform<int>(30,120);

		mBallConstraint = std::make_shared<dart::constraint::BallJointConstraint>(mSimCharacter->getSkeleton()->getBodyNode("RightHand"),
																				mObstacle->getBodyNode(1),
																				anchor);
		mWorld->getConstraintSolver()->addConstraint(mBallConstraint);

	}
	// if(mWorld->hasSkeleton(mObstacle))
		// return;
	// mWorld->addSkeleton(mObstacle);
	

	// Eigen::Vector3d target = mSimCharacter->getSkeleton()->getCOM();
	// target[1] += 0.3;

	// // Eigen::Vector3d dir = 2.0*Eigen::AngleAxisd(dart::math::Random::uniform<double>(-M_PI,M_PI), Eigen::Vector3d::UnitY()).toRotationMatrix().col(2);
	// Eigen::Vector3d dir(0,0,2);

	// Eigen::Vector3d pos = target - dir;
	// Eigen::Vector3d vel = (target - pos)*4.0;
	// vel[1] += 2.0;

	
	

	
}

void
Environment::
generateTargetHandPos()
{
	if(mTargetHandCount<mTargetHandFinishCount)
		return;
	mTargetHandPos = mTargetHandPos2;
	// mTargetHandPos2 += 0.5*Eigen::Vector3d::Random();
	mTargetHandPos2 += 3.0*Eigen::Vector3d::Random();
	// mTargetHandPos2[2] += 5.0;
	mTargetHandPos2[1] = mTargetHandPos[1];
	if(mTargetHandPos2[1]<1.0)
		mTargetHandPos2[1] = 1.3;


	// Eigen::Vector3d proj;
	// Eigen::Vector3d c = mTargetHandPos2 - mInitHandPos;
	// if(c[0]<-0.15)
	// 	c[0] = -0.15;

	// if(c[1]<0.0)
	// 	c[1] = 0.0;
	// double d = c.norm();
	// double r = 0.5;
	// if(d>r)
	// {
	// 	c = c/d*r;
	// 	mTargetHandPos2 = mInitHandPos + c;
	// }


	mTargetHandCount = 0;
	mTargetHandFinishCount = 60;
	// mTargetHandFinishCount = dart::math::Random::uniform<int>(60,90);
}
void
Environment::
updateTargetHandPos()
{
	double t = (double)mTargetHandCount/(double)mTargetHandFinishCount;
	t = MotionUtils::easeInEaseOut(t);
	mCurrentTargetHandPos = t*mTargetHandPos + (1.0 - t)*mTargetHandPos2;
	t = (double)(mTargetHandCount+1)/(double)mTargetHandFinishCount;
	t = MotionUtils::easeInEaseOut(t);
	mCurrentTargetHandVel = ((t*mTargetHandPos + (1.0 - t)*mTargetHandPos2)-mCurrentTargetHandPos)*mControlHz;
	mTargetHandCount++;
}
void
Environment::
generateObstacleForce()
{
	mObstacleCount = 100000;
	this->updateObstacle();
	this->generateObstacle();
}
void
Environment::
applyRootSupportForce()
{
	Eigen::VectorXd pos = mSimCharacter->getSkeleton()->getPositions();
	Eigen::VectorXd vel = mSimCharacter->getSkeleton()->getVelocities();
	pos.head<6>() = mRestRootPosition;
	vel.head<6>().setZero();
	mSimCharacter->getSkeleton()->setPositions(pos);
	mSimCharacter->getSkeleton()->setVelocities(vel);
}
Eigen::VectorXd
Environment::
computePoseDiffState(const Eigen::Vector3d& position,
			const Eigen::MatrixXd& rotation,
			const Eigen::Vector3d& linear_velocity,
			const Eigen::MatrixXd& angular_velocity)
{
	std::vector<Eigen::VectorXd> state;

	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
	Eigen::Matrix3d R_sim_inv = T_sim_inv.linear();
	
	std::vector<Eigen::Vector3d> state_sim = mSimCharacter->getState();
	int n = state_sim.size();

	state.emplace_back(MathUtils::ravel(state_sim));
	mKinCharacter->setPose(position,rotation,linear_velocity,angular_velocity);

	Eigen::Isometry3d T_kin = mKinCharacter->getReferenceTransform();
	std::vector<Eigen::Vector3d> state_kin = mKinCharacter->getState();
	
	std::vector<Eigen::Vector3d> state_sim_kin_diff(n);
	for(int j=0;j<n;j++)
		state_sim_kin_diff[j] = state_sim[j] - state_kin[j];

	state.emplace_back(MathUtils::ravel(state_kin));
	state.emplace_back(MathUtils::ravel(state_sim_kin_diff));

	Eigen::VectorXd s = MathUtils::ravel(state);

	return s;
}
double
Environment::
computePoseDiffReward(const Eigen::Vector3d& position,
			const Eigen::MatrixXd& rotation,
			const Eigen::Vector3d& linear_velocity,
			const Eigen::MatrixXd& angular_velocity)
{
	Eigen::VectorXd joint_weights = mSimCharacter->getJointWeights();

	std::map<std::string,double> rewards;

	auto state_sim_body = mSimCharacter->getStateBody();
	auto state_sim_joint = mSimCharacter->getStateJoint();
	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
	Eigen::Vector3d sim_com = mSimCharacter->getSkeleton()->getCOM();
	Eigen::Vector3d sim_com_vel = mSimCharacter->getSkeleton()->getCOMLinearVelocity();
	Eigen::MatrixXd sim_body_p, sim_body_R, sim_body_v, sim_body_w, sim_joint_p, sim_joint_v;
	
	sim_body_p = state_sim_body["ps"];
	sim_body_R = state_sim_body["Rs"];
	sim_body_v = state_sim_body["vs"];
	sim_body_w = state_sim_body["ws"];

	sim_joint_p = state_sim_joint["p"];
	sim_joint_v = state_sim_joint["v"];

	mKinCharacter->setPose(position,rotation,linear_velocity,angular_velocity);
	auto state_kin_body = mKinCharacter->getStateBody();
	auto state_kin_joint = mKinCharacter->getStateJoint();
	Eigen::Isometry3d T_kin = mKinCharacter->getReferenceTransform();
	Eigen::Isometry3d T_kin_inv = T_kin.inverse();
	Eigen::Vector3d kin_com = mKinCharacter->getSkeleton()->getCOM();
	Eigen::Vector3d kin_com_vel = mKinCharacter->getSkeleton()->getCOMLinearVelocity();
	Eigen::MatrixXd kin_body_p, kin_body_R, kin_body_v, kin_body_w, kin_joint_p, kin_joint_v;
	
	kin_body_p = state_kin_body["ps"];
	kin_body_R = state_kin_body["Rs"];
	kin_body_v = state_kin_body["vs"];
	kin_body_w = state_kin_body["ws"];

	kin_joint_p = state_kin_joint["p"];
	kin_joint_v = state_kin_joint["v"];

	int n = mSimCharacter->getSkeleton()->getNumBodyNodes();

	double error_pos=0.0, error_vel=0.0, error_ee=0.0, error_root=0.0, error_com=0.0;
	// pos error
	Eigen::MatrixXd diff_pos = DARTUtils::computeDiffPositions(sim_joint_p, kin_joint_p);
	Eigen::MatrixXd diff_vel = sim_joint_v - kin_joint_v;
	Eigen::MatrixXd lb_vel = Eigen::MatrixXd::Constant(diff_vel.rows(),diff_vel.cols(), -1.0);
	Eigen::MatrixXd ub_vel = Eigen::MatrixXd::Constant(diff_vel.rows(),diff_vel.cols(), 1.0);
	diff_vel = diff_vel.cwiseMax(lb_vel).cwiseMin(ub_vel);

	for(int i=1;i<n;i++)
	{
		error_pos += joint_weights[i]*(diff_pos.col(i).dot(diff_pos.col(i)));
		error_vel += joint_weights[i]*(diff_vel.col(i).dot(diff_vel.col(i)));
	}

	auto ees = mSimCharacter->getEndEffectors();
	for(int i=0;i<ees.size();i++)
	{
		int idx = ees[i]->getIndexInSkeleton();
		Eigen::Vector3d sim_ee = sim_body_p.col(idx);
		Eigen::Vector3d kin_ee = kin_body_p.col(idx);
		
		Eigen::Vector3d diff_ee_local = T_sim_inv*sim_ee - T_kin_inv*kin_ee;
		error_ee += diff_ee_local.dot(diff_ee_local);
	}

	Eigen::Vector3d diff_com = T_sim_inv*sim_com - T_kin_inv*kin_com;
	Eigen::Vector3d diff_com_vel = T_sim_inv.linear()*sim_com_vel - T_kin_inv.linear()*kin_com_vel;
	
	error_com = 1.0*diff_com.dot(diff_com) + 
				0.1*diff_com_vel.dot(diff_com_vel);

	double r_pos = std::exp(-1.0*error_pos);
	double r_vel = std::exp(-0.1*error_vel);
	double r_ee = std::exp(-error_ee);
	double r_com = std::exp(-error_com);

	return r_pos*r_vel*r_ee*r_com;
}
double
Environment::
computeGroundHeight()
{
	double y = 1e6;

	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		for(int i=0;i<nf;i++)
		{
			mKinCharacter->setPose(motion->getPosition(i),
							motion->getRotation(i),
							motion->getLinearVelocity(i),
							motion->getAngularVelocity(i));

			y= std::min(y, mKinCharacter->getSkeleton()->getBodyNode("LeftFoot")->getCOM()[1]);
			y= std::min(y, mKinCharacter->getSkeleton()->getBodyNode("RightFoot")->getCOM()[1]);
		}
	}
	
	float dy = dynamic_cast<const BoxShape*>(mKinCharacter->getSkeleton()->getBodyNode("LeftFoot")->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	return y - dy;
}
Eigen::MatrixXd
Environment::
getActionSpace()
{
	Eigen::MatrixXd action_space = Eigen::MatrixXd::Ones(this->getDimAction(), 2);
	int n = mSimCharacter->getSkeleton()->getNumDofs();

	action_space.col(0) *= -M_PI*2; // Lower
	action_space.col(1) *=  M_PI*2; // Upper

	return action_space;
}
Eigen::VectorXd
Environment::
getActionWeight()
{
	Eigen::VectorXd action_weight = Eigen::VectorXd::Ones(this->getDimAction());
	int n = mSimCharacter->getSkeleton()->getNumDofs();
	// action_weight *= 0.3;
	return action_weight;
}
Eigen::VectorXd
Environment::
convertToRealActionSpace(const Eigen::VectorXd& a_norm)
{
	Eigen::VectorXd a_real;
	Eigen::VectorXd lo = mActionSpace.col(0), hi =  mActionSpace.col(1);
	a_real = dart::math::clip<Eigen::VectorXd, Eigen::VectorXd>(a_norm, lo, hi);
	a_real = mActionWeight.cwiseProduct(a_real);
	return a_real;
}




















































































// BVH* bvh_lb = new BVH(std::string(ROOT_DIR)+"/data/bvh/walk_long.bvh");
	// BVH* bvh_ub = new BVH(std::string(ROOT_DIR)+"/data/bvh/open_door_long.bvh");
	// // for(int i=0;i<start_frames.size();i++)
	// {
	// 	Motion* motion = MotionUtils::blendUpperLowerMotion(bvh_lb, bvh_lb, 0, 120);

	// 	mMotions.push_back(motion);
	// 	// if(i==0)
	// 	// {
	// 	mSimCharacter->buildBVHIndices(bvh_lb->getNodeNames());
	// 	mKinCharacter->buildBVHIndices(bvh_lb->getNodeNames());
	// 	// }
	// }
	// mCurrentMotion = new Motion(bvh_lb);
	// {
	// 	BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/walk_long.bvh");
	// 	Motion* motion = new Motion(bvh);

	// 	int nf = bvh->getNumFrames();
	// 	for(int i=2910;i<3150;i++)
	// 		motion->append(bvh->getPosition(i), bvh->getRotation(i), false);
	// 	motion->computeVelocity();
	// 	mMotions.push_back(motion);
	// 	mLowerBodyMotions.push_back(motion);
		
	// }

	// {
	// 	BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/walk_long.bvh");
	// 	Motion* motion = new Motion(bvh);

	// 	for(int i=100;i<300;i++)
	// 		motion->append(bvh->getPosition(i), bvh->getRotation(i), false);
	// 	motion->computeVelocity();
	// 	mMotions.push_back(motion);
	// 	mLowerBodyMotions.push_back(motion);
	// }

	// {
	// 	BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/open_door_long.bvh");
	// 	Motion* motion = new Motion(bvh);

	// 	int nf = bvh->getNumFrames();
	// 	for(int i=0;i<nf;i++)
	// 		motion->append(bvh->getPosition(i), bvh->getRotation(i), false);
	// 	motion->computeVelocity();
	// 	mMotions.push_back(motion);

	// 	mUpperBodyMotions.push_back(motion);
	// 	mSimCharacter->buildBVHIndices(bvh->getNodeNames());
	// 	mKinCharacter->buildBVHIndices(bvh->getNodeNames());
	// }

// if(0)
// 	{
// 		Eigen::Vector3d pos0 = motion->getPosition(0);
// 		Eigen::MatrixXd rot0 = motion->getRotation(0);

// 		Eigen::Vector3d pos1 = motion->getPosition(mFrame);
// 		Eigen::MatrixXd rot1 = motion->getRotation(mFrame);

// 		Eigen::Vector3d pos2 = motion->getPosition(mFrame+1);
// 		Eigen::MatrixXd rot2 = motion->getRotation(mFrame+1);

// 		Eigen::Isometry3d T0 = MotionUtils::getReferenceTransform(pos0, rot0);
// 		Eigen::Isometry3d T1 = MotionUtils::getReferenceTransform(pos1, rot1);
// 		Eigen::Isometry3d T2 = MotionUtils::getReferenceTransform(pos2, rot2);

// 		Eigen::Isometry3d T1_inv = T1.inverse();

// 		Eigen::Isometry3d T10 = T0*T1_inv;

// 		pos1 = T10*pos1;
// 		rot1.block<3,3>(0,0) = T10.linear()*rot1.block<3,3>(0,0);

// 		pos2 = T10*pos2;
// 		rot2.block<3,3>(0,0) = T10.linear()*rot2.block<3,3>(0,0);
// 		// mCurrentMotion->clear();
// 		// mCurrentMotion->append(pos1, rot1, false);
// 		// mCurrentMotion->append(pos2, rot2, false);
// 		// mCurrentMotion->computeVelocity();
// 	}

// Environment::
// Environment()
// 	:mControlHz(30),
// 	mSimulationHz(300),
// 	mElapsedFrame(0),
// 	mStartFrame(0),
// 	mCurrentFrame(0),
// 	mMaxElapsedFrame(300),
// 	mWorld(std::make_shared<World>()),
// 	mEvent(nullptr),
// 	mState0Dirty(true),
// 	mState1Dirty(true),
// 	mKinematic(false),
// 	mDoorConstraint(nullptr)
// {
// 	dart::math::Random::generateSeed(true);
// 	mSimCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel_c.xml");
// 	mKinCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel_c.xml");

// 	BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/shaking_hand2.bvh");
// 	Motion* motion = new Motion(bvh);

// 	int nf = bvh->getNumFrames();
// 	Eigen::Isometry3d T_oppo = Eigen::Isometry3d::Identity();
// 	T_oppo.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix();
// 	T_oppo.translation() = Eigen::Vector3d(-0.1,0,2.2);
// 	for(int j=0;j<5;j++)
// 	for(int i=0;i<nf;i++){

// 		motion->append(bvh->getPosition(i), bvh->getRotation(i), false);
// 		mOpponentHandTrajectory.push_back(T_oppo*(bvh->forwardKinematics(bvh->getPosition(i), bvh->getRotation(i),bvh->getNodeIndex("simRightHand"))[0].translation()));
// 		// mOpponentHandTrajectory.push_back(bvh->getPosition(i));
// 		// mOpponentHandTrajectory.push_back(bvh->forwardKinematics(bvh->getPosition(i), bvh->getRotation(i),bvh->getNodeIndex("simRightFoot"))[0].translation());
// 	}

	
// 	// for(int i=0;i<mMaxElapsedFrame-nf+50;i++)
// 	// 	motion->append(bvh->getPosition(nf-1), bvh->getRotation(nf-1), false);
// 	motion->computeVelocity();

// 	mSimCharacter->buildBVHIndices(bvh->getNodeNames());
// 	mKinCharacter->buildBVHIndices(bvh->getNodeNames());
// 	mSimCharacter->setBaseMotionAndCreateMSDSystem(motion);

// 	mGround = DARTUtils::createGround(0.0);
	
// 	mDoor = nullptr;	

// 	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
// 	mSimCharacter->getSkeleton()->enableSelfCollisionCheck();
// 	mWorld->addSkeleton(mSimCharacter->getSkeleton());
// 	mWorld->addSkeleton(mGround);
// 	mWorld->setTimeStep(1.0/(double)mSimulationHz);
// 	mWorld->setGravity(Eigen::Vector3d(0,-9.81,0.0));

// 	mImitationFrameWindow.emplace_back(0);
// 	mImitationFrameWindow.emplace_back(1);
// 	// mImitationFrameWindow.emplace_back(5);

// 	mWeightPos = 4.0;
// 	mWeightVel = 0.1;
// 	mWeightEE = 1.0;
// 	mWeightRoot = 1.0;
// 	mWeightCOM = 1.0;

// 	mRewards.insert(std::make_pair("r_pos", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r_vel", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r_ee", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r_root", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r_com", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r_imit", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r_force", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r_force_orig", std::vector<double>()));
// 	mRewards.insert(std::make_pair("r", std::vector<double>()));

// 	int n = motion->getNumFrames() - mMaxElapsedFrame - 10;
// 	int stride = 30;

// 	// mInitialStateDistribution = new Distribution1D<int>(n/stride, [=](int i)->int{return i*stride;}, 0.7);
// 	mInitialStateDistribution = new Distribution1D<int>(1, [=](int i){return i;}, 0.7);
// 	mStartFrame = mInitialStateDistribution->sample();
	
// 	mEvent = new ObstacleEvent(this, mSimCharacter);
// 	mDoorConstraintOn = false;
// 	this->reset(12);

// 	mActionSpace0 = this->getActionSpace0();
// 	mActionWeight0 = this->getActionWeight0();
// 	mActionSpace1 = this->getActionSpace1();
// 	mActionWeight1 = this->getActionWeight1();


// }


// void
// Environment::
// reset(int frame)
// {
// 	mElapsedFrame = 0;
// 	Eigen::Map<Eigen::ArrayXd> rs(mRewards["r"].data(), mRewards["r"].size());
// 	mInitialStateDistribution->update(rs.sum());
// 	if(frame<0)
// 		mStartFrame = mInitialStateDistribution->sample();
// 	else
// 		mStartFrame = frame;

// 	mContactEOE = false;
// 	mCurrentFrame = mStartFrame;
// 	mEvent->reset();
// 	mSimCharacter->resetMotion(mStartFrame);

// 	mSimCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
// 						mSimCharacter->getRotation(mCurrentFrame),
// 						mSimCharacter->getLinearVelocity(mCurrentFrame),
// 						mSimCharacter->getAngularVelocity(mCurrentFrame));
// 	if(mDoorConstraint == nullptr)
// 	{
// 		int prev = std::max(0, mCurrentFrame-1);
// 		int next = std::min((int)mOpponentHandTrajectory.size()-1, mCurrentFrame+1);
// 		double dt = (next-prev)/30.0;
// 		Eigen::Vector3d pos = mOpponentHandTrajectory[mCurrentFrame];
// 		Eigen::Vector3d vel = (mOpponentHandTrajectory[next]-mOpponentHandTrajectory[prev])/dt;
// 		Eigen::Vector6d p_o = Eigen::compose(Eigen::Vector3d::Zero(),pos);
// 		Eigen::Vector6d v_o = Eigen::compose(Eigen::Vector3d::Zero(),vel);
// 		auto obstacle = dynamic_cast<ObstacleEvent*>(mEvent)->getObstacle();
// 		obstacle->setPositions(p_o);
// 		obstacle->setVelocities(v_o);
// 		mDoorConstraint =
// 	        std::make_shared<dart::constraint::BallJointConstraint>(mSimCharacter->getSkeleton()->getBodyNode("RightHand"),
// 					obstacle->getBodyNode(0),mSimCharacter->getSkeleton()->getBodyNode("RightHand")->getCOM());
// 	}

// 	// if(mDoor ==nullptr)
// 	// {
// 	// 	double door_size = 2.0;
// 	// 	Eigen::Vector3d door_hinge_pos(0.27-door_size,1.01,0.53);
// 	// 	mDoor = DARTUtils::createDoor(door_hinge_pos, door_size);
// 	// 	// mDoorConstraint =
// 	//  //        std::make_shared<dart::constraint::BallJointConstraint>(mSimCharacter->getSkeleton()->getBodyNode("LeftHand"), 
// 	// 	// 			mDoor->getBodyNode(1),mSimCharacter->getSkeleton()->getBodyNode("LeftHand")->getCOM());

// 	//     // mWorld->getConstraintSolver()->addConstraint(mDoorConstraint);	
// 	// 	// mWorld->addSkeleton(mDoor);
// 	// 	// mDoorConstraint->setErrorAllowance(0.9);
// 	// 	// mDoorConstraint->setErrorReductionParameter(1.0);
// 	// 	// mDoorConstraint->setConstraintForceMixing(0.0);
// 	// }
// 	// else
// 	// {
// 	// 	Eigen::VectorXd p = mDoor->getPositions();
// 	// 	Eigen::VectorXd v = mDoor->getVelocities();
// 	// 	p.setZero();
// 	// 	v.setZero();
// 	// 	mDoor->setPositions(p);
// 	// 	mDoor->setVelocities(v);
// 	// 	// if(mDoorConstraintOn == false)
// 	// 		// mWorld->getConstraintSolver()->addConstraint(mDoorConstraint);
// 	// }
// 	mAMPStates.clear();
// 	if(mDoorConstraintOn == true)
// 		mWorld->getConstraintSolver()->removeConstraint(mDoorConstraint);
// 	mDoorConstraintOn = false;
	


// 	mSimCharacter->getSkeleton()->clearConstraintImpulses();
// 	mSimCharacter->getSkeleton()->clearInternalForces();
// 	mSimCharacter->getSkeleton()->clearExternalForces();
// 	auto fss = mSimCharacter->getForceSensors();
// 	for(auto fs: fss)
// 		fs->reset();
	
	

// 	mKinCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
// 						mSimCharacter->getRotation(mCurrentFrame),
// 						mSimCharacter->getLinearVelocity(mCurrentFrame),
// 						mSimCharacter->getAngularVelocity(mCurrentFrame));

// 	for(auto r : mRewards){
// 		mRewards[r.first].clear();
// 		mRewards[r.first].reserve(mMaxElapsedFrame);
// 	}
// 	mState0Dirty = true;
// 	mState1Dirty = true;

// 	mPredefinedAction = 100.0*Eigen::AngleAxisd(dart::math::Random::uniform<double>(-M_PI,M_PI), Eigen::Vector3d::UnitY()).toRotationMatrix()*Eigen::Vector3d::UnitZ();

// }

// void
// Environment::
// step(const Eigen::VectorXd& _action)
// {
// 	if(mElapsedFrame == 65){
// 		// mDoorConstraintOn = false;
// 		// mWorld->getConstraintSolver()->removeConstraint(mDoorConstraint);
// 	}
// 	double alpha = dart::math::Random::uniform<double>(0.0,1.0);
	
// 	Eigen::VectorXd action = this->convertToRealActionSpace0(_action);
// 	// Eigen::VectorXd action1 = this->convertToRealActionSpace1(_action1);

// 	int num_sub_steps = mSimulationHz/mControlHz;
// 	// action1[0] = mPredefinedAction[0];
// 	// action1[1] = mPredefinedAction[1];
// 	// action1[2] = mPredefinedAction[2];
// 	// action1.setZero();
// 	// if(mElapsedFrame == 25 ||
// 	// 	mElapsedFrame == 26 ||
// 	// 	mElapsedFrame == 27 ||
// 	// 	mElapsedFrame == 28 ||
// 	// 	mElapsedFrame == 29){
// 	// 	action1[0] = -1000.0;
// 	// 	action1[2] = -1000.0;
// 	// }
	

// 	int sf = mCurrentFrame%82;
// 	if(sf >= 12 && sf<20 && mDoorConstraintOn == false)
// 	{
// 		double distance = (mOpponentHandTrajectory[sf] - mSimCharacter->getSkeleton()->getBodyNode("RightHand")->getCOM()).norm();
// 		if(distance<1e-1){
// 			mWorld->getConstraintSolver()->addConstraint(mDoorConstraint);
// 			mDoorConstraintOn = true;	
// 		}
		
// 	}
// 	else if( sf == 52)
// 	{
// 		mWorld->getConstraintSolver()->removeConstraint(mDoorConstraint);
// 		mDoorConstraintOn = false;
// 	}
// 	// if(sf<52 && sf>14){
		
// 	// 	if(mDoorConstraintOn == false)
// 	// 		mWorld->getConstraintSolver()->addConstraint(mDoorConstraint);
// 	// 	mDoorConstraintOn = true;

		
// 	// }
// 	// else{
// 	// 	if(mDoorConstraintOn == true)
// 	// 		mWorld->getConstraintSolver()->removeConstraint(mDoorConstraint);
// 	// 	mDoorConstraintOn = false;
// 	// }
// 	Eigen::VectorXd action1 = Eigen::VectorXd::Zero(0);
// 	mSimCharacter->stepMotion(action1);
// 	mKinCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
// 							mSimCharacter->getRotation(mCurrentFrame),
// 							mSimCharacter->getLinearVelocity(mCurrentFrame),
// 							mSimCharacter->getAngularVelocity(mCurrentFrame));
// 	Eigen::MatrixXd base_rot = mSimCharacter->getRotation(mCurrentFrame);
// 	Eigen::MatrixXd base_ang_vel = mSimCharacter->getAngularVelocity(mCurrentFrame);

// 	auto target_pv = mSimCharacter->computeTargetPosAndVel(base_rot, action, base_ang_vel);
// 	mEvent->call();
// 	auto fss = mSimCharacter->getForceSensors();
// 	auto obstacle = dynamic_cast<ObstacleEvent*>(mEvent)->getObstacle();
// 	int prev = std::max(0, mCurrentFrame-1);
// 	int next = std::min((int)mOpponentHandTrajectory.size()-1, mCurrentFrame+1);
// 	double dt = (next-prev)/30.0;
// 	Eigen::Vector3d pos = mOpponentHandTrajectory[mCurrentFrame];
// 	Eigen::Vector3d vel = (mOpponentHandTrajectory[next]-mOpponentHandTrajectory[prev])/dt;
// 	dynamic_cast<ObstacleEvent*>(mEvent)->setLinearPosition(pos);
// 	dynamic_cast<ObstacleEvent*>(mEvent)->setLinearVelocity(vel);
// 	Eigen::Vector6d p_o = Eigen::compose(Eigen::Vector3d::Zero(),dynamic_cast<ObstacleEvent*>(mEvent)->getLinearPosition());
// 	Eigen::Vector6d v_o = Eigen::compose(Eigen::Vector3d::Zero(),dynamic_cast<ObstacleEvent*>(mEvent)->getLinearVelocity());

// 	for(int i=0;i<num_sub_steps;i++)
// 	{
// 		mSimCharacter->actuate(target_pv.first, target_pv.second);
// 		if(mKinematic)
// 		mSimCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
// 								mSimCharacter->getRotation(mCurrentFrame),
// 								mSimCharacter->getLinearVelocity(mCurrentFrame),
// 								mSimCharacter->getAngularVelocity(mCurrentFrame));
// 		obstacle->setPositions(p_o);
// 		obstacle->setVelocities(v_o);
// 		// if(!mKinematic) mWorld->step();	
// 		mWorld->step();	
// 		auto cr = mWorld->getConstraintSolver()->getLastCollisionResult();

// 		for (auto j = 0u; j < cr.getNumContacts(); ++j)
// 		{
// 			auto contact = cr.getContact(j);

// 			auto shapeFrame1 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject1->getShapeFrame());
// 			auto shapeFrame2 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject2->getShapeFrame());

// 			std::string name1 = shapeFrame1->asShapeNode()->getBodyNodePtr()->getSkeleton()->getName();
// 			std::string name2 = shapeFrame2->asShapeNode()->getBodyNodePtr()->getSkeleton()->getName();

// 			if(name1 == "ground")
// 			{
// 				if(shapeFrame2->asShapeNode()->getBodyNodePtr()->getName().find("Foot") == std::string::npos)
// 					mContactEOE = true;
// 			}
// 			if(name2 == "ground")
// 			{
// 				if(shapeFrame1->asShapeNode()->getBodyNodePtr()->getName().find("Foot") == std::string::npos)
// 					mContactEOE = true;
// 			}
// 			if(name1 == "ground" || name2 == "ground")
// 				continue;

// 			Eigen::Vector3d force = contact.force;
// 			if(name2 == "humanoid")
// 				force = -force;

// 			auto fs = mSimCharacter->getClosestForceSensor(contact.point);
// 			// if(fs!=nullptr)
// 				// fs->addExternalForce(force);
// 		}
// 		if(mDoorConstraintOn)
// 		{
// 			Eigen::Vector3d constraint_force = mSimCharacter->getSkeleton()->getBodyNode("RightHand")->getConstraintImpulse().tail<3>();
// 			constraint_force *= mSimulationHz;
// 			auto fs = mSimCharacter->getClosestForceSensor(mSimCharacter->getSkeleton()->getBodyNode("RightHand")->getCOM());
// 			// fs->addExternalForce(-(constraint_force+action.tail<3>()));
// 			// fs->addExternalForce(-constraint_force);
// 		}
		
// 		for(auto fs: fss)
// 			fs->step();
// 	}

// 	mElapsedFrame++;
// 	mCurrentFrame = mStartFrame + mElapsedFrame;
// 	mState0Dirty = true, mState1Dirty = true;
// }

// Eigen::VectorXd
// Environment::
// getState0()
// {
// 	if(mState0Dirty == false)
// 		return mState0;
// 	std::vector<Eigen::VectorXd> state;

// 	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
// 	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
// 	Eigen::Matrix3d R_sim_inv = T_sim_inv.linear();
	
// 	std::vector<Eigen::Vector3d> state_sim = mSimCharacter->getState();
// 	int n = state_sim.size();

// 	state.emplace_back(MathUtils::ravel(state_sim));
// 	Eigen::VectorXd state_kin_save = mKinCharacter->saveState();
// 	for(int i=0;i<mImitationFrameWindow.size();i++)
// 	{
// 		int frame = mCurrentFrame + mImitationFrameWindow[i];
// 		mKinCharacter->setPose(mSimCharacter->getPosition(frame),
// 						mSimCharacter->getRotation(frame),
// 						mSimCharacter->getLinearVelocity(frame),
// 						mSimCharacter->getAngularVelocity(frame));

// 		Eigen::Isometry3d T_kin = mKinCharacter->getReferenceTransform();
// 		std::vector<Eigen::Vector3d> state_kin = mKinCharacter->getState();
		
// 		std::vector<Eigen::Vector3d> state_sim_kin_diff(n+2);
// 		Eigen::Isometry3d T_sim_kin_diff = T_sim_inv*T_kin;
// 		for(int j=0;j<n;j++)
// 			state_sim_kin_diff[j] = state_sim[j] - state_kin[j];

// 		state_sim_kin_diff[n+0] = T_sim_kin_diff.translation();
// 		state_sim_kin_diff[n+1] = T_sim_kin_diff.linear().col(2);

// 		state.emplace_back(MathUtils::ravel(state_kin));
// 		state.emplace_back(MathUtils::ravel(state_sim_kin_diff));
// 	}

// 	mKinCharacter->restoreState(state_kin_save);
// 	state.emplace_back(this->getState1());
// 	Eigen::VectorXd s = MathUtils::ravel(state);

// 	mState0 = s;
// 	mState0Dirty = false;
// 	return s;
// }
// Eigen::VectorXd
// Environment::
// getState1()
// {
// 	// Eigen::VectorXd state0 = this->getState0();
// 	// if(mState1Dirty==false){
// 	// 	Eigen::VectorXd s(mState0.rows() + mState1.rows());
// 	// 	s<<state0,mState1;
// 	// 	// return s;

// 	// 	return mState1;
// 	// }

// 	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
// 	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
// 	Eigen::Matrix3d R_sim_inv = T_sim_inv.linear();

// 	auto fss = mSimCharacter->getForceSensors();
// 	auto statefs = mSimCharacter->getStateForceSensors();

// 	const auto& ps = statefs["ps"];
// 	const auto& vs = statefs["vs"];
// 	const auto& hps = statefs["hps"];
// 	const auto& hvs = statefs["hvs"];
		
// 	int m = ps.cols();
// 	// Eigen::MatrixXd state(m, 12);
// 	std::vector<Eigen::VectorXd> state;

// 	for(int i=0;i<m;i++){
// 		state.emplace_back(ps.col(i));
// 		state.emplace_back(vs.col(i));
// 		state.emplace_back(hps.col(i));
// 		state.emplace_back(hvs.col(i));
// 	}

// 	// Eigen::Vector3d center = this->getTargetCenter();
	
// 	// // center -= T_sim*((Eigen::Vector3d)ps.col(0));
// 	// center = T_sim_inv*center;
// 	// // std::cout<<center.transpose()<<std::endl;
// 	// state.emplace_back(center);
// 	// state.emplace_back(center-(Eigen::Vector3d)ps.col(0));
// 	// state.emplace_back(mPrevAction1);
// 	mState1 = MathUtils::ravel(state);
// 	mState1Dirty = false;
// 	// std::cout<<mState1.transpose()<<std::endl;


// 	// Eigen::VectorXd s(state0.rows() + mState1.rows());
// 	// s<<state0,mState1;
// 	// return s;
// 	return mState1;
// }

// void
// Environment::
// recordAMPState(const Eigen::VectorXd& pos, const Eigen::VectorXd& vel,
// 			const Eigen::VectorXd& pos_next, const Eigen::VectorXd& vel_next)
// {
// 	std::vector<Eigen::VectorXd> state;

// 	Eigen::VectorXd state_kin_save = mKinCharacter->saveState();

// 	mKinCharacter->getSkeleton()->setPositions(pos);
// 	mKinCharacter->getSkeleton()->setVelocities(vel);
// 	Eigen::VectorXd state = MathUtils::ravel(mKinCharacter->getStateAMP());

// 	mKinCharacter->getSkeleton()->setPositions(pos_next);
// 	mKinCharacter->getSkeleton()->setVelocities(vel_next);
// 	Eigen::VectorXd state_next = MathUtils::ravel(mKinCharacter->getStateAMP());

// 	mKinCharacter->restoreState(state_kin_save);

// 	Eigen::VectorXd ss1(state.rows()+state_next.rows());
// 	ss1<<state, state_next;

// 	mAMPState = ss1;
// }

// std::map<std::string,double>
// Environment::
// getReward()
// {
// 	Eigen::VectorXd joint_weights = mSimCharacter->getJointWeights();

// 	std::map<std::string,double> rewards;

// 	auto state_sim_body = mSimCharacter->getStateBody();
// 	auto state_sim_joint = mSimCharacter->getStateJoint();
// 	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
// 	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
// 	Eigen::Vector3d sim_com = mSimCharacter->getSkeleton()->getCOM();
// 	Eigen::Vector3d sim_com_vel = mSimCharacter->getSkeleton()->getCOMLinearVelocity();
// 	Eigen::MatrixXd sim_body_p, sim_body_R, sim_body_v, sim_body_w, sim_joint_p, sim_joint_v;
	
// 	sim_body_p = state_sim_body["ps"];
// 	sim_body_R = state_sim_body["Rs"];
// 	sim_body_v = state_sim_body["vs"];
// 	sim_body_w = state_sim_body["ws"];

// 	sim_joint_p = state_sim_joint["p"];
// 	sim_joint_v = state_sim_joint["v"];

// 	auto state_kin_body = mKinCharacter->getStateBody();
// 	auto state_kin_joint = mKinCharacter->getStateJoint();
// 	Eigen::Isometry3d T_kin = mKinCharacter->getReferenceTransform();
// 	Eigen::Isometry3d T_kin_inv = T_kin.inverse();
// 	Eigen::Vector3d kin_com = mKinCharacter->getSkeleton()->getCOM();
// 	Eigen::Vector3d kin_com_vel = mKinCharacter->getSkeleton()->getCOMLinearVelocity();
// 	Eigen::MatrixXd kin_body_p, kin_body_R, kin_body_v, kin_body_w, kin_joint_p, kin_joint_v;
	
// 	kin_body_p = state_kin_body["ps"];
// 	kin_body_R = state_kin_body["Rs"];
// 	kin_body_v = state_kin_body["vs"];
// 	kin_body_w = state_kin_body["ws"];

// 	kin_joint_p = state_kin_joint["p"];
// 	kin_joint_v = state_kin_joint["v"];

// 	int n = mSimCharacter->getSkeleton()->getNumBodyNodes();

// 	double error_pos=0.0, error_vel=0.0, error_ee=0.0, error_root=0.0, error_com=0.0;
// 	// pos error
// 	Eigen::MatrixXd diff_pos = DARTUtils::computeDiffPositions(sim_joint_p, kin_joint_p);
// 	Eigen::MatrixXd diff_vel = sim_joint_v - kin_joint_v;
// 	Eigen::MatrixXd lb_vel = Eigen::MatrixXd::Constant(diff_vel.rows(),diff_vel.cols(), -1.0);
// 	Eigen::MatrixXd ub_vel = Eigen::MatrixXd::Constant(diff_vel.rows(),diff_vel.cols(), 1.0);
// 	diff_vel = diff_vel.cwiseMax(lb_vel).cwiseMin(ub_vel);

// 	for(int i=1;i<n;i++)
// 	{
// 		error_pos += joint_weights[i]*(diff_pos.col(i).dot(diff_pos.col(i)));
// 		error_vel += joint_weights[i]*(diff_vel.col(i).dot(diff_vel.col(i)));
// 	}

// 	auto ees = mSimCharacter->getEndEffectors();
// 	for(int i=0;i<ees.size();i++)
// 	{
// 		int idx = ees[i]->getIndexInSkeleton();
// 		Eigen::Vector3d sim_ee = sim_body_p.col(idx);
// 		Eigen::Vector3d kin_ee = kin_body_p.col(idx);
		
// 		Eigen::Vector3d diff_ee_local = T_sim_inv*sim_ee - T_kin_inv*kin_ee;
// 		error_ee += diff_ee_local.dot(diff_ee_local);
// 	}

// 	Eigen::Vector3d diff_root_p = sim_body_p.col(0) - kin_body_p.col(0);
// 	Eigen::Vector3d diff_root_R = dart::math::logMap(sim_body_R.block<3,3>(0,0).transpose()*kin_body_R.block<3,3>(0,0));
// 	Eigen::Vector3d diff_root_v = sim_body_v.col(0) - kin_body_v.col(0);
// 	Eigen::Vector3d diff_root_w = sim_body_w.col(0) - kin_body_w.col(0);

// 	error_root = 1.0 * diff_root_p.dot(diff_root_p) + 
// 				0.1* diff_root_R.dot(diff_root_R) + 
// 				0.01* diff_root_v.dot(diff_root_v) + 
// 				0.01* diff_root_w.dot(diff_root_w);

// 	Eigen::Vector3d diff_com = T_sim_inv*sim_com - T_kin_inv*kin_com;
// 	Eigen::Vector3d diff_com_vel = T_sim_inv.linear()*sim_com_vel - T_kin_inv.linear()*kin_com_vel;
	
// 	error_com = 1.0*diff_com.dot(diff_com) + 
// 				0.1*diff_com_vel.dot(diff_com_vel);

// 	double r_pos = std::exp(-mWeightPos*error_pos);
// 	double r_vel = std::exp(-mWeightVel*error_vel);
// 	double r_ee = std::exp(-mWeightEE*error_ee);
// 	double r_root = std::exp(-mWeightRoot*error_root);
// 	double r_com = std::exp(-mWeightCOM*error_com);

// 	r_root = std::max(0.5, r_root);

// 	// double door_angle = mDoor->getPositions()[0];

// 	// double error_door_angle = 1.57*std::min(1.0, ((double)mElapsedFrame/(double)65)) - door_angle;
// 	// error_door_angle = std::max(error_door_angle, 0.0);
// 	// double r_door = std::exp(-4.0*error_door_angle);
// 	double r_imit = r_pos*r_vel*r_ee*r_root*r_com;
// 	rewards.insert(std::make_pair("r_pos",r_pos));
// 	rewards.insert(std::make_pair("r_vel",r_vel));
// 	rewards.insert(std::make_pair("r_ee",r_ee));
// 	rewards.insert(std::make_pair("r_root",r_root));
// 	rewards.insert(std::make_pair("r_com",r_com));
// 	rewards.insert(std::make_pair("r_force",1.0));
// 	rewards.insert(std::make_pair("r_imit",r_imit));
// 	// rewards.insert(std::make_pair("r",0.6*r_imit + 0.4*r_door));
// 	rewards.insert(std::make_pair("r",r_imit));

// 	for(auto rew : rewards)
// 	{
// 		if(dart::math::isNan(rew.second))
// 			rewards[rew.first] = -30.0;
// 		mRewards[rew.first].emplace_back(rewards[rew.first]);
// 	}

// 	return rewards;
// }
// bool
// Environment::
// isSleep()
// {
// 	for(auto fs: mSimCharacter->getForceSensors())
// 		if(!fs->isSleep())
// 			return false;

// 	return true;
// }

// bool
// Environment::
// inspectEndOfEpisode()
// {
// 	double r_mean = 0.0;
// 	{
// 		const std::vector<double>& rs = mRewards["r_imit"];
// 		int n = rs.size();
// 		for(int i=std::max(0,n-30);i<n;i++){
// 			r_mean += rs[i];
// 		}
// 		for(int i=n-30;i<0;i++)
// 			r_mean += 1.0;
// 		r_mean /= 30.0;
// 	}

// 	double r_force_mean = 0.0;
// 	{
// 		const std::vector<double>& rs = mRewards["r_force"];
// 		int n = rs.size();
// 		for(int i=std::max(0,n-30);i<n;i++){
// 			r_force_mean += rs[i];
// 		}
// 		for(int i=n-30;i<0;i++)
// 			r_force_mean += 1.0;
// 		r_force_mean /= 30.0;
// 		// std::cout<<r_force_mean<<std::endl;
// 	}
// 	if(mElapsedFrame>=mMaxElapsedFrame){
// 		return true;
// 	}
// 	else if(r_mean<0.1){
// 		return true;
// 	}
// 	else if(r_force_mean<0.1){
// 		return true;
// 	}
// 	else if(mContactEOE)
// 	{
// 		return true;
// 	}
// 	int sf = mCurrentFrame%82;
// 	if(sf<25 && sf>=20 && mDoorConstraintOn == false)
// 	{
// 		return true;
// 	}

// 	return false;
// }
// int
// Environment::
// getDimState0()
// {
// 	return this->getState0().rows();
// }
// int
// Environment::
// getDimAction0()
// {
// 	int n = mSimCharacter->getSkeleton()->getNumDofs();
// 	return n-6 + this->getDimAction1();
// }
// int
// Environment::
// getDimState1()
// {
// 	return this->getState1().rows();
// }

// int
// Environment::
// getDimAction1()
// {
// 	return 3*mSimCharacter->getForceSensors().size(); //FFF
// }
// int
// Environment::
// getDimStateAMP()
// {
// 	return this->getAMPState();
// }
// Eigen::MatrixXd
// Environment::
// getActionSpace0()
// {
// 	Eigen::MatrixXd action_space = Eigen::MatrixXd::Ones(this->getDimAction0(), 2);
// 	int n = mSimCharacter->getSkeleton()->getNumDofs();
// 	int m = 3*mSimCharacter->getForceSensors().size();

// 	action_space.col(0).head(n-6) *= -M_PI; // Lower
// 	action_space.col(1).head(n-6) *=  M_PI; // Upper

// 	action_space.col(0).tail(m) *= -M_PI; // Lower
// 	action_space.col(1).tail(m) *=  M_PI; // Upper

// 	return action_space;
// }
// Eigen::VectorXd
// Environment::
// getActionWeight0()
// {
// 	Eigen::VectorXd action_weight = Eigen::VectorXd::Ones(this->getDimAction0());
// 	int n = mSimCharacter->getSkeleton()->getNumDofs();
// 	int m = 3*mSimCharacter->getForceSensors().size();
// 	action_weight.head(n-6) *= 0.3;
// 	action_weight.tail(m) *= 500.0; // Lower
// 	return action_weight;
// }
// Eigen::VectorXd
// Environment::
// convertToRealActionSpace0(const Eigen::VectorXd& a_norm)
// {
// 	Eigen::VectorXd a_real;
// 	Eigen::VectorXd lo = mActionSpace0.col(0), hi =  mActionSpace0.col(1);
// 	a_real = dart::math::clip<Eigen::VectorXd, Eigen::VectorXd>(a_norm, lo, hi);
// 	a_real = mActionWeight0.cwiseProduct(a_real);
// 	return a_real;
// }
// Eigen::MatrixXd
// Environment::
// getActionSpace1()
// {
// 	Eigen::MatrixXd action_space = Eigen::MatrixXd::Ones(this->getDimAction1(), 2);
// 	int m = 3*mSimCharacter->getForceSensors().size();

// 	action_space.col(0) *= -1.0;
// 	action_space.col(1) *= 1.0;
// 	return action_space;
// }
// Eigen::VectorXd
// Environment::
// getActionWeight1()
// {
// 	Eigen::VectorXd action_weight = Eigen::VectorXd::Ones(this->getDimAction1());
// 	int m = 3*mSimCharacter->getForceSensors().size();

// 	action_weight.head(m) *= 1000.0; //FFF
// 	// action_weight.tail(n) *= 0.8; //FFF

// 	//action_weight *= 0.3; //PPP
// 	return action_weight;
// }
// Eigen::VectorXd
// Environment::
// convertToRealActionSpace1(const Eigen::VectorXd& a_norm)
// {
// 	Eigen::VectorXd a_real;
// 	Eigen::VectorXd lo = mActionSpace1.col(0), hi =  mActionSpace1.col(1);
// 	a_real = dart::math::clip<Eigen::VectorXd, Eigen::VectorXd>(a_norm, lo, hi);
// 	a_real = mActionWeight1.cwiseProduct(a_real);
	
// 	int m = 3*mSimCharacter->getForceSensors().size();

// 	// a_real = 0.7*mPrevAction1 + 0.3*a_real;

// 	// Eigen::AngleAxisd aa(Eigen::Quaterniond::FromTwoVectors(mPrevAction1.head(m), a_real.head(m)));
// 	// double action_diff = aa.angle();
// 	// if(action_diff>0.5){
// 	// 	double a_real_val = a_real.head(m).norm();
// 	// 	Eigen::AngleAxisd aa2(0.5,aa.axis());
// 	// 	a_real.head(m) = aa2.toRotationMatrix()*(mPrevAction1.head(m).normalized());
// 	// 	a_real.head(m) *= a_real_val;
// 	// }

// 	// a_real.tail(n) += Eigen::VectorXd::Ones(n);
// 	// a_real.tail(n) = Eigen::VectorXd::Ones(n);
// 	return a_real;
// }