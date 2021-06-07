#include "Window.h"
#include "Camera.h"
#include "Event.h"
#include "BVH.h"
#include <fstream>
#include <time.h>
#include <iostream>
#include "DrawUtils.h"
#include "MassSpringDamperSystem.h"
#include "DARTUtils.h"
#include "DARTRendering.h"
#include <dart/dart.hpp>
using namespace py::literals;
using namespace dart::common;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::math;

Window::
Window()
	:GLUTWindow3D(),
	mPlay(false),
	mUseNN(false),
	mDrawTargetPose(false),
	mExplore(false),
	mDrawKinPose(true),
	mDrawSimPose(true),
	mPlotReward(false),
	mFocus(false),
	mDrawCOMvel(false),
	mDraw2DCharacter(true),
	mCapture(false)
	// ,mCurrentForceSensor(nullptr)
{
	mTimePoint = std::chrono::system_clock::now();
	mEnvironment = new Environment();
	mBarPlot.min_val = -1.0;
	mBarPlot.max_val = 1.0;
	mBarPlot.base_val = 0.0;
	mBarPlot.color = Eigen::Vector4d(0.8,0.8,0.8,0.6);
	
	mCamera->setLookAt(Eigen::Vector3d(0.0,1.3,0.8));
	mCamera->setEye( Eigen::Vector3d(-2.0,1.3,-0.8));

	this->reset();

	float y = mEnvironment->getGround()->getBodyNode(0)->getTransform().translation()[1] +
			dynamic_cast<const BoxShape*>(mEnvironment->getGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	Eigen::Vector3d com = mEnvironment->getSimCharacter()->getSkeleton()->getCOM();
	com[1] = y+1.0;
	Eigen::Vector3d dir = mCamera->getEye() - mCamera->getLookAt();
	mCamera->setLookAt(com);
	mCamera->setEye( com + dir );
}
void
Window::
render()
{
	initLights();
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_MULTISAMPLE);
	if(DrawUtils::initialized == false){
		DrawUtils::buildMeshes();
		DARTRendering::gRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/ground.png").c_str());
		mSimRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/simchar.png").c_str());
		mKinRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/kinchar.png").c_str());
		mTargetRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/targetchar.png").c_str());
		mObjectRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/targetchar.png").c_str());
		mObjectRenderOption.drawJoints = false;

		// mSimRenderOption.draw_mode = DrawUtils::eDrawWireSimple;
		
	}
	glColor4f(0.4,0.4,1.2,0.2);
	
	
	glColor4f(1.2,0.4,0.4,0.8);DrawUtils::drawArrow3D(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(), 0.2);
	glColor4f(0.4,1.2,0.4,0.8);DrawUtils::drawArrow3D(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY(), 0.2);
	glColor4f(0.4,0.4,1.2,0.8);DrawUtils::drawArrow3D(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), 0.2);

	if(mDrawSimPose){
		DARTRendering::drawSkeleton(mEnvironment->getSimCharacter()->getSkeleton(),mSimRenderOption);
		auto msd_system = mEnvironment->getSimCharacter()->getCartesianMSD();
		Eigen::Vector3d dir = msd_system->getPosition();
		Eigen::Vector3d pos = mEnvironment->getSimCharacter()->getSkeleton()->getBodyNode(0)->getCOM();
		if(mDrawCOMvel)
		DrawUtils::drawArrow3D(pos, pos+dir, 0.2);
		pos = mEnvironment->getCurrentTargetHandPos();
		glPushMatrix();
		DrawUtils::translate(pos);
		glColor3f(1,0,0);
		// DrawUtils::drawSphere(0.1);
		
		glPopMatrix();

		auto fss = mEnvironment->getSimCharacter()->getForceSensors();
		for(auto fs: fss)
		{
			Eigen::Vector3d dir = fs->getHapticPosition(false);
			Eigen::Vector3d pos = fs->getPosition();
			// DrawUtils::drawArrow3D(pos, pos+dir, 0.1);
		}
		DARTRendering::drawForceSensors(mEnvironment->getSimCharacter(),Eigen::Vector3d(0.8,0.7,0.0),Eigen::Vector3d(0.16,0.4,0.0),mSimRenderOption);

		// Eigen::Vector3d com_vel = mEnvironment->getStateGoal();
		// glColor3f(1,0,0);
		// DrawUtils::drawArrow3D(pos, pos+com_vel, 0.2);
		// Eigen::Vector3d com_vel = (mEnvironment->getSimCharacter()->getSkeleton()->getCOM() - mPrevCOM)*mControlHz;
		// com_vel[1] = 0.0;
		if(mEnvironment->getObstacle()!=nullptr)
			DARTRendering::drawSkeleton(mEnvironment->getObstacle(),mKinRenderOption);
		// if(mEnvironment->isEnableGoal())
		// {
		// 	DARTRendering::drawSkeleton(mEnvironment->getDoor(),mKinRenderOption);
		// 	glColor4f(1.0,1.0,1.0,0.2);
		// 	Eigen::VectorXd p_save = mEnvironment->getDoor()->getPositions();
		// 	Eigen::VectorXd p = p_save;
		// 	p[0] = mEnvironment->getTargetDoorAngle();
		// 	mEnvironment->getDoor()->setPositions(p);
		// 	// DARTRendering::drawSkeleton(mEnvironment->getDoor(),mTargetRenderOption);
		// 	mEnvironment->getDoor()->setPositions(p_save);
		// 	glColor4f(1.0,1.0,1.0,1.0);
		// }


		
	}
	if(mDraw2DCharacter)
		DARTRendering::drawForceSensors(mEnvironment->getSimCharacter(),Eigen::Vector3d(0.8,0.7,0.0),Eigen::Vector3d(0.16,0.4,0.0),mSimRenderOption);
	if(mDrawKinPose)
		DARTRendering::drawSkeleton(mEnvironment->getKinCharacter()->getSkeleton(),mKinRenderOption);

	// if(mEnvironment->isEnableGoal())
	// {
	// 	double target_heading = mEnvironment->getTargetHeading();
	// 	double target_speed = mEnvironment->getTargetSpeed();
	// 	Eigen::Vector3d start = mEnvironment->getSimCharacter()->getReferenceTransform().translation();
	// 	Eigen::Matrix3d R_target = Eigen::AngleAxisd(target_heading, Eigen::Vector3d::UnitY()).toRotationMatrix();
	// 	Eigen::Matrix3d R_ref = mEnvironment->getSimCharacter()->getReferenceTransform().linear();

	// 	Eigen::Vector3d dir = target_speed*R_target.col(2);
	// 	glColor4f(1,0,0,1);
	// 	DrawUtils::drawArrow3D(start, start+dir*0.7, 0.2);
	// }
	
	if(mDrawTargetPose)
	{
		Eigen::VectorXd state = mEnvironment->getKinCharacter()->saveState();
		Eigen::VectorXd p_sim = mEnvironment->getSimCharacter()->getSkeleton()->getPositions();
		Eigen::VectorXd p_target = mEnvironment->getSimCharacter()->getTargetPositions();
		int num_actuated_dof = p_sim.rows()-6;
		p_sim.tail(num_actuated_dof) = p_target.tail(num_actuated_dof);
		mEnvironment->getKinCharacter()->getSkeleton()->setPositions(p_sim);

		// mEnvironment->getKinCharacter()->setPose();
		DARTRendering::drawSkeleton(mEnvironment->getKinCharacter()->getSkeleton(),mTargetRenderOption);
		mEnvironment->getKinCharacter()->restoreState(state);
	}
	float y = mEnvironment->getGround()->getBodyNode(0)->getTransform().translation()[1] +
			dynamic_cast<const BoxShape*>(mEnvironment->getGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;

	DrawUtils::drawGround(y,100.0);
	DrawUtils::disableTexture();
	if(mPlotReward)
	{
		glDisable(GL_LIGHTING);
		glDisable(GL_TEXTURE_2D);

		glMatrixMode(GL_PROJECTION);

		glPushMatrix();
		glLoadIdentity();
		gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		int n = mRewards.size();
		int offset = std::max(0,n-30);
		mBarPlot.vals = Eigen::Map<Eigen::VectorXd>(mRewards.data()+offset, 30 + std::min(0,n-30));
		mBarPlot.background_color = Eigen::Vector4d(1,1,1,0);
		mBarPlot.color = Eigen::Vector4d(0,0,0,1);
		DrawUtils::drawLinePlot(mBarPlot, Eigen::Vector3d(0.69,0.69,0.0),Eigen::Vector3d(0.3,0.3,0.0));	
		mBarPlot.vals = Eigen::Map<Eigen::VectorXd>(mRewardGoals.data()+offset, 30 + std::min(0,n-30));
		mBarPlot.background_color = Eigen::Vector4d(1,1,1,0);
		mBarPlot.color = Eigen::Vector4d(1,0,0,1);
		DrawUtils::drawLinePlot(mBarPlot, Eigen::Vector3d(0.69,0.69,0.0),Eigen::Vector3d(0.3,0.3,0.0));	

		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		// glMatrixMode(oldMode);

		glEnable(GL_LIGHTING);
		glEnable(GL_TEXTURE_2D);
	
	}
	if(mCapture)
		this->capture_screen();
}

void
Window::
reset(int frame)
{
	mEnvironment->reset(frame);
	mObservation = mEnvironment->getState();
	mObservationDiscriminator = mEnvironment->getStateAMP();

	mRewards.clear();
	mRewardGoals.clear();
	mReward = 0.0;
	if(mFocus)
	{
		float y = mEnvironment->getGround()->getBodyNode(0)->getTransform().translation()[1] +
			dynamic_cast<const BoxShape*>(mEnvironment->getGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
		Eigen::Vector3d com = mEnvironment->getSimCharacter()->getSkeleton()->getCOM();
		com[1] = y+1.0;
		Eigen::Vector3d dir = mCamera->getEye() - mCamera->getLookAt();
		mCamera->setLookAt(com);
		mCamera->setEye( com + dir );
	}
}
void
Window::
step()
{
	if(mUseNN)
	{
		Eigen::VectorXd action = policy.attr("compute_action")(mObservation, mExplore).cast<Eigen::VectorXd>();
		mEnvironment->step(action);
	}
	else{
		Eigen::VectorXd action = Eigen::VectorXd::Zero(mEnvironment->getDimAction());
		mEnvironment->step(action);
	}
	// Eigen::MatrixXd ds_dq = mEnvironment->getSimCharacter()->getStateDeriv();
	// Eigen::MatrixXd dphi_ds = policy.attr("compute_grad")(mObservation).cast<Eigen::MatrixXd>();
	// Eigen::MatrixXd dphi_dq = dphi_ds*ds_dq;

	// dphi_dq = Eigen::MatrixXd::Identity(dphi_dq.rows(), dphi_dq.cols()) - dphi_dq;

	mObservationDiscriminator = mEnvironment->getStateAMP();
	mReward = discriminator.attr("compute_reward")(mObservationDiscriminator).cast<double>();

	mObservation = mEnvironment->getState();
	
	mRewardGoal = mEnvironment->getRewardGoal();
	
	mRewardGoals.push_back(mRewardGoal);
	mRewards.push_back(0.5*(mReward+mRewardGoal));
	bool eoe = mEnvironment->inspectEndOfEpisode();
	if(eoe)
		this->reset();

	if(mFocus)
	{
		float y = mEnvironment->getGround()->getBodyNode(0)->getTransform().translation()[1] +
			dynamic_cast<const BoxShape*>(mEnvironment->getGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
		Eigen::Vector3d com = mEnvironment->getSimCharacter()->getSkeleton()->getCOM();
		com[1] = y+1.0;
		Eigen::Vector3d dir = mCamera->getEye() - mCamera->getLookAt();
		mCamera->setLookAt(com);
		mCamera->setEye( com + dir );
	}	
}

void
Window::
initNN(const std::string& config)
{
	mUseNN = true;

	mm = py::module::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = py::module::import("sys");
	py::str module_dir = (std::string(ROOT_DIR)+"/python").c_str();
	sys_module.attr("path").attr("insert")(1, module_dir);

	policy_md = py::module::import("ppo");
	discriminator_md = py::module::import("discriminator");
	py::object pyconfig = policy_md.attr("load_config")(config);
	policy = policy_md.attr("build_policy")(mEnvironment->getDimState(),mEnvironment->getDimAction(),pyconfig);
	discriminator = discriminator_md.attr("build_discriminator")(mEnvironment->getDimStateAMP(), mEnvironment->getStateAMPExpert(), pyconfig);
	//TODO
	
	// policy0 = policy_md.attr("build_policy0")(mEnvironment->getDimState0(),mEnvironment->getDimAction0(),pyconfig);
	// policy1 = policy_md.attr("build_policy1")(mEnvironment->getDimState1(),mEnvironment->getDimAction1(),pyconfig);
}
void
Window::
loadNN(const std::string& checkpoint)
{
	//TODO
	policy_md.attr("load_policy")(policy, checkpoint);

	discriminator_md.attr("load_discriminator")(discriminator, checkpoint);
}
void
Window::
keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
		case '0':mDrawSimPose = !mDrawSimPose;break;
		case '1':mDrawKinPose = !mDrawKinPose;break;
		case '2':mDrawTargetPose = !mDrawTargetPose;break;
		case '3':mExplore = !mExplore;break;
		case '4':mPlotReward = !mPlotReward;break;
		case '5':mFocus = !mFocus;break;
		case '6':mDrawCOMvel = !mDrawCOMvel;break;
		case '7':mDraw2DCharacter = !mDraw2DCharacter;break;
		case 's':this->step();break;
		case '8':mEnvironment->getSimCharacter()->getStateDeriv();break;
		case 'k':mEnvironment->setKinematics(!mEnvironment->isKinematics());break;
		case 'r':this->reset();break;
		case 'R':this->reset(0);break;
		case 'C':mCapture=true;break;
		case ' ':mPlay = !mPlay; break;
		case 'v':mEnvironment->generateObstacleForce();break;
		default:GLUTWindow3D::keyboard(key,x,y);break;
	}
}
void
Window::
special(int key, int x, int y)
{
	switch(key)
	{
		case 100: break;//Left
		case 101: break;//Up
		case 102: break;//Right
		case 103: break;//bottom
		default:GLUTWindow3D::special(key,x,y);break;
	}

}
void
Window::
mouse(int button, int state, int x, int y)
{
	GLUTWindow3D::mouse(button,state,x,y);

	// if(mMouse == 2) // Right
	// {
	// 	if(state==0) // Down
	// 	{
	// 		auto ray = mCamera->getRay(x,y);

	// 		Event* event = mEnvironment->getEvent();
	// 		if(dynamic_cast<ObstacleEvent*>(event)!=nullptr)
	// 		{
	// 			ObstacleEvent* oevent = dynamic_cast<ObstacleEvent*>(event);
	// 			auto obstacle = oevent->getObstacle();
	// 			Eigen::Vector3d p0 = ray.first;
	// 			Eigen::Vector3d p1 = ray.second;



	// 			Eigen::Vector3d p_glob = obstacle->getPositions().segment<3>(3);
	// 			double t = (p1 - p0).dot(p_glob - p0)/(p1 - p0).dot(p1 - p0);
	// 			double distance = (t*(p1-p0) - (p_glob - p0)).norm();

	// 			mInteractionDepth = t;
	// 		}
	// 	}
	// 	else
	// 		mInteractionDepth = -1;
	// }

}
void
Window::
motion(int x, int y)
{
	GLUTWindow3D::motion(x,y);
	// if(mMouse == 2 && mDrag)
	// {
	// 	auto ray = mCamera->getRay(x,y);
	// 	Eigen::Vector3d current_force_point = ray.first + mInteractionDepth*(ray.second-ray.first);
	// 	Event* event = mEnvironment->getEvent();
	// 	if(dynamic_cast<ObstacleEvent*>(event)!=nullptr)
	// 	{
	// 		ObstacleEvent* oevent = dynamic_cast<ObstacleEvent*>(event);
	// 		auto obstacle = oevent->getObstacle();
	// 		Eigen::VectorXd p = obstacle->getPositions();
	// 		Eigen::VectorXd v = obstacle->getVelocities();
	// 		p.segment<3>(0) = Eigen::Vector3d::Zero();
	// 		p.segment<3>(3) = current_force_point;
	// 		v.segment<3>(0) = Eigen::Vector3d::Zero();
	// 		v.segment<3>(3) = Eigen::Vector3d::Zero();//(current_force_point-)/30.0;
	// 		oevent->setLinearPosition(current_force_point);
	// 		oevent->setLinearVelocity(Eigen::Vector3d::Zero());
	// 	}
	// }
	
}
void
Window::
reshape(int w, int h)
{
	mScreenshotTemp.resize(4*w*h);
	mScreenshotTemp2.resize(4*w*h);
	GLUTWindow3D::reshape(w,h);
}
void
Window::
timer(int tic)
{
	auto next_time_point = std::chrono::system_clock::now();
	std::chrono::microseconds micro = std::chrono::duration_cast<std::chrono::microseconds>(next_time_point - mTimePoint);
	mComputedTime = micro.count();

	mTimePoint = next_time_point;
	if(mPlay)
		this->step();
	GLUTWindow3D::timer(tic);
}
#include <chrono>
#include <ctime>
#include "lodepng.h"

std::string timepoint_to_string(const std::chrono::system_clock::time_point& p_tpTime,
                                           const std::string& p_sFormat)
{
    auto converted_timep = std::chrono::system_clock::to_time_t(p_tpTime);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&converted_timep), p_sFormat.c_str());
 
    return oss.str();
}

void
Window::
capture_screen()
{
	static int count = 0;
	static std::string path = timepoint_to_string(std::chrono::system_clock::now(), "../data/png/%Y_%m_%d:%H_%M_%S");
	if(count == 0){
		std::string command = "mkdir " + path;
		system(command.c_str());	
	}
	
	
	char file_name[256];
	std::string file_base = "Capture";

	std::snprintf(file_name, sizeof(file_name), "%s%s%s%.4d.png",
				path.c_str(), "/", file_base.c_str(), count++);

	int tw = glutGet(GLUT_WINDOW_WIDTH);
	int th = glutGet(GLUT_WINDOW_HEIGHT);

	glReadPixels(0, 0,  tw, th, GL_RGBA, GL_UNSIGNED_BYTE, &mScreenshotTemp[0]);

	// reverse temp2 temp1
	for (int row = 0; row < th; row++) {
	memcpy(&mScreenshotTemp2[row * tw * 4],
		   &mScreenshotTemp[(th - row - 1) * tw * 4], tw * 4);
	}
	
	unsigned result = lodepng::encode(file_name, mScreenshotTemp2, tw, th);

	// if there's an error, display it
	if (result) {
	std::cout << "lodepng error " << result << ": "
			<< lodepng_error_text(result) << std::endl;
	} else {
		std::cout << "wrote screenshot " << file_name << "\n";
	}
}


// if(mDraw2DCharacter)
	// DARTRendering::drawForceSensors(mEnvironment->getSimCharacter(),Eigen::Vector3d(0.8,0.7,0.0),Eigen::Vector3d(0.16,0.4,0.0),mSimRenderOption);
	// DrawUtils::drawString3D("this is ground",Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());


	// auto fss = mEnvironment->getSimCharacter()->getForceSensors();
	// for(int i =0;i<fss.size();i++)
	// {
	// 	auto fs = fss[i];
	// 	if(!fs->isSleep())
	// 	{
	// 		Eigen::Vector3d fi = mEnvironment->getSimCharacter()->XXXX;
	// 		// std::cout<<fi.norm()<<std::endl;

	// 		glColor4f(0.4,0.4,1.2,0.6);
	// 		Eigen::Isometry3d T_ref = fs->getBodyNode()->getTransform();
	// 		Eigen::Vector3d start = T_ref*fs->getLocalOffset();
	// 		Eigen::Vector3d dir = fi;
	// 		DrawUtils::drawArrow3D(start, start+dir*0.003, 0.1);
	// 		// DrawUtils::drawArrow3D(start, start+mEnvironment->__fv*0.3, 0.1);
			
	// 		glColor4f(0,0,0,1);
	// 	}

	// }	





	// Eigen::Vector3d swing_position = mEnvironment->getSimCharacter()->getMSDSystem()->mSwingPosition.translation();
	// Eigen::Vector3d stance_position = mEnvironment->getSimCharacter()->getMSDSystem()->mStancePosition.translation();
	// Eigen::Vector3d current_hip_position = mEnvironment->getSimCharacter()->getMSDSystem()->mGlobalHipPosition;

	// glPushMatrix();
	// DrawUtils::translate(swing_position);
	// glTranslatef(0,y,0);
	// glColor3f(1,0,0);
	// DrawUtils::drawSphere(0.1);
	
	// glPopMatrix();
	

	// glPushMatrix();
	// glColor3f(0,0,1);
	// DrawUtils::translate(stance_position);
	// glTranslatef(0,y,0);
	// DrawUtils::drawSphere(0.1);

	// glPopMatrix();

	// glPushMatrix();
	// glColor3f(0,1,0);
	// DrawUtils::translate(current_hip_position);
	// // glTranslatef(0,y,0);
	// DrawUtils::drawSphere(0.1);

	// glPopMatrix();

	// for(int i=0;i<1000;i++)
	// {
	// 	double mTHETA = dart::math::Random::uniform<double>(0.0,0.5*M_PI);
	// 	double mPHI = dart::math::Random::uniform<double>(0.0,2*M_PI);
	// 	Eigen::Vector3d center = Eigen::Vector3d(-0.0903061, 2.44461, 0.534614);

	// 	double x = std::sin(mTHETA)*std::cos(mPHI);
	// 	double y = std::sin(mTHETA)*std::sin(mPHI);
	// 	double z = std::cos(mTHETA);

	// 	center[0] += x;
	// 	center[1] += y;
	// 	center[2] += z;	
	// 	glPushMatrix();
	// 	DrawUtils::translate(center);
	// 	DrawUtils::drawSphere(0.05);
	// 	glPopMatrix();
	// }
	
	// {
	// Eigen::Vector3d center = Eigen::Vector3d(-0.0903061, 2.44461, 0.534614);

	// double r = mEnvironment->mR;
	// double theta = mEnvironment->mTHETA;
	// double phi = mEnvironment->mPHI;
	// double x = r*std::sin(theta)*std::cos(phi);
	// double y = r*std::sin(theta)*std::sin(phi);
	// double z = r*std::cos(theta);

	// center[0] += x;
	// center[1] += y;
	// center[2] += z;

	// Eigen::Isometry3d T_sim = mEnvironment->getSimCharacter()->getReferenceTransform();
	// glPushMatrix();
	// DrawUtils::translate(center);
	// DrawUtils::drawSphere(0.05);
	// glPopMatrix();
	// }
	// DrawUtils::disableTexture();
	// DrawUtils::enableTexture(false);
	// glBegin(GL_LINE_STRIP);
	// glColor3f(0,0,0);
	// glLineWidth(2.0);
	// for(int i=0;i<mHandPositions.size();i++)
	// {
	// 	glVertex3dv(mHandPositions[i].data());
	// }
	// glEnd();
	

	// auto cr = mEnvironment->mWorld->getConstraintSolver()->getLastCollisionResult();

	// for (auto j = 0u; j < cr.getNumContacts(); ++j)
	// {
	// 	auto contact = cr.getContact(j);



	// 	auto shapeFrame1 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject1->getShapeFrame());
	// 	auto shapeFrame2 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject2->getShapeFrame());

	// 	std::string name1 = shapeFrame1->asShapeNode()->getBodyNodePtr()->getSkeleton()->getName();
	// 	std::string name2 = shapeFrame2->asShapeNode()->getBodyNodePtr()->getSkeleton()->getName();

	// 	if(name1 == "ground" or name2 == "ground")
	// 		continue;

	// 	Eigen::Vector3d point = contact.point;
	// 	Eigen::Vector3d force = contact.force;
	// 	if(force.norm()<1e-3)
	// 		continue;
	// 	glColor4f(0.4,0.4,1.2,0.6);

	// 	DrawUtils::drawArrow3D(point, point-force*0.03, 0.1);
	// 	glColor4f(0,0,0,1);
	// }

	// glDisable(GL_LIGHTING);
	// glColor4f(0.4,0.4,0.4,1.0);
	// for(int i=0;i<((int)mWashedRecords.size())-1;i++)
	// {
	// 	Eigen::Vector3d pos0 = mWashedRecords[i].segment<3>(0);
	// 	double val0 = 3e-1*mWashedRecords[i][3];
	// 	Eigen::Vector3d pos1 = mWashedRecords[i+1].segment<3>(0);
	// 	double val1 = 3e-1*mWashedRecords[i+1][3];
	// 	Eigen::Vector3d v = pos1 - pos0;

	// 	Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(),v);
	// 	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	// 	T.linear() = q.toRotationMatrix();
	// 	T.translation() = pos0;
	// 	glPushMatrix();
	// 	DrawUtils::transform(T);
	// 	DrawUtils::drawCylinder2(val0,val1,v.norm());
	// 	DrawUtils::drawSphere(val0);
	// 	glPopMatrix();
	// 	// glPushMatrix();
	// 	// DrawUtils::translate(pos);
	// 	// DrawUtils::scale(Eigen::Vector3d(1.0,1.0,0.01));
	// 	// DrawUtils::drawSphere(val*3e-1);
	// 	// glPopMatrix();
	// }
	// glColor4f(0.4,0.4,0.4,1.0);
	// glEnable(GL_LIGHTING);
	// Eigen::Vector3d pos(0.0, 2.2,0.0);
	// Eigen::Vector3d size(0.2,0.4,0.1);
	// DrawUtils::drawBox(pos, size);
	// DARTRendering::drawSkeleton(mEnvironment->getBall(),mObjectRenderOption);
	// auto fss = mEnvironment->getSimCharacter()->getForceSensors();
	// if(mDrawKinPose)
	// for(auto fs: fss)
	// {
	// 	if(fs->active)
	// 	{
	// 		Eigen::Vector3d force_start = fs->getPosition();
	// 		Eigen::Vector3d force = fs->value*0.1;
	// 		// glColor3f(1.2,0.6,0.6);

	// 		// DrawUtils::drawArrow3D(force_start, force+force_start, 0.2);

	// 		force = fs->command.segment<3>(0)*0.01;
	// 		glColor3f(0.6,0.6,1.2);

	// 		DrawUtils::drawArrow3D(force_start, force+force_start, 0.2);

	// 		force_start = mEnvironment->getSimCharacter()->getSkeleton()->getBodyNode(0)->getCOM();
	// 		force = fs->command.segment<3>(3)*0.01;
	// 		glColor3f(0.6,0.6,1.2);

	// 		DrawUtils::drawArrow3D(force_start, force+force_start, 0.2);
	// 	}
	// }
	// DrawUtils::translate(mEnvironment->mBallEstimatedPosition);
	// glColor3f(1,0,0);
	// DrawUtils::drawSphere(0.1);
	// glPopMatrix();

	// glPushMatrix();
	
	// DrawUtils::translate(mEnvironment->mBallTargetPosition);
	// glColor3f(1,0,0);
	// DrawUtils::drawSphere(0.1);
	// glPopMatrix();
	

	
	// DrawUtils::disableTexture();
	// DrawUtils::enableTexture(false);

	// if(mPlotReward)
	// {
	// 	// auto oldMode = glGetIntegerv(GL_MATRIX_MODE);

	// 	glDisable(GL_LIGHTING);
	// 	glDisable(GL_TEXTURE_2D);

	// 	glMatrixMode(GL_PROJECTION);

	// 	glPushMatrix();
	// 	glLoadIdentity();
	// 	gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
	// 	glMatrixMode(GL_MODELVIEW);
	// 	glPushMatrix();
	// 	glLoadIdentity();
	// 	std::map<std::string, std::vector<double>> cr = mEnvironment->getCummulatedReward();
	// 	int n = cr["r"].size();
	// 	int offset = std::max(0,n-30);

		
	// 	mBarPlot.vals = Eigen::Map<Eigen::VectorXd>(cr["r"].data()+offset, 30 + std::min(0,n-30));
	// 	mBarPlot.background_color = Eigen::Vector4d(1,1,1,0);
	// 	mBarPlot.color = Eigen::Vector4d(0,0,0,1);
	// 	DrawUtils::drawLinePlot(mBarPlot, Eigen::Vector3d(0.69,0.69,0.0),Eigen::Vector3d(0.3,0.3,0.0));	

	// 	mBarPlot.vals = Eigen::Map<Eigen::VectorXd>(cr["r_pos"].data()+offset, 30 + std::min(0,n-30));
	// 	mBarPlot.background_color = Eigen::Vector4d(1,1,1,0);
	// 	mBarPlot.color = Eigen::Vector4d(1,0,0,1);
	// 	DrawUtils::drawLinePlot(mBarPlot, Eigen::Vector3d(0.69,0.69,0.0),Eigen::Vector3d(0.3,0.3,0.0));	

	// 	mBarPlot.vals = Eigen::Map<Eigen::VectorXd>(cr["r_vel"].data()+offset, 30 + std::min(0,n-30));
	// 	mBarPlot.background_color = Eigen::Vector4d(1,1,1,0.5);
	// 	mBarPlot.color = Eigen::Vector4d(0,0,1,1);
	// 	DrawUtils::drawLinePlot(mBarPlot, Eigen::Vector3d(0.69,0.69,0.0),Eigen::Vector3d(0.3,0.3,0.0));	

	// 	int fps = (int)(1.0/mComputedTime*1e6);
	// 	DrawUtils::drawString2D((std::to_string(fps)+" fps").c_str(), Eigen::Vector3d(0.89,0.9,0.0), Eigen::Vector3d(0.0,0.0,0.0));
		
	// 	glPopMatrix();
	// 	glMatrixMode(GL_PROJECTION);
	// 	glPopMatrix();
	// 	// glMatrixMode(oldMode);

	// 	glEnable(GL_LIGHTING);
	// 	glEnable(GL_TEXTURE_2D);
	
	// }
	
	
	// DARTRendering::drawForceSensors(mEnvironment->getSimCharacter(),Eigen::Vector3d(0.8,0.7,0.0),Eigen::Vector3d(0.16,0.4,0.0),mSimRenderOption);
	// // DrawUtils::drawString3D("this is ground",Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());
	// if(mCapture)
	// 	this->capture_screen();