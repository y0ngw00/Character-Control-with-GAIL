#include "Window.h"
#include "Camera.h"
#include "BVH.h"
#include <fstream>
#include <time.h>
#include <iostream>
#include "DrawUtils.h"
#include "DARTUtils.h"
#include "DARTRendering.h"
#include <dart/dart.hpp>
using namespace py::literals;
using namespace dart::common;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::math;

static bool show_demo_window = true;
static bool show_another_window = true;
static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

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
{
	mTimePoint = std::chrono::system_clock::now();
	mEnvironment = new Environment();
	mBarPlot.min_val = -1.0;
	mBarPlot.max_val = 1.0;
	mBarPlot.base_val = 0.0;
	mBarPlot.color = Eigen::Vector4d(0.8,0.8,0.8,0.6);
	
	mCamera->setLookAt(Eigen::Vector3d(0.0,0.3,0.8));
	mCamera->setEye( Eigen::Vector3d(-2.0,3.0,10.0));

	this->reset();

	float y = mEnvironment->getGround()->getBodyNode(0)->getTransform().translation()[1] +
			dynamic_cast<const BoxShape*>(mEnvironment->getGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	Eigen::Vector3d com = mEnvironment->getSimCharacter()->getSkeleton()->getCOM();
	com[1] = y+1.0;
	Eigen::Vector3d dir = mCamera->getEye() - mCamera->getLookAt();
	mCamera->setLookAt(com);
	mCamera->setEye( com + dir );

	char buffer[100];
	std::ifstream txtread;
	std::string txt_path = "/data/bvh/motionlist.txt";
	txtread.open(std::string(ROOT_DIR)+ txt_path);
	if(!txtread.is_open()){
		std::cout<<"Text file does not exist from : "<< txt_path << std::endl;
		return;
	}
	while(txtread>>buffer){
        motion_lists.push_back(std::string(buffer));	
	}
	txtread.close();

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
		DARTRendering::gRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/textures/ground.png").c_str());
		mSimRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/textures/simchar.png").c_str());
		mKinRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/textures/kinchar.png").c_str());
		mTargetRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/textures/targetchar.png").c_str());
		mObjectRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/textures/targetchar.png").c_str());
		mObjectRenderOption.drawJoints = false;
	}

	ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    ImGuiDisplay();
    // Rendering
    ImGui::Render();
    ImGuiIO& io = ImGui::GetIO();



	glColor4f(0.4,0.4,1.2,0.2);
	
	glColor4f(1.2,0.4,0.4,0.8);DrawUtils::drawArrow3D(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(), 0.2);
	glColor4f(0.4,1.2,0.4,0.8);DrawUtils::drawArrow3D(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY(), 0.2);
	glColor4f(0.4,0.4,1.2,0.8);DrawUtils::drawArrow3D(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), 0.2);

	

	Eigen::Vector3d force_dir =mEnvironment->getTargetDirection();
	// Eigen::Matrix3d R_ref = mEnvironment->getSimCharacter()->getReferenceTransform().linear();
	// force_dir = R_ref.inverse() * force_dir;
	Eigen::Vector3d origin = mEnvironment->getSimCharacter()->getSkeleton()->getBodyNode(0)->getWorldTransform().translation();
	glColor4f(0.95,0.1,0.1,0.8); DrawUtils::drawArrow3D(origin, origin + force_dir,0.2);

	if(mDrawSimPose)
		DARTRendering::drawSkeleton(mEnvironment->getSimCharacter()->getSkeleton(),mSimRenderOption);
	if(mDrawKinPose)
		DARTRendering::drawSkeleton(mEnvironment->getKinCharacter()->getSkeleton(),mKinRenderOption);

	if(mDrawTargetPose)
	{
		Eigen::VectorXd state = mEnvironment->getKinCharacter()->saveState();
		Eigen::VectorXd p_sim = mEnvironment->getSimCharacter()->getSkeleton()->getPositions();
		Eigen::VectorXd p_target = mEnvironment->getSimCharacter()->getTargetPositions();
		int num_actuated_dof = p_sim.rows()-6;
		p_sim.tail(num_actuated_dof) = p_target.tail(num_actuated_dof);
		mEnvironment->getKinCharacter()->getSkeleton()->setPositions(p_sim);

		DARTRendering::drawSkeleton(mEnvironment->getKinCharacter()->getSkeleton(),mTargetRenderOption);
		mEnvironment->getKinCharacter()->restoreState(state);
	}
	float y = mEnvironment->getGround()->getBodyNode(0)->getTransform().translation()[1] +
			dynamic_cast<const BoxShape*>(mEnvironment->getGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;

	DrawUtils::drawGround(y,100.0);
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

	ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

	if(mCapture)
		this->capture_screen();
}

void
Window::
ImGuiDisplay()
{
     // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    // if (show_demo_window)
    //     ImGui::ShowDemoWindow(&show_demo_window);

    // // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {

        static int counter = 0;

        ImGui::Begin("Parameter Control Window");                          // Create a window called "Hello, world!" and append into it.

                    // Display some text (you can use a format strings too)
        // ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Control", &mControl);
        ImGui::Text("Current Motion "); ImGui::SameLine();
        ImGui::Text(motion_lists[mMotionType].c_str());
        // ImGui::SliderFloat("Theta", &theta, -180, 180);            // Edit 1 float using a slider from 0.0f to 1.0f
        
        // ImGui::Text("Forwarding height");   
        // ImGui::SliderFloat("Height", &height, 0.6f, 2.0f);

        // ImGui::Text("Speed");   
        // ImGui::SliderFloat("Speed", &speed, 0.5f, 3.0f);

        // static int motionidx = 0;


        for(int n=0; n<motion_lists.size();n++){
        	if(ImGui::Button(motion_lists[n].c_str()))
        		mMotionType = n;
        }
        // }
        ImGui::InputInt("Class", &mMotionType);
        // ImGui::RadioButton("normal", &motionidx, 0); ImGui::SameLine();
        // ImGui::RadioButton("jump", &motionidx, 1); ImGui::SameLine();
        // ImGui::RadioButton("hurt", &motionidx, 2);
        // ImGui::RadioButton("wild", &motionidx, 3); ImGui::SameLine();
        // ImGui::RadioButton("zombie", &motionidx, 4); ImGui::SameLine();

        // mMotionType.setZero();
        // mMotionType[motionidx]=1;
        // ImGui::RadioButton("radio c", &motionidx, 2);
        // ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        // if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
        //     counter++;

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    // // 3. Show another simple window.
    {
        ImGui::Begin("Indicator");   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        // if (ImGui::BeginTable("table1", 3))
        // {
        //     for (int row = 0; row < 4; row++)
        //     {
        //         ImGui::TableNextRow();
        //         for (int column = 0; column < 3; column++)
        //         {
        //             ImGui::TableSetColumnIndex(column);
        //             ImGui::Text("Row %d Column %d", row, column);
        //         }
        //     }
        //     ImGui::EndTable();
        // }
       if (ImGui::CollapsingHeader("Style Reward"))
        {
	        int n = mRewards.size();
	        ImPlot::SetNextPlotLimitsX(n-200,n+1.0,ImGuiCond_Always);
	        ImPlot::SetNextPlotLimitsY(-0.2,1.2);
	        
	        
	        double* x = new double[n]();
	        double* y = new double[n]();
	        for(int i = 0; i < n; i++)
	        {
	            x[i] = i; y[i] = mRewards[i];
	        }
	        if (ImPlot::BeginPlot("reward"))
	        {
	            ImPlot::PlotLine("",x,y,n);       
	            ImPlot::EndPlot();
	        }

	        ImGui::Text("Task Reward "); 
	        int nG = mRewardGoals.size();
	        ImPlot::SetNextPlotLimitsX(nG-200,nG+1.0,ImGuiCond_Always);
	        ImPlot::SetNextPlotLimitsY(-0.2,1.2);
	        
	        
	        double* x2 = new double[nG]();
	        double* y2 = new double[nG]();
	        for(int i = 0; i < n; i++)
	        {
	            x2[i] = i; y2[i] = mRewardGoals[i];
	        }
	        if (ImPlot::BeginPlot("Task reward"))
	        {
	            ImPlot::PlotLine("",x2,y2,nG);       
	            ImPlot::EndPlot();
	        }
    	}

        // if (ImGui::Button("Close Me"))
        //     show_another_window = false;
        ImGui::End();
    }
}


void
Window::
reset(int frame)
{
	if(!mControl){
		mEnvironment->reset(true);
	}
	else{
		mEnvironment->reset(false);
		mEnvironment->setStateLabel(this->mMotionType);
	}
	mObservation = mEnvironment->getState();
	mObservationDiscriminator = mEnvironment->getStateAMP();

	mRewards.clear();
	mRewardGoals.clear();
	mMotionTypes.clear();
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

	if(mControl){
		mEnvironment->setStateLabel(mMotionType);
		// mEnvironment->setTargetMotion(this->mMotionType);		
	}
	else{
		this->mMotionType = mEnvironment->getStateLabel();		
	}

	if(mUseNN)
	{
		Eigen::VectorXd action = policy.attr("compute_action")(mObservation, mExplore).cast<Eigen::VectorXd>();
		mEnvironment->step(action);
	}

	else{

		Eigen::VectorXd action = Eigen::VectorXd::Zero(mEnvironment->getDimAction());
		mEnvironment->step(action);
	}


	mObservationDiscriminator = mEnvironment->getStateAMP();
	mReward = discriminator.attr("compute_reward")(mObservationDiscriminator).cast<double>();

	mObservation = mEnvironment->getState();
	
	mRewardGoal = mEnvironment->getRewardGoal();
	
	mRewardGoals.push_back(mRewardGoal);
	mRewards.push_back(0.5*(mReward));
	mMotionTypes.push_back(mMotionType);
	bool eoe = mEnvironment->inspectEndOfEpisode();
	if(eoe)
		this->reset();
	if(mDrawKinPose){
		mEnvironment->FollowBVH(this->mMotionType);
	}

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

	policy = policy_md.attr("build_policy")(mEnvironment->getDimState(),mEnvironment->getDimStateLabel(),mEnvironment->getDimAction(),pyconfig);
	discriminator = discriminator_md.attr("build_discriminator")(mEnvironment->getDimStateAMP(), mEnvironment->getNumTotalLabel(),mEnvironment->getDimStateLabel(),mEnvironment->getStateAMPExpert(), pyconfig);

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
		case 'r':this->reset();break;
		case 'R':this->reset(0);break;
		case 'C':mCapture=true;break;
		case ' ':mPlay = !mPlay; break;
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



	if(mMouse == 2) // Right
	{
		if(state==0) // Down
		{
			auto ray = mCamera->getRay(x,y);
		}
		else{
		}
	}
}
void
Window::
motion(int x, int y)
{
	GLUTWindow3D::motion(x,y);
	if(mMouse == 2 && mDrag)
	{
		auto ray = mCamera->getRay(x,y);
	}
	else
	{

	}
	
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