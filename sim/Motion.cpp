#include "Motion.h"
#include <sstream>
#include "MathUtils.h"
#include "BVH.h"
#include <Eigen/Geometry>
#include <iostream>

Motion::
Motion(BVH* bvh)
{
	this->registerBVHHierarchy(bvh);
	// this->append(bvh->getPositions(), bvh->getRotations(),true);
}
void
Motion::
append(const Eigen::Vector3d& position,
		const Eigen::MatrixXd& rotation,
		bool compute_velocity)
{
	mPositions.emplace_back(position);
	mRotations.emplace_back(rotation);
	
	mNumFrames = mPositions.size();

	if(compute_velocity)
	{
		int n = mNumFrames;
		int m = mLinearVelocities.size();
		m = std::max(0,m-1); // compute last velocity again
		// std::cout<<m<<" "<<n<<std::endl;
		this->computeVelocity(m,n);
	}
}
void
Motion::
append(const std::vector<Eigen::Vector3d>& positions,
		const std::vector<Eigen::MatrixXd>& rotations,
		bool compute_velocity)
{
	mPositions.insert(mPositions.end(), positions.begin(), positions.end());
	mRotations.insert(mRotations.end(), rotations.begin(), rotations.end());
	mNumFrames = mPositions.size();

	if(compute_velocity)
	{
		int n = mNumFrames;
		int m = mLinearVelocities.size();
		m = std::max(0,m-1); // compute last velocity again
		this->computeVelocity(m,n);
	}
}

void
Motion::
clear()
{
	mPositions.clear();
	mRotations.clear();
	mLinearVelocities.clear();
	mAngularVelocities.clear();

	mNumFrames = 0;
}
void
Motion::
registerBVHHierarchy(BVH* bvh)
{
	mBVH = bvh;
	mTimestep = bvh->getTimestep();
	mNames = bvh->getNodeNames();
	mOffsets = bvh->getOffsets();
	mParents = bvh->getParents();
	mNumJoints = mParents.size();
}
void
Motion::
computeVelocity()
{
	this->computeVelocity(0);
}
void
Motion::
computeVelocity(int start)
{
	this->computeVelocity(start,mNumFrames);
}
void
Motion::
computeVelocity(int start, int end)
{
	assert(end <= mNumFrames);
	int n = end - start;
	std::vector<Eigen::Vector3d> linear_velocities(n);
	std::vector<Eigen::MatrixXd> angular_velocities(n);

	for(int i=0;i<n;i++)
	{
		int idx = start + i;
		int frame1 = std::max(0,idx-1);
		int frame2 = std::min(mNumFrames-1,idx+1);

		double dt_inv = 1.0/(mTimestep*(frame2-frame1));
		if(frame1==frame2)
			dt_inv = 0.0;

		Eigen::Vector3d pos1 = mPositions[frame1];
		Eigen::MatrixXd rot1 = mRotations[frame1];

		Eigen::Vector3d pos2 = mPositions[frame2];
		Eigen::MatrixXd rot2 = mRotations[frame2];

		linear_velocities[i] = (pos2 - pos1)*dt_inv;

		Eigen::Matrix3d R1 = rot1.block<3,3>(0,0);
		Eigen::Matrix3d R2 = rot2.block<3,3>(0,0);

		Eigen::Quaterniond Q1(R1);
		Eigen::Quaterniond Q2(R2);
		Eigen::Vector4d q1;
		Eigen::Vector4d q2;
		q1<<Q1.vec(),Q1.w();
		q2<<Q2.vec(),Q2.w();
		if( (q1-q2).norm() < (q1+q2).norm())
			Q2 = Eigen::Quaterniond(q2[3],q2[0],q2[1],q2[2]);

		Eigen::AngleAxisd aa(Q2 * Q1.conjugate()); // Not sure
		angular_velocities[i].resize(3,mNumJoints);
		
		angular_velocities[i].col(0) = aa.angle()*aa.axis();
		for(int j=1;j<mNumJoints;j++)
		{
			Q1 = Eigen::Quaterniond(rot1.block<3,3>(0,j*3));
			Q2 = Eigen::Quaterniond(rot2.block<3,3>(0,j*3));
			aa = Eigen::AngleAxisd(Q1.conjugate()*Q2);
			angular_velocities[i].col(j) = aa.angle()*aa.axis();
		}
		angular_velocities[i] *= dt_inv;
	}

	if(mLinearVelocities.size()>=start)
	{
		mLinearVelocities.erase(mLinearVelocities.begin()+start,mLinearVelocities.end());
		mAngularVelocities.erase(mAngularVelocities.begin()+start,mAngularVelocities.end());
	}
	// assert(mLinearVelocities.size() == start-1 || mLinearVelocities.size() == 0);
	mLinearVelocities.insert(mLinearVelocities.end(),linear_velocities.begin(),linear_velocities.end());
	mAngularVelocities.insert(mAngularVelocities.end(),angular_velocities.begin(),angular_velocities.end());

	// std::cout<<mLinearVelocities.size()<<std::endl;
	// std::cout<<mPositions.size()<<std::endl;
}








Eigen::MatrixXd
MotionUtils::
computePoseDifferences(Motion* m)
{
	int n = m->getNumFrames();
	Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n, n);
	for(int i=0;i<n;i++)
	{
		const Eigen::MatrixXd& Ri = m->getRotation(i);
		for(int j=i+1;j<n;j++)
		{
			const Eigen::MatrixXd& Rj = m->getRotation(j);
			double dij = computePoseDifference(Ri, Rj);
			D(i, j) = dij;
			D(j, i) = dij;
		}
	}
	return D;
}

double
MotionUtils::
computePoseDifference(const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rj)
{
	int n = Ri.cols()/3;
	double d = 0.0;
	for(int i=0;i<n;i++)
	{
		Eigen::AngleAxisd aa(Ri.block<3,3>(0,i*3).transpose()*Rj.block<3,3>(0,i*3));
		//ToDO : add weights
		d += aa.angle();
	}

	return d;
}
Eigen::MatrixXd
MotionUtils::
computePoseDisplacement(const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rj)
{
	int n = Ri.cols()/3;
	Eigen::MatrixXd d(4,n);
	for(int i=0;i<n;i++)
	{
		Eigen::Quaterniond q(Ri.block<3,3>(0,i*3).transpose()*Rj.block<3,3>(0,i*3));
		d(0,i) = q.w();
		d.block<3,1>(1,i) = q.vec();
	}

	return d;
}
Eigen::MatrixXd
MotionUtils::
addDisplacement(const Eigen::MatrixXd& R, const Eigen::MatrixXd& d)
{
	int n = R.cols()/3;
	Eigen::MatrixXd Rd = R;
	for(int i=0;i<n;i++)
	{
		Eigen::Vector4d d_i = d.col(i);
		if(d_i[1]>1.0-1e-6)
			continue;

		Eigen::Quaterniond q(d_i[0],d_i[1],d_i[2],d_i[3]);
		Rd.block<3,3>(0,i*3) = R.block<3,3>(0,i*3)*q.toRotationMatrix();
	}

	return Rd;
}
double
MotionUtils::
easeInEaseOut(double x, double yp0, double yp1)
{
	double y = (x-1.0)*((yp0+yp1+2)*x*x - (yp0 + 1)*x - 1.0);
	// double y = 2*x*x*x - 3*x*x + 1.0;
	return y;
	// return std::max(0.0,std::min(1.0,y));
}
Eigen::Isometry3d
MotionUtils::
getReferenceTransform(const Eigen::Vector3d& pos, const Eigen::MatrixXd& rot)
{

	Eigen::Vector3d z = rot.col(2);
	Eigen::Vector3d p = pos;
	Eigen::Vector3d y = Eigen::Vector3d::UnitY();
	z -= MathUtils::projectOnVector(z, y);
	p -= MathUtils::projectOnVector(p, y);

	z.normalize();
	Eigen::Vector3d x = y.cross(z);

	Eigen::Isometry3d T_ref;

	T_ref.linear().col(0) = x;
	T_ref.linear().col(1) = y;
	T_ref.linear().col(2) = z;

	T_ref.translation() = p;

	return T_ref;
}

Motion*
MotionUtils::
blendUpperLowerMotion(BVH* bvh_lb, BVH* bvh_ub, int start_lb, int start_ub)
{
	Motion* motion = new Motion(bvh_lb);

	int nf = bvh_lb->getNumFrames();
	auto parents = bvh_lb->getParents();
	
	Eigen::Isometry3d T_lb = getReferenceTransform(bvh_lb->getPosition(start_lb),
													bvh_lb->getRotation(start_lb));
	Eigen::Isometry3d T_ub = getReferenceTransform(bvh_ub->getPosition(start_ub),
													bvh_ub->getRotation(start_ub));
	Eigen::Isometry3d T_diff = T_ub*(T_lb.inverse());
	for(int i=0;i<nf-start_ub;i++)
	{
		Eigen::Vector3d pos = bvh_lb->getPosition(start_lb+i);
		Eigen::MatrixXd rot_lb = bvh_lb->getRotation(start_lb+i);
		Eigen::MatrixXd rot = bvh_ub->getRotation(start_ub);
		int lf = bvh_lb->getNodeIndex("simLeftFoot");
		int rf = bvh_lb->getNodeIndex("simRightFoot");
		rot.block<3,3>(0,0) = rot_lb.block<3,3>(0,0);

		rot.block<3,3>(0,lf*3) = rot_lb.block<3,3>(0,lf*3);lf = parents[lf];
		rot.block<3,3>(0,rf*3) = rot_lb.block<3,3>(0,rf*3);rf = parents[rf];

		rot.block<3,3>(0,lf*3) = rot_lb.block<3,3>(0,lf*3);lf = parents[lf];
		rot.block<3,3>(0,rf*3) = rot_lb.block<3,3>(0,rf*3);rf = parents[rf];

		rot.block<3,3>(0,lf*3) = rot_lb.block<3,3>(0,lf*3);
		rot.block<3,3>(0,rf*3) = rot_lb.block<3,3>(0,rf*3);

		// Eigen::Isometry3d T = getReferenceTransform(pos, rot);

		pos = T_diff.linear()*pos + T_diff.translation();

		rot.block<3,3>(0,0) = T_diff.linear()*rot.block<3,3>(0,0);

		// pos = T_ub.translation();
		// rot.block<3,3>(0,0) = T_ub.linear();
		// pos += T.linear()*T_diff.translation();
		// T.linear() = T.linear()*T_diff.linear();
		motion->append(pos, rot, false);
	}
	motion->computeVelocity();
	return motion;
}
Motion*
MotionUtils::
parseMotionLabel(const std::string& line, int fps)
{
	std::stringstream ss(line);

	std::string bvh_name;
	std::string start_str;
	std::string end_str;
	ss>>bvh_name>>start_str>>end_str;
	int start,end;

	if(start_str.find(":") != std::string::npos)
	{
		std::stringstream ss1(start_str);
		std::stringstream ss2(end_str);
		std::string word;
		int minute, second, frame;
		std::getline(ss1, word, ':');
		minute = std::stoi(word);
		std::getline(ss1, word, ':');
		second = std::stoi(word);
		std::getline(ss1, word, ':');
		frame = std::stoi(word);

		start = (minute*60+second)*fps + frame;

		std::getline(ss2, word, ':');
		minute = std::stoi(word);
		std::getline(ss2, word, ':');
		second = std::stoi(word);
		std::getline(ss2, word, ':');
		frame = std::stoi(word);

		end = (minute*60+second)*fps + frame;
	}
	else
	{
		start = std::stoi(start_str);
		end = std::stoi(end_str);
	}
	
	bvh_name = std::string(ROOT_DIR) + "/data/bvh/" + bvh_name;
	std::cout<<"Parse "<<bvh_name<<" ["<<start<<" : "<<end<<") "<<std::endl;
	BVH* bvh = new BVH(bvh_name);
	Motion* motion = new Motion(bvh);
	for(int j=start;j<end;j++)
		motion->append(bvh->getPosition(j), bvh->getRotation(j),false);
	motion->computeVelocity();

	return motion;
}
