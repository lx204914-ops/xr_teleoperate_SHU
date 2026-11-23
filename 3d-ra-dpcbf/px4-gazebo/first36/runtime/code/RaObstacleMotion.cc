#include <gz/sim/System.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/Util.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <sdf/Element.hh>
#include <fstream>
#include <vector>
#include <mutex>
#include <cmath>
#include <stdexcept>
#include <algorithm>
class RaObstacleMotion final: public gz::sim::System,
 public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
 public gz::sim::ISystemPostUpdate {
 std::uint32_t m=0,n=0;double dt=0,epoch=-1;gz::math::Vector3d offset{0,0,0};
 std::vector<double> data;std::vector<gz::sim::Entity> ids;
 std::mutex mutex;gz::transport::Node node;gz::transport::Node::Publisher actual,status,kinematics;
 public:
 void Configure(const gz::sim::Entity &,const std::shared_ptr<const sdf::Element>&s,
   gz::sim::EntityComponentManager &,gz::sim::EventManager &) override {
   std::ifstream f(s->Get<std::string>("data_file"),std::ios::binary);
   f.read(reinterpret_cast<char*>(&m),4);f.read(reinterpret_cast<char*>(&n),4);f.read(reinterpret_cast<char*>(&dt),8);
   if(!f||m<2||n<1||n>200||dt<=0)throw std::runtime_error("Invalid frozen obstacle tape");
   data.resize(std::size_t(m)*n*10);f.read(reinterpret_cast<char*>(data.data()),data.size()*8);
   if(!f)throw std::runtime_error("Truncated obstacle tape");
   offset=s->Get<gz::math::Vector3d>("offset");ids.resize(n,gz::sim::kNullEntity);
   node.Subscribe("/ra/motion_epoch",&RaObstacleMotion::Command,this);
   actual=node.Advertise<gz::msgs::Pose_V>("/ra/obstacle_actual");
   status=node.Advertise<gz::msgs::Double_V>("/ra/motion_status");
   kinematics=node.Advertise<gz::msgs::Double_V>("/ra/obstacle_kinematics");
 }
 void Command(const gz::msgs::Double_V &msg){
   if(msg.data_size()!=4)return;
   for(int i=0;i<4;i++)if(!std::isfinite(msg.data(i)))return;
   std::lock_guard<std::mutex> lock(mutex);epoch=msg.data(0);offset.Set(msg.data(1),msg.data(2),msg.data(3));
 }
 void PreUpdate(const gz::sim::UpdateInfo &info,gz::sim::EntityComponentManager &ecm) override {
   if(info.paused)return;
   {std::lock_guard<std::mutex> lock(mutex);gz::msgs::Double_V s;s.add_data(epoch);s.add_data(offset.X());s.add_data(offset.Y());s.add_data(offset.Z());status.Publish(s);}
   if(ids[0]==gz::sim::kNullEntity){
     ecm.Each<gz::sim::components::Model,gz::sim::components::Name>(
       [&](const gz::sim::Entity &id,const gz::sim::components::Model*,const gz::sim::components::Name*name){
         for(std::uint32_t j=0;j<n;j++)if(name->Data()=="ra_obstacle_"+std::to_string(j))ids[j]=id;return true;});
   }
   double ep;gz::math::Vector3d off;
   {std::lock_guard<std::mutex> lock(mutex);ep=epoch;off=offset;}
   double sim=std::chrono::duration<double>(info.simTime).count();
   double t=ep<0?0:std::max(0.,sim-ep);t=std::min(t,(m-1)*dt);
   auto k=std::min<std::uint32_t>(static_cast<std::uint32_t>(std::floor((t+1e-10)/dt)),m-2);
   double ds=std::min(dt,std::max(0.,t-k*dt));
   for(std::uint32_t j=0;j<n;j++){
     if(ids[j]==gz::sim::kNullEntity)continue;
     const double *a=&data[(std::size_t(k)*n+j)*10],*b=&data[(std::size_t(k+1)*n+j)*10];
     gz::math::Vector3d p=off;
     for(int l=0;l<3;l++){double jerk=(b[l+6]-a[l+6])/dt;p[l]+=a[l]+ds*a[l+3]+.5*ds*ds*a[l+6]+ds*ds*ds*jerk/6;}
     ecm.SetComponentData<gz::sim::components::WorldPoseCmd>(ids[j],gz::math::Pose3d(p,gz::math::Quaterniond::Identity));
   }
 }
 void PostUpdate(const gz::sim::UpdateInfo &info,const gz::sim::EntityComponentManager &ecm) override {
   if(info.paused)return;
   gz::msgs::Pose_V msg;auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
   msg.mutable_header()->mutable_stamp()->set_sec(ns/1000000000);msg.mutable_header()->mutable_stamp()->set_nsec(ns%1000000000);

   double ep;{std::lock_guard<std::mutex>lock(mutex);ep=epoch;}
   double sim=std::chrono::duration<double>(info.simTime).count();bool moving=ep>=0&&sim>=ep&&sim-ep<(m-1)*dt;
   double t=ep<0?0:std::clamp(sim-ep,0.,(m-1)*dt);auto k=std::min<std::uint32_t>(std::uint32_t(std::floor((t+1e-10)/dt)),m-2);double ds=std::clamp(t-k*dt,0.,dt);
   gz::msgs::Double_V km;km.add_data(sim);km.add_data(n);
   for(std::uint32_t j=0;j<n;j++)if(ids[j]!=gz::sim::kNullEntity){
     auto pose=gz::sim::worldPose(ids[j],ecm);auto *p=msg.add_pose();p->set_name("ra_obstacle_"+std::to_string(j));p->set_id(ids[j]);
     p->mutable_position()->set_x(pose.Pos().X());p->mutable_position()->set_y(pose.Pos().Y());p->mutable_position()->set_z(pose.Pos().Z());

     p->mutable_orientation()->set_w(1.);
     const double*a=&data[(std::size_t(k)*n+j)*10],*b=&data[(std::size_t(k+1)*n+j)*10];
     for(int l=0;l<3;l++)km.add_data(pose.Pos()[l]);
     for(int l=0;l<3;l++){double jerk=(b[l+6]-a[l+6])/dt;km.add_data(moving?a[l+3]+ds*a[l+6]+.5*ds*ds*jerk:0.);}
     for(int l=0;l<3;l++)km.add_data(moving?a[l+6]+ds*(b[l+6]-a[l+6])/dt:0.);
     km.add_data(a[9]);

   }
   actual.Publish(msg);if(km.data_size()==2+10*int(n))kinematics.Publish(km);
 }
};
GZ_ADD_PLUGIN(RaObstacleMotion,gz::sim::System,gz::sim::ISystemConfigure,gz::sim::ISystemPreUpdate,gz::sim::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(RaObstacleMotion,"RaObstacleMotion")

// Direct physics components avoid asynchronous pose-message double differences.
class RaVehicleObserver final: public gz::sim::System,
 public gz::sim::ISystemConfigure,public gz::sim::ISystemPreUpdate,public gz::sim::ISystemPostUpdate {
 gz::sim::Entity link=gz::sim::kNullEntity;gz::transport::Node node;gz::transport::Node::Publisher pub;
 public:
 void Configure(const gz::sim::Entity &,const std::shared_ptr<const sdf::Element>&,
   gz::sim::EntityComponentManager &,gz::sim::EventManager &) override {pub=node.Advertise<gz::msgs::Double_V>("/ra/vehicle_truth");}
 void PreUpdate(const gz::sim::UpdateInfo &,gz::sim::EntityComponentManager &ecm) override {
   if(link!=gz::sim::kNullEntity)return;
   ecm.Each<gz::sim::components::Link,gz::sim::components::Name>(
     [&](const gz::sim::Entity &id,const gz::sim::components::Link*,const gz::sim::components::Name*n){
       if(n->Data()=="base_link"){link=id;gz::sim::Link l(id);l.EnableVelocityChecks(ecm);l.EnableAccelerationChecks(ecm);return false;}return true;});
 }
 void PostUpdate(const gz::sim::UpdateInfo &info,const gz::sim::EntityComponentManager &ecm) override {
   if(info.paused||link==gz::sim::kNullEntity)return;
   gz::sim::Link l(link);auto v=l.WorldLinearVelocity(ecm),a=l.WorldLinearAcceleration(ecm);auto p=l.WorldPose(ecm);
   if(!v||!a||!p)return;
   gz::msgs::Double_V msg;msg.add_data(std::chrono::duration<double>(info.simTime).count());
   for(int i=0;i<3;i++)msg.add_data(p->Pos()[i]);for(int i=0;i<3;i++)msg.add_data((*v)[i]);for(int i=0;i<3;i++)msg.add_data((*a)[i]);
   msg.add_data(p->Rot().X());msg.add_data(p->Rot().Y());msg.add_data(p->Rot().Z());msg.add_data(p->Rot().W());pub.Publish(msg);
 }
};
GZ_ADD_PLUGIN(RaVehicleObserver,gz::sim::System,gz::sim::ISystemConfigure,gz::sim::ISystemPreUpdate,gz::sim::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(RaVehicleObserver,"RaVehicleObserver")
